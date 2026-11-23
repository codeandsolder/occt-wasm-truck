//! Host functions for the embedded module's imports.
//!
//! `-sSTANDALONE_WASM` still leaves eight `wasi_snapshot_preview1` calls and a
//! few Emscripten runtime hooks and filesystem syscalls unresolved. OCCT links
//! them in, but the facade never needs a filesystem, an environment, or a call
//! stack, so each one gets the answer an empty sandbox would give. Output to
//! stdout and stderr is accepted and discarded.

use std::time::{SystemTime, UNIX_EPOCH};

use wasmtime::{
    Caller, Extern, Linker, Memory, ResourceLimiter, Store, TypedFunc, WasmParams, WasmResults,
};

use crate::error::{MemoryLimitExceeded, OcctError, OcctResult};

/// The data a kernel's `Store` carries. wasmtime asks the limiter before every
/// memory or table growth, and can only reach one stored here.
pub struct HostState {
    limiter: Box<dyn ResourceLimiter + Sync>,
    poisoned: bool,
}

impl HostState {
    pub fn new(limiter: Box<dyn ResourceLimiter + Sync>) -> Self {
        Self {
            limiter,
            poisoned: false,
        }
    }

    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

/// Every call into a live guest must pass through this boundary. A trapped
/// call may have left native OCCT state inconsistent.
pub trait GuestCall<P, R> {
    fn call_guest(&self, store: &mut Store<HostState>, params: P) -> OcctResult<R>;
}

impl<P: WasmParams, R: WasmResults> GuestCall<P, R> for TypedFunc<P, R> {
    fn call_guest(&self, store: &mut Store<HostState>, params: P) -> OcctResult<R> {
        if store.data().is_poisoned() {
            return Err(OcctError::Invalidated);
        }
        self.call(&mut *store, params).map_err(|error| {
            store.data_mut().poisoned = true;
            error.into()
        })
    }
}

// OCCT does not check its allocations, and address 0 is writable in wasm, so a
// refused growth that only made malloc return NULL would corrupt the kernel
// instead of failing it. Every refusal traps the call that needed the memory.
impl ResourceLimiter for HostState {
    fn memory_growing(
        &mut self,
        current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        if self.limiter.memory_growing(current, desired, maximum)? {
            Ok(true)
        } else {
            Err(wasmtime::Error::new(MemoryLimitExceeded {
                current,
                requested: desired,
            }))
        }
    }

    fn memory_grow_failed(&mut self, error: wasmtime::Error) -> wasmtime::Result<()> {
        self.limiter.memory_grow_failed(error)?;
        Err(wasmtime::format_err!("linear memory could not grow"))
    }

    fn table_growing(
        &mut self,
        current: usize,
        desired: usize,
        maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        self.limiter.table_growing(current, desired, maximum)
    }

    fn table_grow_failed(&mut self, error: wasmtime::Error) -> wasmtime::Result<()> {
        self.limiter.table_grow_failed(error)
    }

    fn instances(&self) -> usize {
        self.limiter.instances()
    }

    fn tables(&self) -> usize {
        self.limiter.tables()
    }

    fn memories(&self) -> usize {
        self.limiter.memories()
    }
}

const WASI: &str = "wasi_snapshot_preview1";

// WASI errno values; Emscripten's musl uses the same numbering.
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_NOSYS: i32 = 52;

const STDOUT: i32 = 1;
const STDERR: i32 = 2;

fn memory(caller: &mut Caller<'_, HostState>) -> wasmtime::Result<Memory> {
    caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| wasmtime::Error::msg("module has no memory export"))
}

/// Get an address range without allowing any arithmetic to wrap.
fn guest_region(data: &[u8], address: u32, length: usize) -> Option<&[u8]> {
    let start = address as usize;
    data.get(start..start.checked_add(length)?)
}

const ERRNO_FAULT: i32 = 21;
const ERRNO_OVERFLOW: i32 = 61;

/// Discard stdout/stderr but validate the exact WASI ciovec layout and all
/// referenced ranges, returning the count only if the whole write is valid.
fn discard_iovecs(data: &mut [u8], iovs: u32, count: u32, written: u32) -> i32 {
    let Some(table_size) = (count as usize).checked_mul(8) else {
        return ERRNO_OVERFLOW;
    };
    let Some(table) = guest_region(data, iovs, table_size) else {
        return ERRNO_FAULT;
    };
    let (descriptors, []) = table.as_chunks::<8>() else {
        return ERRNO_FAULT;
    };
    let mut total = 0u32;
    for descriptor in descriptors {
        let ([address, length], []) = descriptor.as_chunks::<4>() else {
            return ERRNO_FAULT;
        };
        let length = u32::from_le_bytes(*length);
        if guest_region(data, u32::from_le_bytes(*address), length as usize).is_none() {
            return ERRNO_FAULT;
        }
        let Some(sum) = total.checked_add(length) else {
            return ERRNO_OVERFLOW;
        };
        total = sum;
    }
    let start = written as usize;
    let Some(end) = start.checked_add(4) else {
        return ERRNO_FAULT;
    };
    let Some(destination) = data.get_mut(start..end) else {
        return ERRNO_FAULT;
    };
    destination.copy_from_slice(&total.to_le_bytes());
    ERRNO_SUCCESS
}

fn write_bytes(
    memory: Memory,
    caller: &mut Caller<'_, HostState>,
    ptr: i32,
    bytes: &[u8],
) -> wasmtime::Result<()> {
    memory.write(caller, ptr.cast_unsigned() as usize, bytes)?;
    Ok(())
}

pub fn define_imports(linker: &mut Linker<HostState>) -> OcctResult<()> {
    define_wasi(linker)?;
    define_env(linker)?;
    Ok(())
}

fn define_wasi(linker: &mut Linker<HostState>) -> OcctResult<()> {
    linker.func_wrap(WASI, "proc_exit", |code: i32| -> wasmtime::Result<()> {
        Err(wasmtime::Error::msg(format!(
            "module called proc_exit({code})"
        )))
    })?;

    // Report every byte as written so printf-style callers do not retry.
    linker.func_wrap(
        WASI,
        "fd_write",
        |mut caller: Caller<'_, HostState>,
         fd: i32,
         iovs: u32,
         iovs_len: u32,
         nwritten: u32|
         -> wasmtime::Result<i32> {
            if fd != STDOUT && fd != STDERR {
                return Ok(ERRNO_BADF);
            }
            let memory = memory(&mut caller)?;
            Ok(discard_iovecs(
                memory.data_mut(&mut caller),
                iovs,
                iovs_len,
                nwritten,
            ))
        },
    )?;
    linker.func_wrap(WASI, "fd_read", |_: i32, _: i32, _: i32, _: i32| ERRNO_BADF)?;
    linker.func_wrap(WASI, "fd_seek", |_: i32, _: i64, _: i32, _: i32| ERRNO_BADF)?;
    linker.func_wrap(WASI, "fd_close", |_: i32| ERRNO_BADF)?;

    // An empty environment: zero variables, zero bytes.
    linker.func_wrap(
        WASI,
        "environ_sizes_get",
        |mut caller: Caller<'_, HostState>, count: i32, buf_size: i32| -> wasmtime::Result<i32> {
            let memory = memory(&mut caller)?;
            write_bytes(memory, &mut caller, count, &0u32.to_le_bytes())?;
            write_bytes(memory, &mut caller, buf_size, &0u32.to_le_bytes())?;
            Ok(ERRNO_SUCCESS)
        },
    )?;
    linker.func_wrap(WASI, "environ_get", |_: i32, _: i32| ERRNO_SUCCESS)?;

    linker.func_wrap(
        WASI,
        "clock_time_get",
        |mut caller: Caller<'_, HostState>,
         _clock: i32,
         _precision: i64,
         time: i32|
         -> wasmtime::Result<i32> {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
            let memory = memory(&mut caller)?;
            write_bytes(memory, &mut caller, time, &nanos.to_le_bytes())?;
            Ok(ERRNO_SUCCESS)
        },
    )?;
    Ok(())
}

fn define_env(linker: &mut Linker<HostState>) -> OcctResult<()> {
    linker.func_wrap("env", "emscripten_notify_memory_growth", |_: i32| {})?;
    linker.func_wrap(
        "env",
        "emscripten_get_callstack",
        |_: i32, _: i32, _: i32| 0i32,
    )?;
    linker.func_wrap(
        "env",
        "emscripten_get_preloaded_image_data",
        |_: i32, _: i32, _: i32| 0i32,
    )?;
    linker.func_wrap("env", "OSD_MemInfo_getModuleHeapLength", || 0.0f64)?;
    linker.func_wrap("env", "_emscripten_lookup_name", |_: i32| 0i32)?;

    // Emscripten syscalls return a negated errno.
    linker.func_wrap(
        "env",
        "__syscall_faccessat",
        |_: i32, _: i32, _: i32, _: i32| -ERRNO_NOSYS,
    )?;
    linker.func_wrap("env", "__syscall_chmod", |_: i32, _: i32| -ERRNO_NOSYS)?;
    linker.func_wrap("env", "__syscall_getdents64", |_: i32, _: i32, _: i32| {
        -ERRNO_NOSYS
    })?;
    linker.func_wrap("env", "__syscall_unlinkat", |_: i32, _: i32, _: i32| {
        -ERRNO_NOSYS
    })?;
    linker.func_wrap("env", "__syscall_rmdir", |_: i32| -ERRNO_NOSYS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ERRNO_FAULT, ERRNO_OVERFLOW, ERRNO_SUCCESS, GuestCall, HostState, discard_iovecs};
    use crate::{OcctError, OcctResult};
    use wasmtime::{Engine, Instance, Module, Store, StoreLimits, Val};

    fn descriptor(data: &mut [u8], index: usize, address: u32, length: u32) {
        let entry = &mut data[index * 8..][..8];
        entry[..4].copy_from_slice(&address.to_le_bytes());
        entry[4..].copy_from_slice(&length.to_le_bytes());
    }

    #[test]
    fn discarded_output_checks_table_buffers_and_destination() {
        let mut data = [0_u8; 64];
        descriptor(&mut data, 0, 32, 7);
        descriptor(&mut data, 1, 63, 1);
        assert_eq!(discard_iovecs(&mut data, 0, 2, 24), ERRNO_SUCCESS);
        assert_eq!(&data[24..28], &8_u32.to_le_bytes());
        descriptor(&mut data, 1, 63, 2);
        assert_eq!(discard_iovecs(&mut data, 0, 2, 24), ERRNO_FAULT);
        assert_eq!(&data[24..28], &8_u32.to_le_bytes());
        assert_eq!(discard_iovecs(&mut data, u32::MAX - 3, 2, 24), ERRNO_FAULT);
        assert_ne!(discard_iovecs(&mut data, 0, u32::MAX, 24), ERRNO_SUCCESS);
        assert_eq!(discard_iovecs(&mut data, 0, 1, 62), ERRNO_FAULT);
        assert_eq!(discard_iovecs(&mut data, 64, 0, 24), ERRNO_SUCCESS);
        assert_eq!(&data[24..28], &0_u32.to_le_bytes());
    }

    #[test]
    fn discarded_output_rejects_total_length_overflow() {
        let count = 65_536_u32;
        let mut data = vec![0_u8; count as usize * 8 + 4];
        for index in 0..count as usize {
            descriptor(&mut data, index, 0, count);
        }
        assert_eq!(
            discard_iovecs(&mut data, 0, count, count * 8),
            ERRNO_OVERFLOW
        );
    }

    #[test]
    fn guest_trap_prevents_subsequent_execution() -> OcctResult<()> {
        let engine = Engine::default();
        let mut store = Store::new(&engine, HostState::new(Box::new(StoreLimits::default())));
        let module = Module::new(
            &engine,
            r#"(module
          (global (export "counter") (mut i32) (i32.const 0))
          (func (export "fail") global.get 0 i32.const 1 i32.add global.set 0 unreachable)
          (func (export "next") global.get 0 i32.const 1 i32.add global.set 0)
        )"#,
        )?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let fail = instance.get_typed_func::<(), ()>(&mut store, "fail")?;
        let next = instance.get_typed_func::<(), ()>(&mut store, "next")?;
        assert!(matches!(
            fail.call_guest(&mut store, ()),
            Err(OcctError::Runtime(_))
        ));
        assert!(matches!(
            next.call_guest(&mut store, ()),
            Err(OcctError::Invalidated)
        ));
        let counter = instance
            .get_global(&mut store, "counter")
            .ok_or_else(|| OcctError::Memory("missing test global".to_owned()))?;
        assert!(matches!(counter.get(&mut store), Val::I32(1)));
        Ok(())
    }
}
