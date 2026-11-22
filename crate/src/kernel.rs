//! WASM host implementation for the OCCT kernel.
//!
//! Manages the wasmtime `Engine`, `Store`, and `Instance`, and provides
//! helper methods for memory transfer across the WASM boundary.

use wasmtime::{
    Engine, Instance, Linker, Memory, Module, ResourceLimiter, Store, StoreLimits, TypedFunc,
};

use crate::error::{OcctError, OcctResult};
use crate::host::HostState;
use crate::kernel_generated::GeneratedFuncs;
use crate::types::{
    BoundingBox, CurveData, CurveGeometry, EdgeData, EvolutionData, LabelInfo, Mesh, MeshBatch,
    NurbsCurveData, NurbsSurfaceData, ProjectionData, ShapeHandle, SurfaceData, SurfaceFrame, Vec3,
};

/// Brotli-compressed WASM binary, embedded at compile time.
///
/// The uncompressed binary is ~21 MB; brotli brings it to ~4 MB.
/// [`OcctKernel::compile()`] decompresses it.
static WASM_BINARY: &[u8] = include_bytes!("occt-wasm.wasm.br");

/// A WASI result buffer is exposed as a pointer plus an element count.
struct BufferExport<C> {
    address: TypedFunc<(), u32>,
    count: TypedFunc<(), C>,
}

impl<C: wasmtime::WasmResults + TryInto<u32>> BufferExport<C> {
    fn resolve(
        instance: &Instance,
        store: &mut Store<HostState>,
        address: &str,
    ) -> OcctResult<Self> {
        Self::resolve_with_count(instance, store, address, &format!("{address}_len"))
    }

    fn resolve_with_count(
        instance: &Instance,
        store: &mut Store<HostState>,
        address: &str,
        count: &str,
    ) -> OcctResult<Self> {
        Ok(Self {
            address: instance.get_typed_func(&mut *store, address)?,
            count: instance.get_typed_func(&mut *store, count)?,
        })
    }

    fn query(&self, store: &mut Store<HostState>) -> OcctResult<(u32, u32)> {
        let address = self.address.call(&mut *store, ())?;
        let count = checked_count(self.count.call(&mut *store, ())?)?;
        Ok((address, count))
    }
}

/// Scalars that can be decoded from the little-endian WASM result buffers.
trait GuestElement: Sized {
    fn decode(
        memory: Memory,
        store: &Store<HostState>,
        address: u32,
        count: u32,
    ) -> OcctResult<Vec<Self>>;
}

macro_rules! guest_elements {
    ($($ty:ty),+ $(,)?) => {$(
        impl GuestElement for $ty {
            fn decode(
                memory: Memory,
                store: &Store<HostState>,
                address: u32,
                count: u32,
            ) -> OcctResult<Vec<Self>> {
                read_array(memory, store, address, count, Self::from_le_bytes)
            }
        }
    )+};
}
guest_elements!(i32, u32, f32, f64);

struct ResultReader<'a> {
    memory: Memory,
    store: &'a mut Store<HostState>,
}

impl ResultReader<'_> {
    fn read<T: GuestElement, C: wasmtime::WasmResults + TryInto<u32>>(
        &mut self,
        source: &BufferExport<C>,
    ) -> OcctResult<Vec<T>> {
        let (address, count) = source.query(&mut *self.store)?;
        T::decode(self.memory, self.store, address, count)
    }
}

/// Input buffers allocated for one WASM call. The generated number of
/// heap arguments determines the stack capacity; no heap bookkeeping needed.
pub(super) struct GuestBuffers<const N: usize> {
    pointers: [u32; N],
    used: usize,
}

impl<const N: usize> GuestBuffers<N> {
    pub(super) fn write(&mut self, kernel: &mut OcctKernel, data: &[u8]) -> OcctResult<u32> {
        let slot = self
            .pointers
            .get_mut(self.used)
            .ok_or_else(|| OcctError::Memory("too many guest allocations".to_owned()))?;
        let ptr = kernel
            .fn_alloc
            .call(&mut kernel.store, checked_wasm_len(data.len(), 1)?)?;
        if ptr == 0 {
            core::hint::cold_path();
            return Err(OcctError::Memory("allocation failed".to_owned()));
        }
        *slot = ptr;
        self.used += 1;
        let mem = kernel.memory.data_mut(&mut kernel.store);
        let start = ptr as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| OcctError::Memory("write range overflow".to_owned()))?;
        let destination = mem
            .get_mut(start..end)
            .ok_or_else(|| OcctError::Memory("write out of bounds".to_owned()))?;
        destination.copy_from_slice(data);
        Ok(ptr)
    }
}

/// The OCCT CAD kernel, backed by a sandboxed WASM module.
///
/// Create an instance with [`OcctKernel::new()`], then call methods to
/// create and manipulate shapes. All shapes live in an arena inside the
/// WASM module and are referenced by [`ShapeHandle`].
///
/// # Example
///
/// ```no_run
/// use occt_wasm::{OcctKernel, ShapeHandle};
///
/// let mut kernel = OcctKernel::new().unwrap();
/// let box_shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
/// let volume = kernel.get_volume(box_shape).unwrap();
/// assert!((volume - 6000.0).abs() < 1.0);
/// ```
pub struct OcctKernel {
    pub(crate) store: Store<HostState>,
    instance: Instance,
    memory: Memory,

    // Lifecycle + error functions
    fn_has_error: TypedFunc<(), i32>,
    buffer_error: BufferExport<u32>,

    // Memory management
    fn_alloc: TypedFunc<u32, u32>,
    fn_free: TypedFunc<u32, ()>,

    // Result buffer accessors
    buffer_string_result: BufferExport<u32>,
    buffer_vec_u32_result: BufferExport<u32>,
    buffer_vec_f64_result: BufferExport<u32>,
    buffer_vec_i32_result: BufferExport<u32>,

    // BBox accessors
    fn_get_bbox_xmin: TypedFunc<(), f64>,
    fn_get_bbox_ymin: TypedFunc<(), f64>,
    fn_get_bbox_zmin: TypedFunc<(), f64>,
    fn_get_bbox_xmax: TypedFunc<(), f64>,
    fn_get_bbox_ymax: TypedFunc<(), f64>,
    fn_get_bbox_zmax: TypedFunc<(), f64>,

    // Mesh accessors
    buffer_mesh_positions: BufferExport<i32>,
    buffer_mesh_normals: BufferExport<i32>,
    buffer_mesh_indices: BufferExport<i32>,
    buffer_mesh_face_groups: BufferExport<i32>,

    // MeshBatch accessors
    buffer_mesh_batch_positions: BufferExport<i32>,
    buffer_mesh_batch_normals: BufferExport<i32>,
    buffer_mesh_batch_indices: BufferExport<i32>,
    buffer_mesh_batch_shape_offsets: BufferExport<i32>,

    // Edge accessors
    buffer_edge_points: BufferExport<i32>,
    buffer_edge_groups: BufferExport<i32>,

    // NURBS accessors
    fn_get_nurbs_degree: TypedFunc<(), i32>,
    fn_get_nurbs_rational: TypedFunc<(), i32>,
    fn_get_nurbs_periodic: TypedFunc<(), i32>,
    buffer_nurbs_knots: BufferExport<u32>,
    buffer_nurbs_multiplicities: BufferExport<u32>,
    buffer_nurbs_poles: BufferExport<u32>,
    buffer_nurbs_weights: BufferExport<u32>,

    // Curve geometry accessors
    buffer_curve_kind: BufferExport<u32>,
    fn_get_curve_first: TypedFunc<(), f64>,
    fn_get_curve_last: TypedFunc<(), f64>,
    buffer_curve_params: BufferExport<u32>,
    fn_get_curve_has_bspline: TypedFunc<(), i32>,

    // Surface geometry accessors
    buffer_surface_kind: BufferExport<u32>,
    fn_get_surface_is_direct: TypedFunc<(), i32>,
    buffer_surface_params: BufferExport<u32>,
    fn_get_surface_has_bspline: TypedFunc<(), i32>,
    fn_get_surface_degree_u: TypedFunc<(), i32>,
    fn_get_surface_degree_v: TypedFunc<(), i32>,
    fn_get_surface_nb_poles_u: TypedFunc<(), i32>,
    fn_get_surface_nb_poles_v: TypedFunc<(), i32>,
    fn_get_surface_periodic_u: TypedFunc<(), i32>,
    fn_get_surface_periodic_v: TypedFunc<(), i32>,
    buffer_surface_knots_u: BufferExport<u32>,
    buffer_surface_knots_v: BufferExport<u32>,
    buffer_surface_multiplicities_u: BufferExport<u32>,
    buffer_surface_multiplicities_v: BufferExport<u32>,
    buffer_surface_poles: BufferExport<u32>,
    buffer_surface_weights: BufferExport<u32>,

    // Evolution accessors
    fn_get_evo_result_id: TypedFunc<(), u32>,
    buffer_evo_modified: BufferExport<u32>,
    buffer_evo_generated: BufferExport<u32>,
    buffer_evo_deleted: BufferExport<u32>,
    buffer_evo_edge_modified: BufferExport<u32>,
    buffer_evo_edge_generated: BufferExport<u32>,
    buffer_evo_edge_deleted: BufferExport<u32>,
    buffer_evo_first_shape: BufferExport<u32>,
    buffer_evo_last_shape: BufferExport<u32>,
    buffer_evo_spine_generated: BufferExport<u32>,

    // Projection accessors
    fn_get_proj_visible_outline: TypedFunc<(), u32>,
    fn_get_proj_visible_smooth: TypedFunc<(), u32>,
    fn_get_proj_visible_sharp: TypedFunc<(), u32>,
    fn_get_proj_hidden_outline: TypedFunc<(), u32>,
    fn_get_proj_hidden_smooth: TypedFunc<(), u32>,
    fn_get_proj_hidden_sharp: TypedFunc<(), u32>,

    // Label info accessors
    fn_get_label_info_label_id: TypedFunc<(), i32>,
    buffer_label_info_name: BufferExport<u32>,
    fn_get_label_info_has_color: TypedFunc<(), i32>,
    fn_get_label_info_r: TypedFunc<(), f64>,
    fn_get_label_info_g: TypedFunc<(), f64>,
    fn_get_label_info_b: TypedFunc<(), f64>,
    fn_get_label_info_is_assembly: TypedFunc<(), i32>,
    fn_get_label_info_is_component: TypedFunc<(), i32>,
    fn_get_label_info_shape_id: TypedFunc<(), u32>,

    // Generated method handles
    pub(super) generated: GeneratedFuncs,
}

impl OcctKernel {
    /// Create a new OCCT kernel instance.
    ///
    /// Decompresses and compiles the embedded WASM binary, then initializes
    /// the OCCT runtime. Compiling is most of that cost; to pay it once for
    /// many kernels, or once across processes, see [`OcctKernel::compile()`].
    ///
    /// # Errors
    /// Returns an error if the embedded module cannot be compiled or initialized.
    pub fn new() -> OcctResult<Self> {
        let engine = Engine::new(&Self::engine_config())?;
        Self::from_module(&Self::compile(&engine)?)
    }

    /// The `wasmtime` features the embedded module needs.
    ///
    /// Build the [`Engine`] passed to [`OcctKernel::compile()`] from this,
    /// adding any settings of your own. wasmtime's compilation cache sits
    /// behind its `cache` feature, which this crate leaves off; to use it,
    /// enable `cache` on your own `wasmtime` dependency of the same version.
    #[must_use = "the returned configuration enables the features needed by the OCCT module"]
    pub fn engine_config() -> wasmtime::Config {
        let mut config = wasmtime::Config::new();
        config.wasm_simd(true);
        config.wasm_tail_call(true);
        // The WASM binary uses wasm-opt --experimental-new-eh to convert
        // Emscripten's legacy exceptions to the new (exnref) encoding.
        config.wasm_exceptions(true);
        config
    }

    /// Decompress the embedded WASM binary and compile it for `engine`.
    ///
    /// One [`Module`] creates any number of kernels through
    /// [`OcctKernel::from_module()`], each with its own memory. To skip
    /// compiling in later processes, save it with [`Module::serialize`] and
    /// load it back with [`Module::deserialize`] on an engine built from the
    /// same [`OcctKernel::engine_config()`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use occt_wasm::OcctKernel;
    /// use occt_wasm::wasmtime::Engine;
    ///
    /// let engine = Engine::new(&OcctKernel::engine_config())?;
    /// let module = OcctKernel::compile(&engine)?;
    /// let mut a = OcctKernel::from_module(&module)?;
    /// let mut b = OcctKernel::from_module(&module)?;
    /// # Ok::<(), occt_wasm::OcctError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if decompression or WASM compilation fails.
    pub fn compile(engine: &Engine) -> OcctResult<Module> {
        Ok(Module::new(engine, decompress_wasm()?)?)
    }

    /// Instantiate a kernel from a module made by [`OcctKernel::compile()`].
    ///
    /// # Errors
    /// Returns an error if WASM instantiation, export resolution or initialization fails.
    pub fn from_module(module: &Module) -> OcctResult<Self> {
        Self::from_module_with_limiter(module, StoreLimits::default())
    }

    /// Instantiate a kernel whose memory and table growth `limiter` approves.
    ///
    /// [`StoreLimitsBuilder`](wasmtime::StoreLimitsBuilder) covers the usual
    /// case, a cap on linear memory. The module starts with 128 MiB, so a
    /// smaller cap fails here.
    ///
    /// A refused memory growth traps the call that needed it, which returns
    /// [`OcctError::MemoryLimit`]. After that error the kernel's state is
    /// undefined: drop it, and instantiate a new one from the same module.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use occt_wasm::{OcctError, OcctKernel};
    /// use occt_wasm::wasmtime::{Engine, StoreLimitsBuilder};
    ///
    /// let engine = Engine::new(&OcctKernel::engine_config())?;
    /// let module = OcctKernel::compile(&engine)?;
    /// let limits = StoreLimitsBuilder::new().memory_size(512 << 20).build();
    /// let mut kernel = OcctKernel::from_module_with_limiter(&module, limits)?;
    /// # let step = String::new();
    /// match kernel.import_step(&step) {
    ///     Err(OcctError::MemoryLimit { requested, .. }) => {
    ///         eprintln!("needs more than 512 MiB (asked for {requested} bytes)");
    ///     }
    ///     other => { other?; }
    /// }
    /// # Ok::<(), occt_wasm::OcctError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if instantiation, initialization or export resolution fails,
    /// including if the limiter refuses the module's initial allocation.
    pub fn from_module_with_limiter(
        module: &Module,
        limiter: impl ResourceLimiter + Sync + 'static,
    ) -> OcctResult<Self> {
        let (mut store, instance, memory) = instantiate(module, limiter)?;
        // Resolve all accessor functions
        macro_rules! get_fn {
            ($name:expr => $ret:ty) => {
                instance.get_typed_func::<(), $ret>(&mut store, $name)?
            };
            ($name:expr, $param:ty => $ret:ty) => {
                instance.get_typed_func::<$param, $ret>(&mut store, $name)?
            };
        }

        macro_rules! get_buffer {
            ($name:literal) => {
                BufferExport::resolve(&instance, &mut store, $name)?
            };
        }

        let generated = GeneratedFuncs::resolve(&instance, &mut store)?;

        Ok(Self {
            memory,
            instance,

            fn_has_error: get_fn!("occt_has_error" => i32),
            buffer_error: get_buffer!("occt_get_error"),
            fn_alloc: get_fn!("occt_alloc", u32 => u32),
            fn_free: get_fn!("occt_free", u32 => ()),

            buffer_string_result: get_buffer!("occt_get_string_result"),
            buffer_vec_u32_result: get_buffer!("occt_get_vec_u32_result"),
            buffer_vec_f64_result: get_buffer!("occt_get_vec_f64_result"),
            buffer_vec_i32_result: get_buffer!("occt_get_vec_i32_result"),

            fn_get_bbox_xmin: get_fn!("occt_get_bbox_xmin" => f64),
            fn_get_bbox_ymin: get_fn!("occt_get_bbox_ymin" => f64),
            fn_get_bbox_zmin: get_fn!("occt_get_bbox_zmin" => f64),
            fn_get_bbox_xmax: get_fn!("occt_get_bbox_xmax" => f64),
            fn_get_bbox_ymax: get_fn!("occt_get_bbox_ymax" => f64),
            fn_get_bbox_zmax: get_fn!("occt_get_bbox_zmax" => f64),

            buffer_mesh_positions: get_buffer!("occt_get_mesh_positions"),
            buffer_mesh_normals: get_buffer!("occt_get_mesh_normals"),
            buffer_mesh_indices: get_buffer!("occt_get_mesh_indices"),
            buffer_mesh_face_groups: get_buffer!("occt_get_mesh_face_groups"),

            buffer_mesh_batch_positions: get_buffer!("occt_get_mesh_batch_positions"),
            buffer_mesh_batch_normals: get_buffer!("occt_get_mesh_batch_normals"),
            buffer_mesh_batch_indices: get_buffer!("occt_get_mesh_batch_indices"),
            buffer_mesh_batch_shape_offsets: BufferExport::resolve_with_count(
                &instance,
                &mut store,
                "occt_get_mesh_batch_shape_offsets",
                "occt_get_mesh_batch_shape_count",
            )?,

            buffer_edge_points: get_buffer!("occt_get_edge_points"),
            buffer_edge_groups: get_buffer!("occt_get_edge_groups"),

            fn_get_nurbs_degree: get_fn!("occt_get_nurbs_degree" => i32),
            fn_get_nurbs_rational: get_fn!("occt_get_nurbs_rational" => i32),
            fn_get_nurbs_periodic: get_fn!("occt_get_nurbs_periodic" => i32),
            buffer_nurbs_knots: get_buffer!("occt_get_nurbs_knots"),
            buffer_nurbs_multiplicities: get_buffer!("occt_get_nurbs_multiplicities"),
            buffer_nurbs_poles: get_buffer!("occt_get_nurbs_poles"),
            buffer_nurbs_weights: get_buffer!("occt_get_nurbs_weights"),

            buffer_curve_kind: get_buffer!("occt_get_curve_kind"),
            fn_get_curve_first: get_fn!("occt_get_curve_first" => f64),
            fn_get_curve_last: get_fn!("occt_get_curve_last" => f64),
            buffer_curve_params: get_buffer!("occt_get_curve_params"),
            fn_get_curve_has_bspline: get_fn!("occt_get_curve_has_bspline" => i32),

            buffer_surface_kind: get_buffer!("occt_get_surface_kind"),
            fn_get_surface_is_direct: get_fn!("occt_get_surface_is_direct" => i32),
            buffer_surface_params: get_buffer!("occt_get_surface_params"),
            fn_get_surface_has_bspline: get_fn!("occt_get_surface_has_bspline" => i32),
            fn_get_surface_degree_u: get_fn!("occt_get_surface_degree_u" => i32),
            fn_get_surface_degree_v: get_fn!("occt_get_surface_degree_v" => i32),
            fn_get_surface_nb_poles_u: get_fn!("occt_get_surface_nb_poles_u" => i32),
            fn_get_surface_nb_poles_v: get_fn!("occt_get_surface_nb_poles_v" => i32),
            fn_get_surface_periodic_u: get_fn!("occt_get_surface_periodic_u" => i32),
            fn_get_surface_periodic_v: get_fn!("occt_get_surface_periodic_v" => i32),
            buffer_surface_knots_u: get_buffer!("occt_get_surface_knots_u"),
            buffer_surface_knots_v: get_buffer!("occt_get_surface_knots_v"),
            buffer_surface_multiplicities_u: get_buffer!("occt_get_surface_multiplicities_u"),
            buffer_surface_multiplicities_v: get_buffer!("occt_get_surface_multiplicities_v"),
            buffer_surface_poles: get_buffer!("occt_get_surface_poles"),
            buffer_surface_weights: get_buffer!("occt_get_surface_weights"),

            fn_get_evo_result_id: get_fn!("occt_get_evo_result_id" => u32),
            buffer_evo_modified: get_buffer!("occt_get_evo_modified"),
            buffer_evo_generated: get_buffer!("occt_get_evo_generated"),
            buffer_evo_deleted: get_buffer!("occt_get_evo_deleted"),
            buffer_evo_edge_modified: get_buffer!("occt_get_evo_edge_modified"),
            buffer_evo_edge_generated: get_buffer!("occt_get_evo_edge_generated"),
            buffer_evo_edge_deleted: get_buffer!("occt_get_evo_edge_deleted"),
            buffer_evo_first_shape: get_buffer!("occt_get_evo_first_shape"),
            buffer_evo_last_shape: get_buffer!("occt_get_evo_last_shape"),
            buffer_evo_spine_generated: get_buffer!("occt_get_evo_spine_generated"),

            fn_get_proj_visible_outline: get_fn!("occt_get_proj_visible_outline" => u32),
            fn_get_proj_visible_smooth: get_fn!("occt_get_proj_visible_smooth" => u32),
            fn_get_proj_visible_sharp: get_fn!("occt_get_proj_visible_sharp" => u32),
            fn_get_proj_hidden_outline: get_fn!("occt_get_proj_hidden_outline" => u32),
            fn_get_proj_hidden_smooth: get_fn!("occt_get_proj_hidden_smooth" => u32),
            fn_get_proj_hidden_sharp: get_fn!("occt_get_proj_hidden_sharp" => u32),

            fn_get_label_info_label_id: get_fn!("occt_get_label_info_label_id" => i32),
            buffer_label_info_name: get_buffer!("occt_get_label_info_name"),
            fn_get_label_info_has_color: get_fn!("occt_get_label_info_has_color" => i32),
            fn_get_label_info_r: get_fn!("occt_get_label_info_r" => f64),
            fn_get_label_info_g: get_fn!("occt_get_label_info_g" => f64),
            fn_get_label_info_b: get_fn!("occt_get_label_info_b" => f64),
            fn_get_label_info_is_assembly: get_fn!("occt_get_label_info_is_assembly" => i32),
            fn_get_label_info_is_component: get_fn!("occt_get_label_info_is_component" => i32),
            fn_get_label_info_shape_id: get_fn!("occt_get_label_info_shape_id" => u32),

            generated,
            store,
        })
    }

    // === Memory helpers ===

    /// Execute a guest call with scoped input allocations. A trap invalidates
    /// the kernel, so do not make further guest calls (even free) after one.
    pub(super) fn with_guest_buffers<const N: usize, T>(
        &mut self,
        call: impl FnOnce(&mut Self, &mut GuestBuffers<N>) -> OcctResult<T>,
    ) -> OcctResult<T> {
        let mut buffers = GuestBuffers {
            pointers: [0; N],
            used: 0,
        };
        let result = call(self, &mut buffers);
        if !matches!(
            &result,
            Err(OcctError::Runtime(_) | OcctError::MemoryLimit { .. })
        ) {
            for ptr in buffers.pointers.into_iter().take(buffers.used) {
                self.fn_free.call(&mut self.store, ptr)?;
            }
        }
        result
    }

    /// Read an owned byte slice from WASM memory.
    fn read_bytes(&self, ptr: u32, len: u32) -> OcctResult<Vec<u8>> {
        Ok(read_slice(self.memory, &self.store, ptr, len)?.to_vec())
    }

    // === Error handling ===

    /// Check if the WASM module has a pending error and return it.
    pub(crate) fn check_error(&mut self, operation: &str) -> OcctResult<()> {
        let has_error = self.fn_has_error.call(&mut self.store, ())?;
        if has_error != 0 {
            // Errors are the exception, not the norm — this branch runs after
            // every facade call. Hint to LLVM that it's cold so the happy path
            // stays tight in the instruction cache. (stabilized in Rust 1.95)
            core::hint::cold_path();
            return Err(self.read_last_error(operation));
        }
        Ok(())
    }

    /// Read the last error message from the WASM module.
    #[cold]
    pub(crate) fn read_last_error(&mut self, operation: &str) -> OcctError {
        let (ptr, len) = self.buffer_error.query(&mut self.store).unwrap_or((0, 0));
        let message = if ptr != 0 && len > 0 {
            self.read_bytes(ptr, len)
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .unwrap_or_else(|| "unknown error".to_owned())
        } else {
            "unknown error".to_owned()
        };
        OcctError::Operation {
            operation: operation.to_owned(),
            message,
        }
    }

    // === Result buffer readers ===

    /// Read a string result from the WASM string buffer.
    pub(crate) fn read_string_result(&mut self) -> OcctResult<String> {
        let bytes = self.read_bytes_result()?;
        String::from_utf8(bytes).map_err(|e| OcctError::Memory(e.to_string()))
    }

    /// Read a raw-bytes result (binary payloads such as STL).
    pub(crate) fn read_bytes_result(&mut self) -> OcctResult<Vec<u8>> {
        let (ptr, len) = self.buffer_string_result.query(&mut self.store)?;
        self.read_bytes(ptr, len)
    }

    /// Read a `Vec<u32>` result.
    pub(crate) fn read_vec_u32_result(&mut self) -> OcctResult<Vec<u32>> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        reader.read(&self.buffer_vec_u32_result)
    }

    /// Read a `Vec<ShapeHandle>` result.
    pub(crate) fn read_vec_shape_result(&mut self) -> OcctResult<Vec<ShapeHandle>> {
        Ok(self
            .read_vec_u32_result()?
            .into_iter()
            .map(ShapeHandle)
            .collect())
    }

    /// Read a `Vec<f64>` result.
    pub(crate) fn read_vec_f64_result(&mut self) -> OcctResult<Vec<f64>> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        reader.read(&self.buffer_vec_f64_result)
    }

    /// Read a `Vec<i32>` result.
    pub(crate) fn read_vec_i32_result(&mut self) -> OcctResult<Vec<i32>> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        reader.read(&self.buffer_vec_i32_result)
    }

    /// Read a bounding box result.
    pub(crate) fn read_bbox_result(&mut self) -> OcctResult<BoundingBox> {
        Ok(BoundingBox {
            min: Vec3 {
                x: self.fn_get_bbox_xmin.call(&mut self.store, ())?,
                y: self.fn_get_bbox_ymin.call(&mut self.store, ())?,
                z: self.fn_get_bbox_zmin.call(&mut self.store, ())?,
            },
            max: Vec3 {
                x: self.fn_get_bbox_xmax.call(&mut self.store, ())?,
                y: self.fn_get_bbox_ymax.call(&mut self.store, ())?,
                z: self.fn_get_bbox_zmax.call(&mut self.store, ())?,
            },
        })
    }

    /// Read a mesh result.
    pub(crate) fn read_mesh_result(&mut self) -> OcctResult<Mesh> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        Ok(Mesh {
            positions: reader.read(&self.buffer_mesh_positions)?,
            normals: reader.read(&self.buffer_mesh_normals)?,
            indices: reader.read(&self.buffer_mesh_indices)?,
            face_groups: reader.read(&self.buffer_mesh_face_groups)?,
        })
    }

    /// Read a mesh batch result.
    pub(crate) fn read_mesh_batch_result(&mut self) -> OcctResult<MeshBatch> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        let (off_ptr, shape_count) = self
            .buffer_mesh_batch_shape_offsets
            .query(&mut *reader.store)?;

        Ok(MeshBatch {
            positions: reader.read(&self.buffer_mesh_batch_positions)?,
            normals: reader.read(&self.buffer_mesh_batch_normals)?,
            indices: reader.read(&self.buffer_mesh_batch_indices)?,
            shape_offsets: read_array(
                self.memory,
                &*reader.store,
                off_ptr,
                checked_wasm_product(shape_count, 4)?,
                i32::from_le_bytes,
            )?,
        })
    }

    /// Read an edge data result.
    pub(crate) fn read_edge_result(&mut self) -> OcctResult<EdgeData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        Ok(EdgeData {
            points: reader.read(&self.buffer_edge_points)?,
            edge_groups: reader.read(&self.buffer_edge_groups)?,
        })
    }

    /// Read a NURBS curve data result.
    pub(crate) fn read_nurbs_result(&mut self) -> OcctResult<NurbsCurveData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        let degree = self.fn_get_nurbs_degree.call(&mut *reader.store, ())?;
        let rational = self.fn_get_nurbs_rational.call(&mut *reader.store, ())? != 0;
        let periodic = self.fn_get_nurbs_periodic.call(&mut *reader.store, ())? != 0;

        Ok(NurbsCurveData {
            degree,
            rational,
            periodic,
            knots: reader.read(&self.buffer_nurbs_knots)?,
            multiplicities: reader.read(&self.buffer_nurbs_multiplicities)?,
            poles: reader.read(&self.buffer_nurbs_poles)?,
            weights: reader.read(&self.buffer_nurbs_weights)?,
        })
    }

    /// Read a curve geometry result.
    pub(crate) fn read_curve_result(&mut self) -> OcctResult<CurveData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        let (kind_ptr, kind_len) = self.buffer_curve_kind.query(&mut *reader.store)?;
        let kind = String::from_utf8(
            read_slice(reader.memory, reader.store, kind_ptr, kind_len).map(<[u8]>::to_vec)?,
        )
        .map_err(|e| OcctError::Memory(e.to_string()))?;
        let first = self.fn_get_curve_first.call(&mut *reader.store, ())?;
        let last = self.fn_get_curve_last.call(&mut *reader.store, ())?;
        let params = reader.read(&self.buffer_curve_params)?;
        let has_bspline = self.fn_get_curve_has_bspline.call(&mut *reader.store, ())? != 0;
        let bspline = if has_bspline {
            Some(self.read_nurbs_result()?)
        } else {
            None
        };
        Ok(CurveData {
            first,
            last,
            geometry: decode_curve_geometry(kind, &params, bspline)?,
        })
    }

    /// Read a surface geometry result.
    pub(crate) fn read_surface_result(&mut self) -> OcctResult<SurfaceData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        let (kind_ptr, kind_len) = self.buffer_surface_kind.query(&mut *reader.store)?;
        let kind = String::from_utf8(
            read_slice(reader.memory, reader.store, kind_ptr, kind_len).map(<[u8]>::to_vec)?,
        )
        .map_err(|e| OcctError::Memory(e.to_string()))?;
        let is_direct = self.fn_get_surface_is_direct.call(&mut *reader.store, ())? != 0;
        let params = reader.read(&self.buffer_surface_params)?;
        let has_bspline = self
            .fn_get_surface_has_bspline
            .call(&mut *reader.store, ())?
            != 0;
        let bspline = if has_bspline {
            Some(self.read_nurbs_surface()?)
        } else {
            None
        };
        decode_surface_data(kind, &params, is_direct, bspline)
    }

    fn read_nurbs_surface(&mut self) -> OcctResult<NurbsSurfaceData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        Ok(NurbsSurfaceData {
            degree_u: self.fn_get_surface_degree_u.call(&mut *reader.store, ())?,
            degree_v: self.fn_get_surface_degree_v.call(&mut *reader.store, ())?,
            nb_poles_u: self
                .fn_get_surface_nb_poles_u
                .call(&mut *reader.store, ())?,
            nb_poles_v: self
                .fn_get_surface_nb_poles_v
                .call(&mut *reader.store, ())?,
            periodic_u: self
                .fn_get_surface_periodic_u
                .call(&mut *reader.store, ())?
                != 0,
            periodic_v: self
                .fn_get_surface_periodic_v
                .call(&mut *reader.store, ())?
                != 0,
            knots_u: reader.read(&self.buffer_surface_knots_u)?,
            knots_v: reader.read(&self.buffer_surface_knots_v)?,
            multiplicities_u: reader.read(&self.buffer_surface_multiplicities_u)?,
            multiplicities_v: reader.read(&self.buffer_surface_multiplicities_v)?,
            poles: reader.read(&self.buffer_surface_poles)?,
            weights: reader.read(&self.buffer_surface_weights)?,
        })
    }

    /// Read an evolution data result.
    pub(crate) fn read_evolution_result(&mut self) -> OcctResult<EvolutionData> {
        let mut reader = ResultReader {
            memory: self.memory,
            store: &mut self.store,
        };
        let result_id = self.fn_get_evo_result_id.call(&mut *reader.store, ())?;

        Ok(EvolutionData {
            result_id,
            modified: reader.read(&self.buffer_evo_modified)?,
            generated: reader.read(&self.buffer_evo_generated)?,
            deleted: reader.read(&self.buffer_evo_deleted)?,
            edge_modified: reader.read(&self.buffer_evo_edge_modified)?,
            edge_generated: reader.read(&self.buffer_evo_edge_generated)?,
            edge_deleted: reader.read(&self.buffer_evo_edge_deleted)?,
            first_shape: reader.read(&self.buffer_evo_first_shape)?,
            last_shape: reader.read(&self.buffer_evo_last_shape)?,
            spine_generated: reader.read(&self.buffer_evo_spine_generated)?,
        })
    }

    /// Read a projection data result.
    pub(crate) fn read_projection_result(&mut self) -> OcctResult<ProjectionData> {
        Ok(ProjectionData {
            visible_outline: ShapeHandle(
                self.fn_get_proj_visible_outline.call(&mut self.store, ())?,
            ),
            visible_smooth: ShapeHandle(self.fn_get_proj_visible_smooth.call(&mut self.store, ())?),
            visible_sharp: ShapeHandle(self.fn_get_proj_visible_sharp.call(&mut self.store, ())?),
            hidden_outline: ShapeHandle(self.fn_get_proj_hidden_outline.call(&mut self.store, ())?),
            hidden_smooth: ShapeHandle(self.fn_get_proj_hidden_smooth.call(&mut self.store, ())?),
            hidden_sharp: ShapeHandle(self.fn_get_proj_hidden_sharp.call(&mut self.store, ())?),
        })
    }

    /// Read a label info result.
    pub(crate) fn read_label_info_result(&mut self) -> OcctResult<LabelInfo> {
        let label_id = self.fn_get_label_info_label_id.call(&mut self.store, ())?;
        let (name_ptr, name_len) = self.buffer_label_info_name.query(&mut self.store)?;
        let name_bytes = self.read_bytes(name_ptr, name_len)?;
        let name = String::from_utf8(name_bytes).map_err(|e| OcctError::Memory(e.to_string()))?;

        Ok(LabelInfo {
            label_id,
            name,
            has_color: self.fn_get_label_info_has_color.call(&mut self.store, ())? != 0,
            r: self.fn_get_label_info_r.call(&mut self.store, ())?,
            g: self.fn_get_label_info_g.call(&mut self.store, ())?,
            b: self.fn_get_label_info_b.call(&mut self.store, ())?,
            is_assembly: self
                .fn_get_label_info_is_assembly
                .call(&mut self.store, ())?
                != 0,
            is_component: self
                .fn_get_label_info_is_component
                .call(&mut self.store, ())?
                != 0,
            shape_id: self.fn_get_label_info_shape_id.call(&mut self.store, ())?,
        })
    }
}

impl Drop for OcctKernel {
    fn drop(&mut self) {
        // Best-effort cleanup
        if let Ok(destroy) = self
            .instance
            .get_typed_func::<(), ()>(&mut self.store, "occt_destroy")
        {
            let _ = destroy.call(&mut self.store, ());
        }
    }
}

/// Borrow a byte slice from WASM memory after validating its full range.
fn read_slice(memory: Memory, store: &Store<HostState>, ptr: u32, len: u32) -> OcctResult<&[u8]> {
    let mem = memory.data(store);
    let start = ptr as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| OcctError::Memory("read range overflow".to_owned()))?;
    mem.get(start..end)
        .ok_or_else(|| OcctError::Memory("read out of bounds".to_owned()))
}

/// Decode little-endian elements without an intermediate byte allocation.
fn read_array<const N: usize, T>(
    memory: Memory,
    store: &Store<HostState>,
    ptr: u32,
    count: u32,
    decode: impl Fn([u8; N]) -> T,
) -> OcctResult<Vec<T>> {
    let width = checked_wasm_len(N, 1)?;
    let len = checked_wasm_product(count, width)?;
    let (chunks, remainder) = read_slice(memory, store, ptr, len)?.as_chunks::<N>();
    debug_assert!(remainder.is_empty());
    Ok(chunks.iter().copied().map(decode).collect())
}

/// Instantiate the WASI reactor and initialize its C++ runtime.
fn instantiate(
    module: &Module,
    limiter: impl ResourceLimiter + Sync + 'static,
) -> OcctResult<(Store<HostState>, Instance, Memory)> {
    let engine = module.engine();
    let mut store = Store::new(engine, HostState::new(Box::new(limiter)));
    store.limiter(|state| state);
    let mut linker = Linker::new(engine);
    crate::host::define_imports(&mut linker)?;
    // An import a future build adds instantiates as a trap instead of
    // failing `new()`; it only matters if it is ever called.
    linker.define_unknown_imports_as_traps(module)?;
    let instance = linker.instantiate(&mut store, module)?;

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| OcctError::Memory("no memory export".to_owned()))?;

    // A WASI reactor runs its C++ static constructors in `_initialize`, which
    // the host must call before any other export. Skipping it leaves globals
    // such as `std::cout` unconstructed, and OCCT's STEP code traps without them.
    let initialize: TypedFunc<(), ()> = instance.get_typed_func(&mut store, "_initialize")?;
    initialize.call(&mut store, ())?;

    // Call occt_init
    let init: TypedFunc<(), i32> = instance.get_typed_func(&mut store, "occt_init")?;
    let result = init.call(&mut store, ())?;
    if result != 0 {
        return Err(OcctError::Memory("occt_init failed".to_owned()));
    }

    Ok((store, instance, memory))
}

/// Decompress the embedded brotli-compressed WASM binary.
fn decompress_wasm() -> OcctResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut input: &[u8] = WASM_BINARY;
    brotli::BrotliDecompress(&mut input, &mut output)
        .map_err(|e| OcctError::Memory(format!("brotli decompression failed: {e}")))?;
    Ok(output)
}

const fn vec3(v: [f64; 3]) -> Vec3 {
    Vec3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

fn malformed(kind: &str) -> OcctError {
    OcctError::Memory(format!("malformed {kind} geometry result"))
}

fn decode_curve_geometry(
    kind: String,
    params: &[f64],
    bspline: Option<NurbsCurveData>,
) -> OcctResult<CurveGeometry> {
    let frame = |p: &[f64]| -> Option<(Vec3, Vec3, Vec3)> {
        let &[lx, ly, lz, ax, ay, az, xx, xy, xz, ..] = p else {
            return None;
        };
        Some((vec3([lx, ly, lz]), vec3([ax, ay, az]), vec3([xx, xy, xz])))
    };
    let geometry = match kind.as_str() {
        "line" => {
            let &[px, py, pz, dx, dy, dz] = params else {
                return Err(malformed(&kind));
            };
            CurveGeometry::Line {
                point: vec3([px, py, pz]),
                direction: vec3([dx, dy, dz]),
            }
        }
        "circle" => {
            let ((center, axis, x_direction), Some(&radius)) = (
                frame(params).ok_or_else(|| malformed(&kind))?,
                params.get(9),
            ) else {
                return Err(malformed(&kind));
            };
            CurveGeometry::Circle {
                center,
                axis,
                x_direction,
                radius,
            }
        }
        "ellipse" | "hyperbola" => {
            let ((center, axis, major_direction), Some(&major_radius), Some(&minor_radius)) = (
                frame(params).ok_or_else(|| malformed(&kind))?,
                params.get(9),
                params.get(10),
            ) else {
                return Err(malformed(&kind));
            };
            if kind == "ellipse" {
                CurveGeometry::Ellipse {
                    center,
                    axis,
                    major_direction,
                    major_radius,
                    minor_radius,
                }
            } else {
                CurveGeometry::Hyperbola {
                    center,
                    axis,
                    major_direction,
                    major_radius,
                    minor_radius,
                }
            }
        }
        "parabola" => {
            let ((vertex, axis, x_direction), Some(&focal)) = (
                frame(params).ok_or_else(|| malformed(&kind))?,
                params.get(9),
            ) else {
                return Err(malformed(&kind));
            };
            CurveGeometry::Parabola {
                vertex,
                axis,
                x_direction,
                focal,
            }
        }
        "bspline" => CurveGeometry::BSpline(bspline.ok_or_else(|| malformed(&kind))?),
        _ => CurveGeometry::Other { kind, bspline },
    };
    Ok(geometry)
}

fn decode_surface_data(
    kind: String,
    params: &[f64],
    is_direct: bool,
    bspline: Option<NurbsSurfaceData>,
) -> OcctResult<SurfaceData> {
    let frame = || -> OcctResult<SurfaceFrame> {
        let &[lx, ly, lz, ax, ay, az, xx, xy, xz, ..] = params else {
            return Err(malformed(&kind));
        };
        Ok(SurfaceFrame {
            location: vec3([lx, ly, lz]),
            axis: vec3([ax, ay, az]),
            x_direction: vec3([xx, xy, xz]),
            is_direct,
        })
    };
    let value =
        |at: usize| -> OcctResult<f64> { params.get(at).copied().ok_or_else(|| malformed(&kind)) };
    let data = match kind.as_str() {
        "plane" => SurfaceData::Plane(frame()?),
        "cylinder" => SurfaceData::Cylinder {
            frame: frame()?,
            radius: value(9)?,
        },
        "cone" => SurfaceData::Cone {
            frame: frame()?,
            ref_radius: value(9)?,
            semi_angle: value(10)?,
        },
        "sphere" => SurfaceData::Sphere {
            frame: frame()?,
            radius: value(9)?,
        },
        "torus" => SurfaceData::Torus {
            frame: frame()?,
            major_radius: value(9)?,
            minor_radius: value(10)?,
        },
        "bspline" => SurfaceData::BSpline(bspline.ok_or_else(|| malformed(&kind))?),
        _ => SurfaceData::Other { kind, bspline },
    };
    Ok(data)
}

/// Validate the element count and total byte size before crossing the wasm32 ABI.
pub(super) fn checked_wasm_len(count: usize, width: u32) -> OcctResult<u32> {
    let count = u32::try_from(count)
        .map_err(|_| OcctError::Memory("element count exceeds wasm32".to_owned()))?;
    checked_wasm_product(count, width)?;
    Ok(count)
}

fn checked_wasm_product(count: u32, width: u32) -> OcctResult<u32> {
    count
        .checked_mul(width)
        .ok_or_else(|| OcctError::Memory("buffer size exceeds wasm32".to_owned()))
}

fn checked_count(count: impl TryInto<u32>) -> OcctResult<u32> {
    count
        .try_into()
        .map_err(|_| OcctError::Memory("negative buffer element count".to_owned()))
}

#[cfg(test)]
mod memory_tests {
    use super::{checked_count, checked_wasm_len, checked_wasm_product};
    use crate::OcctResult;

    #[test]
    fn rejects_out_of_range_buffer_counts() -> OcctResult<()> {
        assert_eq!(checked_wasm_len(0, 8)?, 0);
        assert_eq!(checked_wasm_len(12, 8)?, 12);
        assert_eq!(checked_wasm_product(u32::MAX / 8, 8)?, u32::MAX - 7);
        assert!(checked_wasm_product(u32::MAX / 8 + 1, 8).is_err());
        assert!(checked_wasm_len((u32::MAX / 8 + 1) as usize, 8).is_err());
        if let Ok(too_large) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(checked_wasm_len(too_large, 1).is_err());
        }
        assert!(checked_count(-1).is_err());
        assert_eq!(checked_count(i32::MAX)?, i32::MAX.cast_unsigned());
        Ok(())
    }
}
