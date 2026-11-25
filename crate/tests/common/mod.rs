//! Share Wasmtime's content-addressed compilation cache between test processes.
//! Ordinary libtest also retains the in-process module; Nextest can load the
//! same compiled module without recompiling the embedded kernel for each case.

use std::{fs, path::Path, sync::OnceLock};

use occt_wasm::{
    OcctKernel,
    wasmtime::{Cache, CacheConfig, Engine, Module},
};

fn compile_cached() -> anyhow::Result<Module> {
    // Cargo's integration-test scratch directory is specific to the target
    // directory. No global cache, user configuration, or persistent daemon.
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("occt-test-cache");
    fs::create_dir_all(&root)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("compile.lock"))?;
    // Prevent a cold-cache stampede when Nextest launches multiple tests.
    // The OS releases the lock if the compiling test process terminates.
    lock.lock()?;
    let mut cache_config = CacheConfig::new();
    cache_config.with_directory(root.canonicalize()?.join("modules"));
    let mut config = OcctKernel::engine_config();
    config.cache(Some(Cache::new(cache_config)?));
    let engine = Engine::new(&config)?;
    Ok(OcctKernel::compile(&engine)?)
}

pub fn shared_module() -> Result<&'static Module, &'static str> {
    static MODULE: OnceLock<Result<Module, String>> = OnceLock::new();
    MODULE
        .get_or_init(|| compile_cached().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(String::as_str)
}
