//! Rust host API emitter for the facade code generator.
//!
//! Generates `kernel_generated.rs` — the Rust wrapper methods that call
//! into the WASM module via wasmtime. Each `MethodSpec` produces one
//! `pub fn method_name(&mut self, ...) -> OcctResult<T>` method on `OcctKernel`.

use std::fmt::Write as _;

use super::types::{FacadeParam, MethodSpec, ReturnType};
use super::wasi_emitter::camel_to_snake;

/// Convert a `FacadeParam` to a Rust function parameter declaration.
fn param_to_rust(param: &FacadeParam) -> String {
    match param {
        FacadeParam::ShapeId(name) => format!("{}: ShapeHandle", rust_param_name(name)),
        FacadeParam::DocId(name) => format!("{}: DocumentHandle", rust_param_name(name)),
        FacadeParam::Double(name) => format!("{}: f64", rust_param_name(name)),
        FacadeParam::Bool(name) => format!("{}: bool", rust_param_name(name)),
        FacadeParam::Int(name) => format!("{}: i32", rust_param_name(name)),
        FacadeParam::Uint32(name) => format!("{}: u32", rust_param_name(name)),
        FacadeParam::String(name) => format!("{}: &str", rust_param_name(name)),
        FacadeParam::Bytes(name) => format!("{}: &[u8]", rust_param_name(name)),
        FacadeParam::VectorShapeIds(name) => format!("{}: &[ShapeHandle]", rust_param_name(name)),
        FacadeParam::VectorDouble(name) => format!("{}: &[f64]", rust_param_name(name)),
        FacadeParam::VectorInt(name) => format!("{}: &[i32]", rust_param_name(name)),
    }
}

/// Convert a `camelCase` param name to `snake_case` for Rust.
fn rust_param_name(name: &str) -> String {
    camel_to_snake(name)
}

/// Build the Rust parameter list for a method.
fn rust_param_list(params: &[FacadeParam]) -> String {
    if params.is_empty() {
        return String::new();
    }
    params
        .iter()
        .map(param_to_rust)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The Rust return type for a given `ReturnType`.
const fn rust_return_type(rt: ReturnType) -> &'static str {
    match rt {
        ReturnType::ShapeId => "OcctResult<ShapeHandle>",
        ReturnType::DocId => "OcctResult<DocumentHandle>",
        ReturnType::Uint32 => "OcctResult<u32>",
        ReturnType::Bool => "OcctResult<bool>",
        ReturnType::Void => "OcctResult<()>",
        ReturnType::Double => "OcctResult<f64>",
        ReturnType::Int => "OcctResult<i32>",
        ReturnType::String => "OcctResult<String>",
        ReturnType::Bytes => "OcctResult<Vec<u8>>",
        ReturnType::VectorUint32 => "OcctResult<Vec<u32>>",
        ReturnType::VectorShapeIds => "OcctResult<Vec<ShapeHandle>>",
        ReturnType::VectorDouble => "OcctResult<Vec<f64>>",
        ReturnType::VectorInt => "OcctResult<Vec<i32>>",
        ReturnType::BBoxData => "OcctResult<BoundingBox>",
        ReturnType::MeshData => "OcctResult<Mesh>",
        ReturnType::MeshBatchData => "OcctResult<MeshBatch>",
        ReturnType::EdgeData => "OcctResult<EdgeData>",
        ReturnType::NurbsCurveData => "OcctResult<NurbsCurveData>",
        ReturnType::CurveData => "OcctResult<CurveData>",
        ReturnType::SurfaceData => "OcctResult<SurfaceData>",
        ReturnType::EvolutionData => "OcctResult<EvolutionData>",
        ReturnType::ProjectionData => "OcctResult<ProjectionData>",
        ReturnType::XCAFLabelInfo => "OcctResult<LabelInfo>",
    }
}

/// Number of bytes per element for a marshalled parameter.
const fn heap_element_width(param: &FacadeParam) -> Option<u32> {
    match param {
        FacadeParam::String(_) | FacadeParam::Bytes(_) => Some(1),
        FacadeParam::VectorShapeIds(_) | FacadeParam::VectorInt(_) => Some(4),
        FacadeParam::VectorDouble(_) => Some(8),
        _ => None,
    }
}

/// Validate each heap argument before any allocation occurs.
fn emit_wasm_call_preflight(buf: &mut String, params: &[FacadeParam]) {
    for param in params {
        if let Some(width) = heap_element_width(param) {
            let rname = rust_param_name(param.name());
            let _ = writeln!(
                buf,
                "        let {rname}_len = checked_wasm_len({rname}.len(), {width})?;"
            );
        }
    }
}

/// Emit guest allocations inside a scope which frees them on ordinary errors.
fn emit_wasm_call_allocations(buf: &mut String, params: &[FacadeParam]) {
    for param in params {
        let rname = rust_param_name(param.name());
        let data_expr = match param {
            FacadeParam::String(_) => format!("{rname}.as_bytes()"),
            FacadeParam::Bytes(_) => rname.clone(),
            FacadeParam::VectorShapeIds(_)
            | FacadeParam::VectorDouble(_)
            | FacadeParam::VectorInt(_) => {
                let value = if matches!(param, FacadeParam::VectorShapeIds(_)) {
                    "v.0"
                } else {
                    "v"
                };
                let _ = writeln!(
                    buf,
                    "            let {rname}_bytes: Vec<u8> = {rname}.iter().flat_map(|v| {value}.to_le_bytes()).collect();"
                );
                format!("&{rname}_bytes")
            }
            _ => continue,
        };
        let _ = writeln!(
            buf,
            "            let {rname}_ptr = buffers.write(kernel, {data_expr})?;"
        );
    }
}

/// Generate the WASM call expression arguments.
fn emit_wasm_call_args(params: &[FacadeParam]) -> String {
    if params.is_empty() {
        return String::new();
    }
    let args: Vec<String> = params
        .iter()
        .flat_map(|p| match p {
            FacadeParam::ShapeId(name) | FacadeParam::DocId(name) => {
                vec![format!("{}.0", rust_param_name(name))]
            }
            FacadeParam::Double(name) | FacadeParam::Int(name) | FacadeParam::Uint32(name) => {
                vec![rust_param_name(name)]
            }
            FacadeParam::Bool(name) => vec![format!("i32::from({})", rust_param_name(name))],
            FacadeParam::String(name)
            | FacadeParam::Bytes(name)
            | FacadeParam::VectorShapeIds(name)
            | FacadeParam::VectorDouble(name)
            | FacadeParam::VectorInt(name) => {
                let rname = rust_param_name(name);
                vec![format!("{rname}_ptr"), format!("{rname}_len")]
            }
        })
        .collect();
    args.join(", ")
}

/// Count the number of WASM-level parameters (strings/vectors expand to 2).
fn wasm_param_count(params: &[FacadeParam]) -> usize {
    params
        .iter()
        .map(|p| match p {
            FacadeParam::String(_)
            | FacadeParam::Bytes(_)
            | FacadeParam::VectorShapeIds(_)
            | FacadeParam::VectorDouble(_)
            | FacadeParam::VectorInt(_) => 2,
            _ => 1,
        })
        .sum()
}

/// Generate the wasmtime `TypedFunc` type for a method's parameters.
fn wasm_typed_func_type(spec: &MethodSpec) -> String {
    let param_count = wasm_param_count(spec.params);

    // Build param types tuple
    let param_types: Vec<&str> = spec
        .params
        .iter()
        .flat_map(|p| match p {
            FacadeParam::ShapeId(_) | FacadeParam::DocId(_) | FacadeParam::Uint32(_) => vec!["u32"],
            FacadeParam::Double(_) => vec!["f64"],
            FacadeParam::Bool(_) | FacadeParam::Int(_) => vec!["i32"],
            FacadeParam::String(_)
            | FacadeParam::Bytes(_)
            | FacadeParam::VectorShapeIds(_)
            | FacadeParam::VectorDouble(_)
            | FacadeParam::VectorInt(_) => vec!["u32", "u32"],
        })
        .collect();

    // All non-scalar returns cross the WASM boundary as an `i32` status/length;
    // the match is exhaustive so a new `ReturnType` is a compile error.
    let ret_type = match spec.return_type {
        ReturnType::ShapeId | ReturnType::DocId | ReturnType::Uint32 => "u32",
        ReturnType::Double => "f64",
        ReturnType::Bool
        | ReturnType::Void
        | ReturnType::Int
        | ReturnType::String
        | ReturnType::Bytes
        | ReturnType::VectorUint32
        | ReturnType::VectorShapeIds
        | ReturnType::VectorDouble
        | ReturnType::VectorInt
        | ReturnType::BBoxData
        | ReturnType::NurbsCurveData
        | ReturnType::CurveData
        | ReturnType::SurfaceData
        | ReturnType::EvolutionData
        | ReturnType::MeshData
        | ReturnType::MeshBatchData
        | ReturnType::EdgeData
        | ReturnType::ProjectionData
        | ReturnType::XCAFLabelInfo => "i32",
    };

    if param_count == 0 {
        format!("TypedFunc<(), {ret_type}>")
    } else if param_count == 1 {
        format!("TypedFunc<({},), {ret_type}>", param_types[0])
    } else {
        format!("TypedFunc<({}), {ret_type}>", param_types.join(", "))
    }
}

/// How the C ABI signals errors for the result category.
enum ResultCheck {
    Flag,
    Handle,
    Status,
}

/// Describe the binding name, error check and successful return expression.
const fn result_binding(rt: ReturnType) -> (&'static str, ResultCheck, &'static str) {
    use ResultCheck::{Flag, Handle, Status};
    match rt {
        ReturnType::ShapeId => ("result", Handle, "Ok(ShapeHandle(result))"),
        ReturnType::DocId => ("result", Flag, "Ok(DocumentHandle(result))"),
        ReturnType::Uint32 | ReturnType::Double | ReturnType::Int => ("result", Flag, "Ok(result)"),
        ReturnType::Bool => ("result", Status, "Ok(result != 0)"),
        ReturnType::Void => ("result", Status, "Ok(())"),
        ReturnType::String => ("len", Status, "self.read_string_result()"),
        ReturnType::Bytes => ("len", Status, "self.read_bytes_result()"),
        ReturnType::VectorUint32 => ("len", Status, "self.read_vec_u32_result()"),
        ReturnType::VectorShapeIds => ("len", Status, "self.read_vec_shape_result()"),
        ReturnType::VectorDouble => ("len", Status, "self.read_vec_f64_result()"),
        ReturnType::VectorInt => ("len", Status, "self.read_vec_i32_result()"),
        ReturnType::BBoxData => ("status", Status, "self.read_bbox_result()"),
        ReturnType::MeshData => ("status", Status, "self.read_mesh_result()"),
        ReturnType::MeshBatchData => ("status", Status, "self.read_mesh_batch_result()"),
        ReturnType::EdgeData => ("status", Status, "self.read_edge_result()"),
        ReturnType::NurbsCurveData => ("status", Status, "self.read_nurbs_result()"),
        ReturnType::CurveData => ("status", Status, "self.read_curve_result()"),
        ReturnType::SurfaceData => ("status", Status, "self.read_surface_result()"),
        ReturnType::EvolutionData => ("status", Status, "self.read_evolution_result()"),
        ReturnType::ProjectionData => ("status", Status, "self.read_projection_result()"),
        ReturnType::XCAFLabelInfo => ("status", Status, "self.read_label_info_result()"),
    }
}

/// Emit a single method implementation.
fn emit_rust_method(buf: &mut String, spec: &MethodSpec) {
    let snake_name = camel_to_snake(spec.name);
    let rust_ret = rust_return_type(spec.return_type);
    let rust_params = rust_param_list(spec.params);

    let params_str = if rust_params.is_empty() {
        String::from("&mut self")
    } else {
        format!("&mut self, {rust_params}")
    };

    let _ = writeln!(
        buf,
        "    /// Call the `{}` OCCT facade operation.",
        spec.name
    );
    buf.push_str(
        r"    ///
    /// # Errors
    /// Returns an error if the WASM call traps, the operation fails,
    /// or marshalling its arguments or results fails.
",
    );
    let _ = writeln!(
        buf,
        "    pub fn {snake_name}({params_str}) -> {rust_ret} {{"
    );

    emit_wasm_call_preflight(buf, spec.params);

    // Build the call arguments
    let call_args = emit_wasm_call_args(spec.params);
    let call_tuple = if wasm_param_count(spec.params) == 0 {
        "()".to_owned()
    } else if wasm_param_count(spec.params) == 1 {
        format!("({call_args},)")
    } else {
        format!("({call_args})")
    };

    let fn_field = format!("generated.fn_{snake_name}");
    let heap_count = spec
        .params
        .iter()
        .filter(|param| heap_element_width(param).is_some())
        .count();

    let (name, check, expression) = result_binding(spec.return_type);
    if heap_count == 0 {
        let _ = writeln!(
            buf,
            "        let {name} = self.{fn_field}.call(&mut self.store, {call_tuple})?;"
        );
    } else {
        let _ = writeln!(
            buf,
            "        let {name} = self.with_guest_buffers::<{heap_count}, _>(|kernel, buffers| {{"
        );
        emit_wasm_call_allocations(buf, spec.params);
        let _ = writeln!(
            buf,
            "            Ok(kernel.{fn_field}.call(&mut kernel.store, {call_tuple})?)"
        );
        let _ = writeln!(buf, "        }})?;");
    }

    if matches!(check, ResultCheck::Flag | ResultCheck::Handle) {
        let _ = writeln!(buf, "        self.check_error(\"{snake_name}\")?;");
    }
    if let Some(condition) = match check {
        ResultCheck::Flag => None,
        ResultCheck::Handle => Some(format!("{name} == 0")),
        ResultCheck::Status => Some(format!("{name} < 0")),
    } {
        let _ = writeln!(buf, "        if {condition} {{");
        let _ = writeln!(
            buf,
            "            return Err(self.read_last_error(\"{snake_name}\"));"
        );
        let _ = writeln!(buf, "        }}");
    }
    let _ = writeln!(buf, "        {expression}");

    let _ = writeln!(buf, "    }}");
}

/// Generate the typed function field declarations for the struct.
fn emit_func_fields(buf: &mut String, methods: &[&MethodSpec]) {
    for spec in methods {
        if !spec.has_wasi_binding() {
            continue;
        }
        let snake_name = camel_to_snake(spec.name);
        let func_type = wasm_typed_func_type(spec);
        let _ = writeln!(buf, "    fn_{snake_name}: {func_type},");
    }
}

/// Generate the function lookup code for initialization.
fn emit_func_lookups(buf: &mut String, methods: &[&MethodSpec]) {
    for spec in methods {
        if !spec.has_wasi_binding() {
            continue;
        }
        let snake_name = camel_to_snake(spec.name);
        let _ = writeln!(
            buf,
            "            fn_{snake_name}: instance.get_typed_func(&mut *store, \"occt_{snake_name}\")?,",
        );
    }
}

/// Generate the contents of `crate/src/kernel_generated.rs`.
///
/// This file contains:
/// 1. The function fields for the `OcctKernel` struct
/// 2. The `impl OcctKernel` block with all method wrappers
/// 3. The function lookup initialization code
pub fn emit_rust_host(methods: &[&MethodSpec]) -> String {
    let mut buf = String::with_capacity(32768);

    buf.push_str(
        r"// AUTO-GENERATED by cargo xtask codegen -- DO NOT EDIT
// Rust host API for occt-wasm WASI module via wasmtime

use wasmtime::TypedFunc;

use crate::error::OcctResult;
use crate::kernel::checked_wasm_len;
use crate::types::{
    BoundingBox, CurveData, EdgeData, EvolutionData, LabelInfo, Mesh, MeshBatch,
    DocumentHandle, NurbsCurveData, ProjectionData, ShapeHandle, SurfaceData,
};

",
    );

    // Struct fields section (to be included in the kernel struct)
    buf.push_str(
        r"/// Cached WASM function handles for kernel methods.
///
/// Include this in the `OcctKernel` struct definition:
/// ```ignore
/// pub struct OcctKernel {
///     // ... core fields ...
///     #[doc(hidden)]
///     generated: GeneratedFuncs,
/// }
/// ```
#[allow(clippy::type_complexity, clippy::redundant_pub_crate, clippy::struct_field_names)]
pub(crate) struct GeneratedFuncs {
",
    );
    emit_func_fields(&mut buf, methods);
    let _ = writeln!(buf, "}}");
    let _ = writeln!(buf);

    // Init function to resolve all typed funcs
    buf.push_str(
        r"impl GeneratedFuncs {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn resolve<T: 'static>(
        instance: &wasmtime::Instance,
        store: &mut wasmtime::Store<T>,
    ) -> OcctResult<Self> {
        Ok(Self {
",
    );
    emit_func_lookups(&mut buf, methods);
    buf.push_str(
        r"        })
    }
}

",
    );

    // Method implementations
    buf.push_str(
        r"/// Generated kernel methods.
///
/// All facade methods wrapped as safe Rust functions.
",
    );
    // Each vector parameter adds a fixed block of marshalling, so a method with
    // several of them outgrows the line limit without being complex.
    let _ = writeln!(
        buf,
        "#[allow(clippy::too_many_arguments, clippy::too_many_lines)]"
    );
    let _ = writeln!(buf, "impl crate::kernel::OcctKernel {{");

    let generable: Vec<&&MethodSpec> = methods.iter().filter(|m| m.has_wasi_binding()).collect();

    for (i, spec) in generable.iter().enumerate() {
        emit_rust_method(&mut buf, spec);
        if i + 1 < generable.len() {
            let _ = writeln!(buf);
        }
    }

    let _ = writeln!(buf, "}}");

    buf.trim_end().to_owned() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::types::{FacadeParam, MethodKind, MethodSpec, ReturnType};

    static XCAF_NEW_DOCUMENT: MethodSpec = MethodSpec {
        name: "xcafNewDocument",
        kind: MethodKind::CustomBody,
        params: &[],
        return_type: ReturnType::DocId,
        occt_class: "",
        ctor_args: "",
        setup_code: "return 1;",
        includes: &[],
        category: "xcaf",
    };

    static XCAF_CLOSE: MethodSpec = MethodSpec {
        name: "xcafClose",
        kind: MethodKind::CustomBody,
        params: &[FacadeParam::DocId("docId")],
        return_type: ReturnType::Void,
        occt_class: "",
        ctor_args: "",
        setup_code: "(void)docId;",
        includes: &[],
        category: "xcaf",
    };

    #[test]
    fn document_ids_are_document_handles() {
        let mut buf = String::new();
        emit_rust_method(&mut buf, &XCAF_NEW_DOCUMENT);
        assert!(buf.contains("-> OcctResult<DocumentHandle>"));
        assert!(buf.contains("Ok(DocumentHandle(result))"));

        let mut buf = String::new();
        emit_rust_method(&mut buf, &XCAF_CLOSE);
        assert!(buf.contains("doc_id: DocumentHandle"));
        assert!(buf.contains("(doc_id.0,)"));
    }

    static MAKE_BOX: MethodSpec = MethodSpec {
        name: "makeBox",
        kind: MethodKind::SimpleShape,
        params: &[
            FacadeParam::Double("dx"),
            FacadeParam::Double("dy"),
            FacadeParam::Double("dz"),
        ],
        return_type: ReturnType::ShapeId,
        occt_class: "BRepPrimAPI_MakeBox",
        ctor_args: "dx, dy, dz",
        setup_code: "",
        includes: &[],
        category: "primitives",
    };

    static FUSE: MethodSpec = MethodSpec {
        name: "fuse",
        kind: MethodKind::BooleanOp,
        params: &[FacadeParam::ShapeId("a"), FacadeParam::ShapeId("b")],
        return_type: ReturnType::ShapeId,
        occt_class: "BRepAlgoAPI_Fuse",
        ctor_args: "get(a), get(b)",
        setup_code: "",
        includes: &[],
        category: "booleans",
    };

    #[test]
    fn generates_make_box_method() {
        let output = emit_rust_host(&[&MAKE_BOX]);
        assert!(output.contains(
            "pub fn make_box(&mut self, dx: f64, dy: f64, dz: f64) -> OcctResult<ShapeHandle>"
        ));
        assert!(output.contains("/// Call the `makeBox` OCCT facade operation."));
        assert!(output.contains("/// # Errors"));
        assert!(output.contains("fn_make_box"));
        assert!(output.contains("ShapeHandle(result)"));
    }

    #[test]
    fn generates_fuse_method() {
        let output = emit_rust_host(&[&FUSE]);
        assert!(output.contains(
            "pub fn fuse(&mut self, a: ShapeHandle, b: ShapeHandle) -> OcctResult<ShapeHandle>"
        ));
        assert!(output.contains("a.0"));
        assert!(output.contains("b.0"));
    }

    #[test]
    fn generates_struct_fields() {
        let output = emit_rust_host(&[&MAKE_BOX, &FUSE]);
        assert!(output.contains("fn_make_box: TypedFunc<(f64, f64, f64), u32>"));
        assert!(output.contains("fn_fuse: TypedFunc<(u32, u32), u32>"));
    }

    #[test]
    fn generates_func_lookups() {
        let output = emit_rust_host(&[&MAKE_BOX]);
        assert!(output.contains("instance.get_typed_func(&mut *store, \"occt_make_box\")"));
    }

    #[test]
    fn string_param_generates_write_bytes() {
        static IMPORT: MethodSpec = MethodSpec {
            name: "importStep",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::String("data")],
            return_type: ReturnType::ShapeId,
            occt_class: "",
            ctor_args: "",
            setup_code: "return store(importStepImpl(data));",
            includes: &[],
            category: "io",
        };
        let output = emit_rust_host(&[&IMPORT]);
        assert!(output.contains("data: &str"));
        assert!(output.contains("buffers.write(kernel, data.as_bytes())"));
        assert!(output.contains("self.with_guest_buffers::<1, _>"));
    }

    static EXPORT_BYTES: MethodSpec = MethodSpec {
        name: "exportStlBinary",
        kind: MethodKind::CustomBodyRaw,
        params: &[
            FacadeParam::ShapeId("id"),
            FacadeParam::Double("linearDeflection"),
        ],
        return_type: ReturnType::Bytes,
        occt_class: "",
        ctor_args: "",
        setup_code: "return exportStl(id, linearDeflection, false);",
        includes: &[],
        category: "io",
    };

    static IMPORT_BYTES: MethodSpec = MethodSpec {
        name: "importStlBinary",
        kind: MethodKind::CustomBodyRaw,
        params: &[FacadeParam::Bytes("data")],
        return_type: ReturnType::ShapeId,
        occt_class: "",
        ctor_args: "",
        setup_code: "return importStl(data);",
        includes: &[],
        category: "io",
    };

    #[test]
    fn bytes_param_writes_slice_directly() {
        let output = emit_rust_host(&[&IMPORT_BYTES]);
        assert!(output.contains("data: &[u8]"));
        assert!(output.contains("buffers.write(kernel, data)?"));
        assert!(output.contains("checked_wasm_len(data.len(), 1)?"));
        assert!(!output.contains("data.len() as u32"));
        assert!(!output.contains("data_ptr as i32"));
        assert!(output.contains("self.with_guest_buffers::<1, _>"));
        assert!(output.contains("fn_import_stl_binary: TypedFunc<(u32, u32), u32>"));
    }

    #[test]
    fn all_heap_lengths_are_checked_before_any_allocation() {
        let mut output = String::new();
        let params = &[
            FacadeParam::VectorShapeIds("shapes"),
            FacadeParam::VectorDouble("values"),
            FacadeParam::VectorInt("indices"),
            FacadeParam::String("name"),
        ];
        emit_wasm_call_preflight(&mut output, params);
        emit_wasm_call_allocations(&mut output, params);
        let preflight = concat!(
            "        let shapes_len = checked_wasm_len(shapes.len(), 4)?;\n",
            "        let values_len = checked_wasm_len(values.len(), 8)?;\n",
            "        let indices_len = checked_wasm_len(indices.len(), 4)?;\n",
            "        let name_len = checked_wasm_len(name.len(), 1)?;\n",
        );
        assert!(output.starts_with(preflight));
        assert!(output.contains("buffers.write(kernel, &shapes_bytes)?"));
        assert!(output.contains("buffers.write(kernel, &values_bytes)?"));
        assert!(output.contains("buffers.write(kernel, &indices_bytes)?"));
    }

    #[test]
    fn bytes_return_reads_raw_bytes() {
        let output = emit_rust_host(&[&EXPORT_BYTES]);
        assert!(output.contains("-> OcctResult<Vec<u8>>"));
        assert!(output.contains("self.read_bytes_result()"));
        assert!(!output.contains("self.read_string_result()"));
    }

    #[test]
    fn bool_return_method() {
        static IS_VALID: MethodSpec = MethodSpec {
            name: "isValid",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::ShapeId("id")],
            return_type: ReturnType::Bool,
            occt_class: "",
            ctor_args: "",
            setup_code: "BRepCheck_Analyzer checker(get(id));\nreturn checker.IsValid();",
            includes: &["BRepCheck_Analyzer.hxx"],
            category: "healing",
        };
        let output = emit_rust_host(&[&IS_VALID]);
        assert!(output.contains("pub fn is_valid(&mut self, id: ShapeHandle) -> OcctResult<bool>"));
        assert!(output.contains("Ok(result != 0)"));
    }

    #[test]
    fn struct_return_method_emits_reader() {
        // Exercises the `emit_struct_result` path: every struct-returning variant
        // shares the same status capture + `status < 0` guard + reader expression.
        static GET_BBOX: MethodSpec = MethodSpec {
            name: "getBoundingBox",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::ShapeId("id"), FacadeParam::Bool("useTri")],
            return_type: ReturnType::BBoxData,
            occt_class: "",
            ctor_args: "",
            setup_code: "return computeBBox(get(id), useTri);",
            includes: &[],
            category: "query",
        };
        let output = emit_rust_host(&[&GET_BBOX]);
        assert!(output.contains(
            "pub fn get_bounding_box(&mut self, id: ShapeHandle, use_tri: bool) -> OcctResult<BoundingBox>"
        ));
        assert!(output.contains("let status = self.generated.fn_get_bounding_box.call"));
        assert!(output.contains("if status < 0 {"));
        assert!(output.contains("self.read_bbox_result()"));
    }
}
