//! Rust host API emitter for the facade code generator.
//!
//! Generates `kernel_generated.rs` — the Rust wrapper methods that call
//! into the WASM module via wasmtime. Each `MethodSpec` produces one
//! `pub fn method_name(&mut self, ...) -> OcctResult<T>` method on `OcctKernel`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::emitter::group_by_category;
use super::types::{FacadeParam, MethodSpec, ReturnType};
use super::wasi_emitter::camel_to_snake;

/// Rust API representation of a facade parameter.
const fn param_type(param: &FacadeParam) -> &'static str {
    match param {
        FacadeParam::ShapeId(_) => "ShapeHandle",
        FacadeParam::DocId(_) => "DocumentHandle",
        FacadeParam::Double(_) => "f64",
        FacadeParam::Bool(_) => "bool",
        FacadeParam::Int(_) => "i32",
        FacadeParam::Uint32(_) => "u32",
        FacadeParam::String(_) => "&str",
        FacadeParam::Bytes(_) => "&[u8]",
        FacadeParam::VectorShapeIds(_) => "&[ShapeHandle]",
        FacadeParam::VectorDouble(_) => "&[f64]",
        FacadeParam::VectorInt(_) => "&[i32]",
    }
}

fn param_to_rust(param: &FacadeParam) -> String {
    format!("{}: {}", rust_param_name(param.name()), param_type(param))
}

/// The Rust method name is independent from the guest symbol.
fn method_name(spec: &MethodSpec) -> String {
    match spec.name {
        "toBrep" => "export_brep".to_owned(),
        "fromBrep" => "import_brep".to_owned(),
        "toBrepBinary" => "export_brep_binary".to_owned(),
        "fromBrepBinary" => "import_brep_binary".to_owned(),
        name => camel_to_snake(name),
    }
}

/// Wide operations require named parameters: many of their scalars and arrays
/// have the same physical type but different geometric meaning.
fn request_name(spec: &MethodSpec) -> Option<String> {
    (spec.params.len() > 6).then(|| format!("{}Request", type_name(&method_name(spec))))
}

fn request_borrows(spec: &MethodSpec) -> bool {
    spec.params
        .iter()
        .any(|param| param_type(param).starts_with('&'))
}

fn emit_requests(buf: &mut String, methods: &[&MethodSpec]) {
    for spec in methods {
        if !spec.has_wasi_binding() {
            continue;
        }
        let Some(name) = request_name(spec) else {
            continue;
        };
        let lifetime = if request_borrows(spec) { "<'a>" } else { "" };
        let _ = writeln!(
            buf,
            "/// Named arguments for the `{}` facade operation.",
            spec.name
        );
        let _ = writeln!(buf, "#[derive(Debug, Clone, Copy)]");
        let _ = writeln!(buf, "pub struct {name}{lifetime} {{");
        for param in spec.params {
            let field = rust_param_name(param.name());
            let ty = param_type(param);
            let ty = ty
                .strip_prefix('&')
                .map_or_else(|| ty.to_owned(), |target| format!("&'a {target}"));
            let _ = writeln!(
                buf,
                "    /// Value of the facade parameter `{}`.",
                param.name()
            );
            let _ = writeln!(buf, "    pub {field}: {ty},");
        }
        buf.push_str("}\n\n");
    }
}

fn arg_expr(param: &FacadeParam, named: bool) -> String {
    let name = rust_param_name(param.name());
    if named {
        format!("request.{name}")
    } else {
        name
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

/// Validate all heap lengths before allocating any guest memory.
fn emit_wasm_call_preflight(buf: &mut String, params: &[FacadeParam], named: bool) {
    let mut lengths = params
        .iter()
        .filter_map(|param| {
            let width = heap_element_width(param)?;
            let expr = arg_expr(param, named);
            Some(format!("checked_wasm_len({expr}.len(), {width})?"))
        })
        .peekable();
    if lengths.peek().is_some() {
        let _ = writeln!(
            buf,
            "        let guest_lengths = [{}];",
            lengths.collect::<Vec<_>>().join(", ")
        );
    }
}

/// Allocate guest buffers inside a single cleanup scope. The two arrays
/// share the ABI order of the heap-bearing parameters.
fn emit_wasm_call_allocations(buf: &mut String, params: &[FacadeParam], named: bool) {
    let allocations = params
        .iter()
        .filter_map(|param| {
            let expr = arg_expr(param, named);
            let data = match param {
                FacadeParam::String(_) => format!("{expr}.as_bytes()"),
                FacadeParam::Bytes(_) => expr,
                FacadeParam::VectorShapeIds(_)
                | FacadeParam::VectorDouble(_)
                | FacadeParam::VectorInt(_) => {
                    let value = if matches!(param, FacadeParam::VectorShapeIds(_)) {
                        "v.0"
                    } else {
                        "v"
                    };
                    format!(
                        "&{expr}.iter().flat_map(|v| {value}.to_le_bytes()).collect::<Vec<_>>()"
                    )
                }
                _ => return None,
            };
            Some(format!("buffers.write(kernel, {data})?"))
        })
        .collect::<Vec<_>>();
    if !allocations.is_empty() {
        let _ = writeln!(
            buf,
            "            let guest_ptrs = [{}];",
            allocations.join(", ")
        );
    }
}

/// Generate ABI scalars in the original facade order.
fn emit_wasm_call_args(params: &[FacadeParam], named: bool) -> String {
    let mut heap_index = 0;
    params
        .iter()
        .flat_map(|param| {
            let expr = arg_expr(param, named);
            match param {
                FacadeParam::ShapeId(_) | FacadeParam::DocId(_) => vec![format!("{expr}.0")],
                FacadeParam::Double(_) | FacadeParam::Int(_) | FacadeParam::Uint32(_) => vec![expr],
                FacadeParam::Bool(_) => vec![format!("i32::from({expr})")],
                _ => {
                    let index = heap_index;
                    heap_index += 1;
                    vec![
                        format!("guest_ptrs[{index}]"),
                        format!("guest_lengths[{index}]"),
                    ]
                }
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate the wasmtime `TypedFunc` type for a method's parameters.
fn wasm_typed_func_type(spec: &MethodSpec) -> String {
    let param_count = spec.flattened_param_count();

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
    let snake_name = method_name(spec);
    let guest_name = camel_to_snake(spec.name);
    let request = request_name(spec);
    let named = request.is_some();
    let rust_ret = rust_return_type(spec.return_type);
    let rust_params = if let Some(request) = request {
        let lifetime = if request_borrows(spec) { "<'_>" } else { "" };
        format!("request: {request}{lifetime}")
    } else {
        rust_param_list(spec.params)
    };

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

    emit_wasm_call_preflight(buf, spec.params, named);

    // Build the call arguments
    let call_args = emit_wasm_call_args(spec.params, named);
    let call_tuple = if spec.flattened_param_count() == 0 {
        "()".to_owned()
    } else if spec.flattened_param_count() == 1 {
        format!("({call_args},)")
    } else {
        format!("({call_args})")
    };

    let fn_field = format!("generated.{}.{guest_name}", spec.category);
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
        emit_wasm_call_allocations(buf, spec.params, named);
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

/// Convert a generated snake-case identifier to a Rust type name.
fn type_name(name: &str) -> String {
    name.split('_')
        .filter_map(|part| {
            let mut chars = part.chars();
            let first = chars.next()?;
            Some(first.to_ascii_uppercase().to_string() + chars.as_str())
        })
        .collect()
}

fn export_group(category: &str) -> String {
    format!("{}Exports", type_name(category))
}

/// Intern long ABI signatures. The Wasmtime call shape is unchanged, and
/// identical physical signatures share one alias rather than repeating tuples.
fn signature_aliases(methods: &[&MethodSpec]) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::new();
    for spec in methods {
        if spec.has_wasi_binding() && spec.flattened_param_count() >= 7 {
            let signature = wasm_typed_func_type(spec);
            aliases
                .entry(signature)
                .or_insert_with(|| format!("{}Abi", type_name(&camel_to_snake(spec.name))));
        }
    }
    aliases
}

fn emit_export_groups(buf: &mut String, methods: &[&MethodSpec]) {
    let aliases = signature_aliases(methods);
    for (signature, name) in &aliases {
        // The alias describes the exact guest ABI, not a second wrapper type.
        let _ = writeln!(buf, "type {name} = {signature};");
    }
    let _ = writeln!(buf);
    let generable = methods
        .iter()
        .copied()
        .filter(|spec| spec.has_wasi_binding())
        .collect::<Vec<_>>();
    let groups = group_by_category(&generable);
    for (category, specs) in &groups {
        let name = export_group(category);
        let _ = writeln!(buf, "struct {name} {{");
        for spec in specs {
            if !spec.has_wasi_binding() {
                continue;
            }
            let field = camel_to_snake(spec.name);
            let signature = wasm_typed_func_type(spec);
            let ty = aliases.get(&signature).unwrap_or(&signature);
            let _ = writeln!(buf, "    {field}: {ty},");
        }
        let _ = writeln!(buf, "}}");
        let _ = writeln!(buf, "impl {name} {{");
        buf.push_str(
            "    fn resolve<T: 'static>(instance: &wasmtime::Instance, store: &mut wasmtime::Store<T>) -> OcctResult<Self> {\n        Ok(Self {\n",
        );
        for spec in specs {
            if spec.has_wasi_binding() {
                let field = camel_to_snake(spec.name);
                let _ = writeln!(
                    buf,
                    "            {field}: instance.get_typed_func(&mut *store, \"occt_{field}\")?,"
                );
            }
        }
        buf.push_str("        })\n    }\n}\n\n");
    }
    buf.push_str("/// Cached typed WASM exports, grouped by facade domain.\npub(super) struct GeneratedFuncs {\n");
    for (category, _) in &groups {
        let name = export_group(category);
        let _ = writeln!(buf, "    {category}: {name},");
    }
    buf.push_str("}\n\nimpl GeneratedFuncs {\n    pub(super) fn resolve<T: 'static>(instance: &wasmtime::Instance, store: &mut wasmtime::Store<T>) -> OcctResult<Self> {\n        Ok(Self {\n");
    for (category, _) in &groups {
        let name = export_group(category);
        let _ = writeln!(
            buf,
            "            {category}: {name}::resolve(instance, store)?,"
        );
    }
    buf.push_str("        })\n    }\n}\n\n");
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

    emit_requests(&mut buf, methods);
    emit_export_groups(&mut buf, methods);

    // Method implementations
    buf.push_str(
        r"/// Generated kernel methods.
///
/// All facade methods wrapped as safe Rust functions.
",
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
        assert!(output.contains("generated.primitives.make_box"));
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
        assert!(output.contains("make_box: TypedFunc<(f64, f64, f64), u32>"));
        assert!(output.contains("fuse: TypedFunc<(u32, u32), u32>"));
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
        assert!(output.contains("import_stl_binary: TypedFunc<(u32, u32), u32>"));
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
        emit_wasm_call_preflight(&mut output, params, false);
        emit_wasm_call_allocations(&mut output, params, false);
        let preflight = "        let guest_lengths = [checked_wasm_len(shapes.len(), 4)?, checked_wasm_len(values.len(), 8)?, checked_wasm_len(indices.len(), 4)?, checked_wasm_len(name.len(), 1)?];";
        assert!(output.starts_with(preflight));
        assert!(output.contains("let guest_ptrs = ["));
        assert!(output.contains("shapes.iter().flat_map(|v| v.0.to_le_bytes())"));
        assert!(output.contains("values.iter().flat_map(|v| v.to_le_bytes())"));
        assert!(output.contains("indices.iter().flat_map(|v| v.to_le_bytes())"));
        assert_eq!(
            emit_wasm_call_args(params, false),
            "guest_ptrs[0], guest_lengths[0], guest_ptrs[1], guest_lengths[1], guest_ptrs[2], guest_lengths[2], guest_ptrs[3], guest_lengths[3]"
        );
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
        assert!(output.contains("let status = self.generated.query.get_bounding_box.call"));
        assert!(output.contains("if status < 0 {"));
        assert!(output.contains("self.read_bbox_result()"));
    }
}
