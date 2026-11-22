//! WASI C-ABI export emitter for the facade code generator.
//!
//! Generates `wasi_exports.cpp` — a set of `extern "C"` functions that wrap
//! `OcctKernel` methods for use with non-JS WASM runtimes (e.g. wasmtime).
//!
//! Design:
//! - Global kernel singleton (`occt_init` / `occt_destroy`)
//! - Error protocol: `occt_has_error` / `occt_get_error` / `occt_clear_error`
//! - Complex return types use static buffers + accessor functions
//! - All names are `snake_case` with `occt_` prefix

use std::fmt::Write as _;

use super::types::{FacadeParam, MethodSpec, ReturnType};

/// Convert `camelCase` to `snake_case`.
pub fn camel_to_snake(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                // Don't insert underscore between consecutive capitals
                // unless followed by a lowercase letter (e.g. "HLR" stays "hlr",
                // "glTF" stays "gl_tf")
                let prev_upper = s
                    .as_bytes()
                    .get(i.wrapping_sub(1))
                    .is_some_and(u8::is_ascii_uppercase);
                let next_lower = s.as_bytes().get(i + 1).is_some_and(u8::is_ascii_lowercase);
                if !prev_upper || next_lower {
                    result.push('_');
                }
            }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result
}

/// Format a parameter as a C-ABI formal parameter declaration.
///
/// String and vector types expand to pointer+length pairs.
fn param_to_c_abi(param: &FacadeParam) -> String {
    match param {
        FacadeParam::ShapeId(name) | FacadeParam::DocId(name) | FacadeParam::Uint32(name) => {
            format!("uint32_t {name}")
        }
        FacadeParam::Double(name) => format!("double {name}"),
        FacadeParam::Bool(name) | FacadeParam::Int(name) => format!("int32_t {name}"),
        FacadeParam::String(name) | FacadeParam::Bytes(name) => {
            format!("const char* {name}_ptr, uint32_t {name}_len")
        }
        FacadeParam::VectorShapeIds(name) => {
            format!("const uint32_t* {name}_ptr, uint32_t {name}_len")
        }
        FacadeParam::VectorDouble(name) => format!("const double* {name}_ptr, uint32_t {name}_len"),
        FacadeParam::VectorInt(name) => format!("const int32_t* {name}_ptr, uint32_t {name}_len"),
    }
}

/// Build the C-ABI parameter list for a function.
fn c_abi_param_list(params: &[FacadeParam]) -> String {
    if params.is_empty() {
        return String::new();
    }
    params
        .iter()
        .map(param_to_c_abi)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate the expression to pass a parameter to the kernel method.
///
/// String/vector types are reconstructed from ptr+len pairs.
fn param_to_call_arg(param: &FacadeParam) -> String {
    match param {
        FacadeParam::ShapeId(name)
        | FacadeParam::DocId(name)
        | FacadeParam::Double(name)
        | FacadeParam::Int(name)
        | FacadeParam::Uint32(name) => (*name).to_owned(),
        FacadeParam::Bool(name) => format!("({name} != 0)"),
        FacadeParam::String(name) | FacadeParam::Bytes(name) => {
            format!("std::string({name}_ptr, {name}_len)")
        }
        FacadeParam::VectorShapeIds(name) => {
            format!("std::vector<uint32_t>({name}_ptr, {name}_ptr + {name}_len)")
        }
        FacadeParam::VectorDouble(name) => {
            format!("std::vector<double>({name}_ptr, {name}_ptr + {name}_len)")
        }
        FacadeParam::VectorInt(name) => {
            format!("std::vector<int>({name}_ptr, {name}_ptr + {name}_len)")
        }
    }
}

/// Build the call arguments for a kernel method.
fn call_args(params: &[FacadeParam]) -> String {
    params
        .iter()
        .map(param_to_call_arg)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The C return type for a given `ReturnType`.
///
/// All non-scalar returns (strings, vectors, structs) cross the C ABI as an
/// `int32_t` status/length, with the payload read back out of band. The match
/// is exhaustive so a newly added `ReturnType` is a compile error rather than
/// silently falling through to `int32_t`.
const fn c_return_type(rt: ReturnType) -> &'static str {
    match rt {
        ReturnType::ShapeId | ReturnType::DocId | ReturnType::Uint32 => "uint32_t",
        ReturnType::Double => "double",
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
        | ReturnType::XCAFLabelInfo => "int32_t",
    }
}

/// The error sentinel value for a return type.
///
/// Exhaustive for the same reason as [`c_return_type`]: the `int32_t`-returning
/// variants share the `-1` sentinel, but each is listed so adding a variant
/// forces a deliberate choice.
const fn error_sentinel(rt: ReturnType) -> &'static str {
    match rt {
        ReturnType::ShapeId | ReturnType::DocId | ReturnType::Uint32 => "0",
        ReturnType::Double => "std::numeric_limits<double>::quiet_NaN()",
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
        | ReturnType::XCAFLabelInfo => "-1",
    }
}

/// Select how a facade result crosses the C ABI.
enum ResultTransfer {
    Direct,
    Boolean,
    Void,
    SizedBuffer(&'static str),
    AssignedBuffer(&'static str),
    ReconstructedBuffer {
        name: &'static str,
        ty: &'static str,
    },
    Curve,
}

const fn result_transfer(rt: ReturnType) -> ResultTransfer {
    use ResultTransfer::{
        AssignedBuffer, Boolean, Curve, Direct, ReconstructedBuffer, SizedBuffer, Void,
    };
    match rt {
        ReturnType::ShapeId
        | ReturnType::DocId
        | ReturnType::Uint32
        | ReturnType::Double
        | ReturnType::Int => Direct,
        ReturnType::Bool => Boolean,
        ReturnType::Void => Void,
        ReturnType::String | ReturnType::Bytes => SizedBuffer("g_string_buf"),
        ReturnType::VectorUint32 | ReturnType::VectorShapeIds => SizedBuffer("g_vec_u32_buf"),
        ReturnType::VectorDouble => SizedBuffer("g_vec_f64_buf"),
        ReturnType::VectorInt => SizedBuffer("g_vec_i32_buf"),
        ReturnType::BBoxData => AssignedBuffer("g_bbox_buf"),
        ReturnType::MeshData => ReconstructedBuffer {
            name: "g_mesh_buf",
            ty: "MeshData",
        },
        ReturnType::MeshBatchData => ReconstructedBuffer {
            name: "g_mesh_batch_buf",
            ty: "MeshBatchData",
        },
        ReturnType::EdgeData => ReconstructedBuffer {
            name: "g_edge_buf",
            ty: "EdgeData",
        },
        ReturnType::NurbsCurveData => AssignedBuffer("g_nurbs_buf"),
        ReturnType::CurveData => Curve,
        ReturnType::SurfaceData => AssignedBuffer("g_surface_buf"),
        ReturnType::EvolutionData => AssignedBuffer("g_evo_buf"),
        ReturnType::ProjectionData => AssignedBuffer("g_proj_buf"),
        ReturnType::XCAFLabelInfo => AssignedBuffer("g_label_info_buf"),
    }
}

/// Generate the body of a WASI export wrapper for one method.
fn emit_wasi_method(buf: &mut String, spec: &MethodSpec) {
    let snake_name = camel_to_snake(spec.name);
    let ret = c_return_type(spec.return_type);
    let params = c_abi_param_list(spec.params);
    let sentinel = error_sentinel(spec.return_type);
    let args = call_args(spec.params);
    let call = format!("g_kernel->{}({args})", spec.name);

    let _ = writeln!(buf, "{ret} occt_{snake_name}({params}) {{");
    let _ = writeln!(buf, "    CLEAR_ERROR();");
    let _ = writeln!(buf, "    try {{");
    match result_transfer(spec.return_type) {
        ResultTransfer::Direct => {
            let _ = writeln!(buf, "        return {call};");
        }
        ResultTransfer::Boolean => {
            let _ = writeln!(buf, "        return {call} ? 1 : 0;");
        }
        ResultTransfer::Void => {
            let _ = writeln!(buf, "        {call};");
            let _ = writeln!(buf, "        return 0;");
        }
        ResultTransfer::SizedBuffer(name) => {
            let _ = writeln!(buf, "        {name} = {call};");
            let _ = writeln!(buf, "        return static_cast<int32_t>({name}.size());");
        }
        ResultTransfer::AssignedBuffer(name) => {
            let _ = writeln!(buf, "        {name} = {call};");
            let _ = writeln!(buf, "        return 0;");
        }
        ResultTransfer::ReconstructedBuffer { name, ty } => {
            let _ = writeln!(buf, "        {name}.~{ty}();");
            let _ = writeln!(buf, "        new (&{name}) {ty}({call});");
            let _ = writeln!(buf, "        return 0;");
        }
        ResultTransfer::Curve => {
            let _ = writeln!(buf, "        g_curve_buf = {call};");
            // The curve's B-spline form crosses through the NURBS accessors.
            let _ = writeln!(buf, "        g_nurbs_buf = g_curve_buf.bspline;");
            let _ = writeln!(buf, "        return 0;");
        }
    }

    let _ = writeln!(buf, "    }} catch (const std::exception& e) {{");
    let _ = writeln!(buf, "        SET_ERROR(e.what());");
    let _ = writeln!(buf, "        return {sentinel};");
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

fn emit_surface_bspline_accessors(buf: &mut String) {
    for (export, field) in [
        ("degree_u", "degreeU"),
        ("degree_v", "degreeV"),
        ("nb_poles_u", "nbPolesU"),
        ("nb_poles_v", "nbPolesV"),
    ] {
        let _ = writeln!(
            buf,
            "int32_t occt_get_surface_{export}() {{ return g_surface_buf.bspline.{field}; }}"
        );
    }
    for (export, field) in [("periodic_u", "periodicU"), ("periodic_v", "periodicV")] {
        let _ = writeln!(
            buf,
            "int32_t occt_get_surface_{export}() {{ return g_surface_buf.bspline.{field} ? 1 : 0; }}"
        );
    }
    for (export, field) in [
        ("knots_u", "knotsU"),
        ("knots_v", "knotsV"),
        ("poles", "poles"),
        ("weights", "weights"),
    ] {
        let _ = writeln!(
            buf,
            "const double* occt_get_surface_{export}() {{ return g_surface_buf.bspline.{field}.data(); }}"
        );
        let _ = writeln!(
            buf,
            "uint32_t occt_get_surface_{export}_len() {{ return static_cast<uint32_t>(g_surface_buf.bspline.{field}.size()); }}"
        );
    }
    for (export, field) in [
        ("multiplicities_u", "multiplicitiesU"),
        ("multiplicities_v", "multiplicitiesV"),
    ] {
        let _ = writeln!(
            buf,
            "const int32_t* occt_get_surface_{export}() {{ return reinterpret_cast<const int32_t*>(g_surface_buf.bspline.{field}.data()); }}"
        );
        let _ = writeln!(
            buf,
            "uint32_t occt_get_surface_{export}_len() {{ return static_cast<uint32_t>(g_surface_buf.bspline.{field}.size()); }}"
        );
    }
    let _ = writeln!(buf);
}

fn emit_evolution_accessors(buf: &mut String) {
    // Evolution result accessors
    let _ = writeln!(buf, "// Evolution result");
    let _ = writeln!(
        buf,
        "uint32_t occt_get_evo_result_id() {{ return g_evo_buf.resultId; }}"
    );
    for (export, field) in [
        ("modified", "modified"),
        ("generated", "generated"),
        ("deleted", "deleted"),
        ("edge_modified", "edgeModified"),
        ("edge_generated", "edgeGenerated"),
        ("edge_deleted", "edgeDeleted"),
        ("first_shape", "firstShape"),
        ("last_shape", "lastShape"),
        ("spine_generated", "spineGenerated"),
        ("input_shapes", "inputShapes"),
        ("result_counts", "resultCounts"),
        ("modified_indexed", "modifiedIndexed"),
        ("generated_indexed", "generatedIndexed"),
        ("deleted_indexed", "deletedIndexed"),
        ("edge_modified_indexed", "edgeModifiedIndexed"),
        ("edge_generated_indexed", "edgeGeneratedIndexed"),
        ("edge_deleted_indexed", "edgeDeletedIndexed"),
        ("first_shape_indexed", "firstShapeIndexed"),
        ("last_shape_indexed", "lastShapeIndexed"),
        ("spine_generated_indexed", "spineGeneratedIndexed"),
    ] {
        let _ = writeln!(
            buf,
            "const int32_t* occt_get_evo_{export}() {{ return reinterpret_cast<const int32_t*>(g_evo_buf.{field}.data()); }}"
        );
        let _ = writeln!(
            buf,
            "uint32_t occt_get_evo_{export}_len() {{ return static_cast<uint32_t>(g_evo_buf.{field}.size()); }}"
        );
    }
    let _ = writeln!(buf);
}

/// Generate the contents of `facade/generated/wasi_exports.cpp`.
pub fn emit_wasi_exports(methods: &[&MethodSpec]) -> String {
    let mut buf = String::with_capacity(16384);

    buf.push_str(include_str!("templates/wasi_preamble.cpp.in"));
    buf.push('\n');

    // BBox accessors
    let _ = writeln!(buf, "// BBox result");
    for field in &["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"] {
        let _ = writeln!(
            buf,
            "double occt_get_bbox_{field}() {{ return g_bbox_buf.{field}; }}"
        );
    }
    let _ = writeln!(buf);

    buf.push_str(include_str!("templates/wasi_geometry_accessors.cpp.in"));

    emit_surface_bspline_accessors(&mut buf);
    emit_evolution_accessors(&mut buf);

    // Projection result accessors
    let _ = writeln!(buf, "// Projection result");
    for field in &[
        "visibleOutline",
        "visibleSmooth",
        "visibleSharp",
        "hiddenOutline",
        "hiddenSmooth",
        "hiddenSharp",
    ] {
        let snake = camel_to_snake(field);
        let _ = writeln!(
            buf,
            "uint32_t occt_get_proj_{snake}() {{ return g_proj_buf.{field}; }}"
        );
    }
    let _ = writeln!(buf);

    buf.push_str(include_str!("templates/wasi_label_info.cpp.in"));
    buf.push('\n');

    // Generated method wrappers
    let _ = writeln!(buf, "// === Kernel methods ===");
    let _ = writeln!(buf);

    for spec in methods {
        if !spec.has_wasi_binding() {
            continue;
        }
        emit_wasi_method(&mut buf, spec);
        let _ = writeln!(buf);
    }

    // Close extern "C"
    let _ = writeln!(buf, "}} // extern \"C\"");

    buf.trim_end().to_owned() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::types::MethodKind;

    #[test]
    fn camel_to_snake_basic() {
        assert_eq!(camel_to_snake("makeBox"), "make_box");
        assert_eq!(camel_to_snake("getBoundingBox"), "get_bounding_box");
        assert_eq!(camel_to_snake("fuse"), "fuse");
        assert_eq!(camel_to_snake("importStep"), "import_step");
        assert_eq!(camel_to_snake("getShapeType"), "get_shape_type");
        assert_eq!(camel_to_snake("xcafNewDocument"), "xcaf_new_document");
        assert_eq!(camel_to_snake("releaseAll"), "release_all");
        assert_eq!(camel_to_snake("fuseAll"), "fuse_all");
        assert_eq!(camel_to_snake("toBREP"), "to_brep");
        assert_eq!(camel_to_snake("fromBREP"), "from_brep");
        assert_eq!(camel_to_snake("xcafExportGLTF"), "xcaf_export_gltf");
    }

    #[test]
    fn emits_lifecycle_functions() {
        let output = emit_wasi_exports(&[]);
        assert!(output.contains("int32_t occt_init()"));
        assert!(output.contains("void occt_destroy()"));
        assert!(output.contains("int32_t occt_has_error()"));
        assert!(output.contains("void* occt_alloc(uint32_t size)"));
    }

    #[test]
    fn emits_simple_shape_method() {
        use super::super::types::{FacadeParam, MethodSpec, ReturnType};

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

        let output = emit_wasi_exports(&[&MAKE_BOX]);
        assert!(output.contains("uint32_t occt_make_box(double dx, double dy, double dz)"));
        assert!(output.contains("g_kernel->makeBox(dx, dy, dz)"));
        assert!(
            output.contains("return 0;"),
            "error sentinel for ShapeId should be 0"
        );
    }

    #[test]
    fn emits_string_param_method() {
        use super::super::types::{FacadeParam, MethodSpec, ReturnType};

        static IMPORT_STEP: MethodSpec = MethodSpec {
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

        let output = emit_wasi_exports(&[&IMPORT_STEP]);
        assert!(output.contains("occt_import_step(const char* data_ptr, uint32_t data_len)"));
        assert!(output.contains("std::string(data_ptr, data_len)"));
    }

    #[test]
    fn bytes_cross_the_c_abi_like_strings() {
        use super::super::types::{FacadeParam, MethodSpec, ReturnType};

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

        let output = emit_wasi_exports(&[&EXPORT_BYTES, &IMPORT_BYTES]);
        assert!(
            output.contains("int32_t occt_export_stl_binary(uint32_t id, double linearDeflection)")
        );
        assert!(output.contains("g_string_buf = g_kernel->exportStlBinary(id, linearDeflection);"));
        assert!(output.contains("occt_import_stl_binary(const char* data_ptr, uint32_t data_len)"));
        assert!(output.contains("importStlBinary(std::string(data_ptr, data_len))"));
    }

    #[test]
    fn emits_vector_param_method() {
        use super::super::types::{FacadeParam, MethodSpec, ReturnType};

        static FUSE_ALL: MethodSpec = MethodSpec {
            name: "fuseAll",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::VectorShapeIds("shapeIds")],
            return_type: ReturnType::ShapeId,
            occt_class: "",
            ctor_args: "",
            setup_code: "// body",
            includes: &[],
            category: "booleans",
        };

        let output = emit_wasi_exports(&[&FUSE_ALL]);
        assert!(
            output.contains("occt_fuse_all(const uint32_t* shapeIds_ptr, uint32_t shapeIds_len)")
        );
        assert!(
            output.contains("std::vector<uint32_t>(shapeIds_ptr, shapeIds_ptr + shapeIds_len)")
        );
    }

    #[test]
    fn emits_string_return_method() {
        use super::super::types::{FacadeParam, MethodSpec, ReturnType};

        static EXPORT_STEP: MethodSpec = MethodSpec {
            name: "exportStep",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::ShapeId("id")],
            return_type: ReturnType::String,
            occt_class: "",
            ctor_args: "",
            setup_code: "return exportStepImpl(get(id));",
            includes: &[],
            category: "io",
        };

        let output = emit_wasi_exports(&[&EXPORT_STEP]);
        assert!(output.contains("int32_t occt_export_step(uint32_t id)"));
        assert!(output.contains("g_string_buf = g_kernel->exportStep(id)"));
        assert!(output.contains("g_string_buf.size()"));
    }
}
