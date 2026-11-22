//! C++ code emitter for the facade code generator.
//!
//! Emits `kernel.cpp` (method implementations) and `bindings.cpp` (Embind
//! reference) from a slice of [`MethodSpec`] descriptors.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::types::{FacadeParam, MethodKind, MethodSpec, ReturnType};

/// Format a [`FacadeParam`] as a C++ formal parameter declaration.
fn param_to_cpp(param: &FacadeParam) -> String {
    match param {
        FacadeParam::ShapeId(name) | FacadeParam::DocId(name) | FacadeParam::Uint32(name) => {
            format!("uint32_t {name}")
        }
        FacadeParam::Double(name) => format!("double {name}"),
        FacadeParam::VectorShapeIds(name) => format!("std::vector<uint32_t> {name}"),
        FacadeParam::Bool(name) => format!("bool {name}"),
        FacadeParam::Int(name) => format!("int {name}"),
        FacadeParam::String(name) | FacadeParam::Bytes(name) => {
            format!("const std::string& {name}")
        }
        FacadeParam::VectorDouble(name) => format!("std::vector<double> {name}"),
        FacadeParam::VectorInt(name) => format!("std::vector<int> {name}"),
    }
}

/// Build the C++ parameter list string for a method signature.
fn param_list(params: &[FacadeParam]) -> String {
    params
        .iter()
        .map(param_to_cpp)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Emit a `SimpleShape` method body.
fn emit_simple_shape(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let cls = spec.occt_class;
    let args = spec.ctor_args;

    let _ = writeln!(
        buf,
        "uint32_t OcctKernel::{name}({params}) {{",
        params = param_list(spec.params)
    );
    let _ = writeln!(buf, "    try {{");
    let _ = writeln!(buf, "        {cls} maker({args});");
    let _ = writeln!(buf, "        maker.Build();");
    let _ = writeln!(buf, "        if (!maker.IsDone()) {{");
    let _ = writeln!(
        buf,
        "            throw std::runtime_error(\"{name}: construction failed\");"
    );
    let _ = writeln!(buf, "        }}");
    let _ = writeln!(buf, "        return store(maker.Shape());");
    let _ = writeln!(buf, "    }} catch (const Standard_Failure& e) {{");
    let _ = writeln!(
        buf,
        "        throw std::runtime_error(std::string(\"{name}: \") + e.what());"
    );
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

/// Emit a `BooleanOp` method body.
///
/// The `ctor_args` field in the spec already contains the full expression
/// (e.g. `"get(a), get(b)"`), so we pass it directly to the OCCT constructor.
fn emit_boolean_op(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let cls = spec.occt_class;
    let args = spec.ctor_args;

    let _ = writeln!(
        buf,
        "uint32_t OcctKernel::{name}({params}) {{",
        params = param_list(spec.params)
    );
    let _ = writeln!(buf, "    try {{");
    let _ = writeln!(buf, "        {cls} op;");
    let _ = writeln!(buf, "        buildBoolean(op, {args});");
    if cls == "BRepAlgoAPI_Fuse" {
        let _ = writeln!(buf, "        recoverLostFuse(op, {{{args}}});");
    }
    let _ = writeln!(buf, "        if (!op.IsDone() || op.HasErrors()) {{");
    let _ = writeln!(
        buf,
        "            throw std::runtime_error(\"{name}: boolean operation failed\");"
    );
    let _ = writeln!(buf, "        }}");
    let _ = writeln!(buf, "        return store(op.Shape());");
    let _ = writeln!(buf, "    }} catch (const Standard_Failure& e) {{");
    let _ = writeln!(
        buf,
        "        throw std::runtime_error(std::string(\"{name}: \") + e.what());"
    );
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

/// Emit a `FilletLike` method body.
///
/// Expects params in order: `ShapeId` (solid), `VectorShapeIds` (edges/faces),
/// then one or more scalar params (radius, distance, etc.).
fn emit_fillet_like(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let cls = spec.occt_class;
    let args = spec.ctor_args;

    let _ = writeln!(
        buf,
        "uint32_t OcctKernel::{name}({params}) {{",
        params = param_list(spec.params)
    );
    let _ = writeln!(buf, "    try {{");
    let _ = writeln!(buf, "        {cls} maker({args});");

    // Find the vector param and the scalar param(s) for the Add() call.
    let vec_param = spec.params.iter().find_map(|p| {
        if let FacadeParam::VectorShapeIds(n) = p {
            Some(*n)
        } else {
            None
        }
    });
    let scalar_params: Vec<&str> = spec
        .params
        .iter()
        .filter_map(|p| match p {
            FacadeParam::Double(n) | FacadeParam::Int(n) => Some(*n),
            _ => None,
        })
        .collect();

    let solid_param = spec.params.iter().find_map(|p| {
        if let FacadeParam::ShapeId(n) = p {
            Some(*n)
        } else {
            None
        }
    });

    let failure = vec_param.map_or_else(
        || format!("std::string(\"{name}: operation failed\")"),
        |vec_name| {
            let scalar_args = scalar_params.join(", ");
            let _ = writeln!(buf, "        std::vector<TopoDS_Edge> edges;");
            let _ = writeln!(buf, "        edges.reserve({vec_name}.size());");
            let _ = writeln!(buf, "        for (uint32_t eid : {vec_name}) {{");
            let _ = writeln!(buf, "            edges.push_back(TopoDS::Edge(get(eid)));");
            let _ = writeln!(buf, "            maker.Add({scalar_args}, edges.back());");
            let _ = writeln!(buf, "        }}");
            match (cls, solid_param, scalar_params.first()) {
                ("BRepFilletAPI_MakeFillet", _, _) => {
                    format!("filletFailure(maker, {vec_name}, edges, \"{name}\")")
                }
                ("BRepFilletAPI_MakeChamfer", Some(solid), Some(distance)) => format!(
                    "chamferFailure(get({solid}), {distance}, {vec_name}, edges, \"{name}\")"
                ),
                _ => format!("std::string(\"{name}: operation failed\")"),
            }
        },
    );

    let _ = writeln!(buf, "        maker.Build();");
    let _ = writeln!(buf, "        if (!maker.IsDone()) {{");
    let _ = writeln!(buf, "            throw std::runtime_error({failure});");
    let _ = writeln!(buf, "        }}");
    let _ = writeln!(
        buf,
        "        return store(validateFilletResult(unwrapSingletonSolid(maker.Shape()), \"{name}\", true));"
    );
    let _ = writeln!(buf, "    }} catch (const Standard_Failure& e) {{");
    let _ = writeln!(
        buf,
        "        throw std::runtime_error(std::string(\"{name}: \") + e.what());"
    );
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

/// Emit a `SetupShape` method body.
///
/// Emits `setup_code` verbatim, then constructs the OCCT class with `ctor_args`,
/// and stores the result. No `Build()`/`IsDone()` check.
fn emit_setup_shape(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let cls = spec.occt_class;
    let args = spec.ctor_args;

    let _ = writeln!(
        buf,
        "uint32_t OcctKernel::{name}({params}) {{",
        params = param_list(spec.params)
    );
    let _ = writeln!(buf, "    try {{");

    // Emit setup code lines with proper indentation.
    if !spec.setup_code.is_empty() {
        for line in spec.setup_code.lines() {
            let _ = writeln!(buf, "        {line}");
        }
    }

    let _ = writeln!(buf, "        {cls} maker({args});");
    let _ = writeln!(buf, "        return store(maker.Shape());");
    let _ = writeln!(buf, "    }} catch (const Standard_Failure& e) {{");
    let _ = writeln!(
        buf,
        "        throw std::runtime_error(std::string(\"{name}: \") + e.what());"
    );
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

/// Map a `ReturnType` to its C++ type spelling.
const fn cpp_return_type(ret: ReturnType) -> &'static str {
    match ret {
        ReturnType::ShapeId | ReturnType::DocId | ReturnType::Uint32 => "uint32_t",
        ReturnType::Bool => "bool",
        ReturnType::Void => "void",
        ReturnType::VectorUint32 | ReturnType::VectorShapeIds | ReturnType::StepImport => {
            "std::vector<uint32_t>"
        }
        ReturnType::VectorDouble | ReturnType::BRepProperties => "std::vector<double>",
        ReturnType::Double => "double",
        ReturnType::String | ReturnType::Bytes => "std::string",
        ReturnType::Int => "int",
        ReturnType::VectorInt => "std::vector<int>",
        ReturnType::BBoxData => "BBoxData",
        ReturnType::NurbsCurveData => "NurbsCurveData",
        ReturnType::CurveData => "CurveData",
        ReturnType::SurfaceData => "SurfaceData",
        ReturnType::EvolutionData => "EvolutionData",
        ReturnType::MeshData => "MeshData",
        ReturnType::MeshBatchData => "MeshBatchData",
        ReturnType::EdgeData => "EdgeData",
        ReturnType::ProjectionData => "ProjectionData",
        ReturnType::XCAFLabelInfo => "XCAFLabelInfo",
    }
}

/// Emit a `CustomBody` method — the `setup_code` field contains the complete body.
fn emit_custom_body(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let ret_type = cpp_return_type(spec.return_type);

    let _ = writeln!(
        buf,
        "{ret_type} OcctKernel::{name}({params}) {{",
        params = param_list(spec.params)
    );
    let _ = writeln!(buf, "    try {{");

    for line in spec.setup_code.lines() {
        let _ = writeln!(buf, "        {line}");
    }

    let _ = writeln!(buf, "    }} catch (const Standard_Failure& e) {{");
    let _ = writeln!(
        buf,
        "        throw std::runtime_error(std::string(\"{name}: \") + e.what());"
    );
    let _ = writeln!(buf, "    }}");
    let _ = writeln!(buf, "}}");
}

/// Emit a `CustomBodyRaw` method — `setup_code` is the full body, emitted with
/// no `Standard_Failure` try/catch wrapper (for methods that never enter OCCT).
fn emit_custom_body_raw(buf: &mut String, spec: &MethodSpec) {
    let ret_type = cpp_return_type(spec.return_type);

    let _ = writeln!(
        buf,
        "{ret_type} OcctKernel::{name}({params}) {{",
        name = spec.name,
        params = param_list(spec.params)
    );

    for line in spec.setup_code.lines() {
        let _ = writeln!(buf, "    {line}");
    }

    let _ = writeln!(buf, "}}");
}

/// Derive the OCCT include header for a class name (e.g. `BRepPrimAPI_MakeBox`
/// becomes `<BRepPrimAPI_MakeBox.hxx>`).
fn class_to_include(cls: &str) -> String {
    format!("{cls}.hxx")
}

/// Collect all unique `#include` paths needed by the given methods.
fn collect_includes(methods: &[&MethodSpec]) -> BTreeSet<String> {
    let mut includes = BTreeSet::new();

    // Always-needed headers.
    includes.insert("Standard_Failure.hxx".to_owned());

    if methods
        .iter()
        .any(|m| matches!(m.return_type, ReturnType::EvolutionData))
    {
        includes.insert("TopExp.hxx".to_owned());
        includes.insert("TopExp_Explorer.hxx".to_owned());
        includes.insert("NCollection_IndexedMap.hxx".to_owned());
        includes.insert("TopTools_ShapeMapHasher.hxx".to_owned());
    }

    if needs_unwrap_singleton_solid(methods) {
        includes.insert("TopExp_Explorer.hxx".to_owned());
        includes.insert("TopoDS_Shape.hxx".to_owned());
        // validateFilletResult, emitted alongside unwrapSingletonSolid.
        includes.insert("BRepCheck_Analyzer.hxx".to_owned());
        includes.insert("ShapeFix_Shape.hxx".to_owned());
    }

    if needs_solid_join_type(methods) {
        includes.insert("GeomAbs_JoinType.hxx".to_owned());
    }

    if needs_faulty_edges(methods) {
        includes.insert("BRepAdaptor_Curve.hxx".to_owned());
        includes.insert("BRepFilletAPI_MakeFillet.hxx".to_owned());
        includes.insert("BRep_Tool.hxx".to_owned());
        includes.insert("ChFi3d_ChBuilder.hxx".to_owned());
        includes.insert("Extrema_GenLocateExtPS.hxx".to_owned());
        includes.insert("Extrema_POnSurf.hxx".to_owned());
        includes.insert("Geom2d_Curve.hxx".to_owned());
        includes.insert("GeomAdaptor_Surface.hxx".to_owned());
        includes.insert("Geom_Surface.hxx".to_owned());
        includes.insert("gp.hxx".to_owned());
        includes.insert("gp_Pnt.hxx".to_owned());
        includes.insert("gp_Pnt2d.hxx".to_owned());
        includes.insert("gp_Vec.hxx".to_owned());
        includes.insert("NCollection_DataMap.hxx".to_owned());
        includes.insert("NCollection_IndexedDataMap.hxx".to_owned());
        includes.insert("NCollection_List.hxx".to_owned());
        includes.insert("Precision.hxx".to_owned());
        includes.insert("TopAbs_Orientation.hxx".to_owned());
        includes.insert("TopExp.hxx".to_owned());
        includes.insert("TopExp_Explorer.hxx".to_owned());
        includes.insert("TopTools_ShapeMapHasher.hxx".to_owned());
        includes.insert("TopoDS.hxx".to_owned());
        includes.insert("TopoDS_Edge.hxx".to_owned());
        includes.insert("TopoDS_Face.hxx".to_owned());
    }

    if needs_build_boolean(methods) {
        includes.insert("GProp_GProps.hxx".to_owned());
        includes.insert("NCollection_IndexedMap.hxx".to_owned());
        includes.insert("TopExp.hxx".to_owned());
        includes.insert("TopExp_Explorer.hxx".to_owned());
        includes.insert("TopTools_ShapeMapHasher.hxx".to_owned());
        includes.insert("algorithm".to_owned());
        includes.insert("cmath".to_owned());
        includes.insert("vector".to_owned());
        includes.insert("BRepAlgoAPI_BooleanOperation.hxx".to_owned());
        includes.insert("NCollection_List.hxx".to_owned());
        includes.insert("TopoDS_Shape.hxx".to_owned());
    }

    if needs_finite_check(methods) {
        includes.insert("BRep_Tool.hxx".to_owned());
        includes.insert("NCollection_List.hxx".to_owned());
        includes.insert("Precision.hxx".to_owned());
        includes.insert("TopExp_Explorer.hxx".to_owned());
        includes.insert("TopoDS.hxx".to_owned());
        includes.insert("TopoDS_Shape.hxx".to_owned());
    }

    for spec in methods {
        if matches!(spec.kind, MethodKind::Skip) {
            continue;
        }
        if !spec.occt_class.is_empty() {
            includes.insert(class_to_include(spec.occt_class));
        }
        for inc in spec.includes {
            includes.insert((*inc).to_owned());
        }

        // FilletLike methods need TopoDS.hxx for downcasting.
        if matches!(spec.kind, MethodKind::FilletLike) {
            includes.insert("TopoDS.hxx".to_owned());
        }
    }
    includes
}

/// Group methods by category, preserving insertion order within each group.
pub(super) fn group_by_category<'a>(
    methods: &[&'a MethodSpec],
) -> Vec<(&'a str, Vec<&'a MethodSpec>)> {
    let mut groups: Vec<(&str, Vec<&MethodSpec>)> = Vec::new();
    for spec in methods {
        if matches!(spec.kind, MethodKind::Skip) {
            continue;
        }
        if let Some(group) = groups.iter_mut().find(|(cat, _)| *cat == spec.category) {
            group.1.push(spec);
        } else {
            groups.push((spec.category, vec![spec]));
        }
    }
    groups
}

/// Do any methods need the `unwrapSingletonSolid` helper?
///
/// Every `FilletLike` method calls it, plus any `CustomBody` spec that names it.
fn needs_unwrap_singleton_solid(methods: &[&MethodSpec]) -> bool {
    methods.iter().any(|m| {
        matches!(m.kind, MethodKind::FilletLike) || m.setup_code.contains("unwrapSingletonSolid")
    })
}

/// Do any methods need the `filletFailure` / `chamferFailure` helpers?
///
/// The `FilletLike` template calls them, as does any `CustomBody` spec that names them.
fn needs_faulty_edges(methods: &[&MethodSpec]) -> bool {
    methods.iter().any(|m| {
        matches!(m.kind, MethodKind::FilletLike)
            || m.setup_code.contains("filletFailure(")
            || m.setup_code.contains("chamferFailure(")
            || m.setup_code.contains("chamferEdgesFailure(")
            || m.setup_code.contains("addFilletRadii(")
    })
}

/// Do any methods need the `buildBoolean` helper?
///
/// Every `BooleanOp` method calls it, plus any `CustomBody` spec that names it.
fn needs_build_boolean(methods: &[&MethodSpec]) -> bool {
    methods
        .iter()
        .any(|m| matches!(m.kind, MethodKind::BooleanOp) || m.setup_code.contains("buildBoolean("))
}

/// Do any methods need the `isFiniteShape` / `allFinite` OBB guard?
///
/// `buildBoolean` uses it, as do the n-way booleans that name `allFinite`.
fn needs_finite_check(methods: &[&MethodSpec]) -> bool {
    needs_build_boolean(methods) || methods.iter().any(|m| m.setup_code.contains("allFinite("))
}

fn needs_solid_join_type(methods: &[&MethodSpec]) -> bool {
    methods
        .iter()
        .any(|m| m.setup_code.contains("solidJoinType"))
}

fn any_body_calls(methods: &[&MethodSpec], helper: &str) -> bool {
    methods.iter().any(|m| m.setup_code.contains(helper))
}

/// Emit ordered C++ fragments only when a generated facade body needs them.
fn emit_referenced_templates(
    buf: &mut String,
    methods: &[&MethodSpec],
    templates: &[(&str, &str)],
) {
    for (symbol, source) in templates {
        if any_body_calls(methods, symbol) {
            buf.push_str(source);
            buf.push('\n');
        }
    }
}

/// Emit the geometry readers' helpers, each only if a method body calls it.
fn emit_geometry_helpers(buf: &mut String, methods: &[&MethodSpec]) {
    emit_referenced_templates(
        buf,
        methods,
        &[
            (
                "curveKindName(",
                include_str!("templates/curve_kind.cpp.in"),
            ),
            (
                "surfaceKindName(",
                include_str!("templates/surface_kind.cpp.in"),
            ),
            (
                "toNurbsCurveData(",
                include_str!("templates/nurbs_curve.cpp.in"),
            ),
            (
                "toNurbsSurfaceData(",
                include_str!("templates/nurbs_surface.cpp.in"),
            ),
        ],
    );
}

/// Emit static helper functions that generated methods depend on.
///
/// Emits `unwrapSingletonSolid` + `validateFilletResult` for the fillet/chamfer
/// family, `solidJoinType` for the 3D offset join variants, `buildBoolean` for
/// the two-operand booleans, `buildEvolution` if any method returns
/// `EvolutionData`, and the geometry readers' helpers.
fn emit_helper_functions(buf: &mut String, methods: &[&MethodSpec]) {
    let needs_evolution = methods
        .iter()
        .any(|m| matches!(m.return_type, ReturnType::EvolutionData));
    let needs_unwrap = needs_unwrap_singleton_solid(methods);
    let needs_join = needs_solid_join_type(methods);
    let needs_boolean = needs_build_boolean(methods);
    let needs_finite = needs_finite_check(methods);
    let needs_faulty = needs_faulty_edges(methods);

    let needs_geometry = [
        "curveKindName(",
        "surfaceKindName(",
        "toNurbsCurveData(",
        "toNurbsSurfaceData(",
    ]
    .iter()
    .any(|helper| any_body_calls(methods, helper));

    if needs_evolution
        || needs_unwrap
        || needs_join
        || needs_boolean
        || needs_finite
        || needs_faulty
        || needs_geometry
    {
        let _ = writeln!(buf, "// === helper functions ===");
        let _ = writeln!(buf);
    }

    emit_geometry_helpers(buf, methods);

    if needs_join {
        buf.push_str(include_str!("templates/solid_join_type.cpp.in"));
        let _ = writeln!(buf);
    }

    if needs_finite {
        buf.push_str(include_str!("templates/finite_shape.cpp.in"));
        let _ = writeln!(buf);
    }

    if needs_boolean {
        buf.push_str(include_str!("templates/boolean.cpp.in"));
        let _ = writeln!(buf);
    }

    if needs_faulty {
        buf.push_str(include_str!("templates/fillet_chamfer_diagnostics.cpp.in"));
        let _ = writeln!(buf);
    }

    if needs_unwrap {
        buf.push_str(include_str!("templates/unwrap_singleton_solid.cpp.in"));
        let _ = writeln!(buf);

        buf.push_str(include_str!("templates/fillet_validation.cpp.in"));
        let _ = writeln!(buf);
    }

    if needs_evolution {
        buf.push_str(include_str!("templates/evolution.cpp.in"));
        let _ = writeln!(buf);
    }

    emit_sweep_helpers(buf, methods);
}

/// Emit the helpers behind the sweeps' history, each only if a method body calls it.
fn emit_sweep_helpers(buf: &mut String, methods: &[&MethodSpec]) {
    emit_referenced_templates(
        buf,
        methods,
        &[
            ("buildSweep(", include_str!("templates/sweep.cpp.in")),
            (
                "appendSweepCaps(",
                include_str!("templates/sweep_caps.cpp.in"),
            ),
            (
                "appendSpineGenerated(",
                include_str!("templates/spine_generated.cpp.in"),
            ),
            (
                "RevolHistory ",
                include_str!("templates/revol_history.cpp.in"),
            ),
            (
                "PipeHistory ",
                include_str!("templates/pipe_history.cpp.in"),
            ),
            (
                "HistoryMaker ",
                include_str!("templates/history_adapter.cpp.in"),
            ),
        ],
    );
}

/// Generate the contents of `facade/generated/kernel.cpp`.
pub fn emit_kernel(methods: &[&MethodSpec]) -> String {
    let mut buf = String::with_capacity(4096);

    // Header.
    let _ = writeln!(
        buf,
        "// AUTO-GENERATED by cargo xtask codegen -- DO NOT EDIT"
    );
    let _ = writeln!(buf);
    let _ = writeln!(buf, "#include \"occt_kernel.h\"");
    let _ = writeln!(buf);

    // OCCT includes.
    let includes = collect_includes(methods);
    for inc in &includes {
        let _ = writeln!(buf, "#include <{inc}>");
    }
    let _ = writeln!(buf);

    // Standard C++ includes.
    let _ = writeln!(buf, "#include <algorithm>");
    let _ = writeln!(buf, "#include <array>");
    let _ = writeln!(buf, "#include <cmath>");
    let _ = writeln!(buf, "#include <cstdio>");
    let _ = writeln!(buf, "#include <cstdlib>");
    let _ = writeln!(buf, "#include <fstream>");
    let _ = writeln!(buf, "#include <iomanip>");
    let _ = writeln!(buf, "#include <memory>");
    let _ = writeln!(buf, "#include <set>");
    let _ = writeln!(buf, "#include <unordered_set>");
    let _ = writeln!(buf, "#include <sstream>");
    let _ = writeln!(buf, "#include <stdexcept>");
    let _ = writeln!(buf, "#include <string>");
    let _ = writeln!(buf, "#include <vector>");
    let _ = writeln!(buf);

    // Emit helper functions needed by generated methods.
    emit_helper_functions(&mut buf, methods);

    // Methods grouped by category.
    let groups = group_by_category(methods);
    for (i, (category, specs)) in groups.iter().enumerate() {
        let _ = writeln!(buf, "// === {category} ===");
        let _ = writeln!(buf);

        for spec in specs {
            match spec.kind {
                MethodKind::SimpleShape => emit_simple_shape(&mut buf, spec),
                MethodKind::BooleanOp => emit_boolean_op(&mut buf, spec),
                MethodKind::FilletLike => emit_fillet_like(&mut buf, spec),
                MethodKind::SetupShape => emit_setup_shape(&mut buf, spec),
                MethodKind::CustomBody => emit_custom_body(&mut buf, spec),
                MethodKind::CustomBodyRaw => emit_custom_body_raw(&mut buf, spec),
                MethodKind::Skip => {}
            }
            let _ = writeln!(buf);
        }

        // Blank line between category groups, but not after the last one.
        if i + 1 < groups.len() {
            // Already have a trailing newline from the last method.
        }
    }

    // Trim trailing whitespace.
    buf.trim_end().to_owned() + "\n"
}

/// Generate the contents of `facade/generated/bindings.cpp` as a reference
/// file.
///
/// This is **not** compiled or linked. It shows the `.function()` lines that
/// belong in the hand-written `facade/src/bindings.cpp` inside the
/// `class_<OcctKernel>("OcctKernel")` block.
/// Close an Embind chain: replace the trailing `)\n` with `);\n`.
/// Bind a `ReturnType::Bytes` method through a lambda that copies the bytes
/// into a fresh `Uint8Array`. Binding the member directly would hand Embind a
/// `std::string`, which it decodes as UTF-8 and so mangles every byte >= 0x80.
fn emit_bytes_binding(buf: &mut String, spec: &MethodSpec) {
    let name = spec.name;
    let params = param_list(spec.params);
    let args = spec
        .params
        .iter()
        .map(|p| p.name())
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        buf,
        "        .function(\"{name}\", +[](OcctKernel& kernel, {params}) {{"
    );
    let _ = writeln!(
        buf,
        "            const std::string bytes = kernel.{name}({args});"
    );
    let _ = writeln!(
        buf,
        "            return val::global(\"Uint8Array\").new_(typed_memory_view("
    );
    let _ = writeln!(
        buf,
        "                bytes.size(), reinterpret_cast<const uint8_t*>(bytes.data())));"
    );
    let _ = writeln!(buf, "        }})");
}

fn close_embind_chain(buf: &mut String) {
    if buf.ends_with(")\n") {
        buf.truncate(buf.len() - 2);
        buf.push_str(");\n");
    }
}

/// Embind representation of a result data type.
enum DataKind {
    Class,
    Value,
}

enum MemberKind {
    Function,
    Property,
    Field,
}

struct DataRegistration {
    name: &'static str,
    kind: DataKind,
    members: &'static [(&'static str, MemberKind)],
}

const DATA_REGISTRATIONS: &[DataRegistration] = &[
    DataRegistration {
        name: "MeshData",
        kind: DataKind::Class,
        members: &[
            ("getPositionsPtr", MemberKind::Function),
            ("getNormalsPtr", MemberKind::Function),
            ("getUvsPtr", MemberKind::Function),
            ("getIndicesPtr", MemberKind::Function),
            ("positionCount", MemberKind::Property),
            ("uvCount", MemberKind::Property),
            ("normalCount", MemberKind::Property),
            ("indexCount", MemberKind::Property),
            ("getFaceGroupsPtr", MemberKind::Function),
            ("faceGroupCount", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "MeshBatchData",
        kind: DataKind::Class,
        members: &[
            ("getPositionsPtr", MemberKind::Function),
            ("getNormalsPtr", MemberKind::Function),
            ("getIndicesPtr", MemberKind::Function),
            ("getShapeOffsetsPtr", MemberKind::Function),
            ("positionCount", MemberKind::Property),
            ("normalCount", MemberKind::Property),
            ("indexCount", MemberKind::Property),
            ("shapeCount", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "BBoxData",
        kind: DataKind::Value,
        members: &[
            ("xmin", MemberKind::Field),
            ("ymin", MemberKind::Field),
            ("zmin", MemberKind::Field),
            ("xmax", MemberKind::Field),
            ("ymax", MemberKind::Field),
            ("zmax", MemberKind::Field),
        ],
    },
    DataRegistration {
        name: "EdgeData",
        kind: DataKind::Class,
        members: &[
            ("getPointsPtr", MemberKind::Function),
            ("getEdgeGroupsPtr", MemberKind::Function),
            ("pointCount", MemberKind::Property),
            ("edgeGroupCount", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "ProjectionData",
        kind: DataKind::Value,
        members: &[
            ("visibleOutline", MemberKind::Field),
            ("visibleSmooth", MemberKind::Field),
            ("visibleSharp", MemberKind::Field),
            ("hiddenOutline", MemberKind::Field),
            ("hiddenSmooth", MemberKind::Field),
            ("hiddenSharp", MemberKind::Field),
        ],
    },
    DataRegistration {
        name: "NurbsCurveData",
        kind: DataKind::Class,
        members: &[
            ("degree", MemberKind::Property),
            ("rational", MemberKind::Property),
            ("periodic", MemberKind::Property),
            ("knots", MemberKind::Property),
            ("multiplicities", MemberKind::Property),
            ("poles", MemberKind::Property),
            ("weights", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "NurbsSurfaceData",
        kind: DataKind::Class,
        members: &[
            ("degreeU", MemberKind::Property),
            ("degreeV", MemberKind::Property),
            ("nbPolesU", MemberKind::Property),
            ("nbPolesV", MemberKind::Property),
            ("periodicU", MemberKind::Property),
            ("periodicV", MemberKind::Property),
            ("knotsU", MemberKind::Property),
            ("knotsV", MemberKind::Property),
            ("multiplicitiesU", MemberKind::Property),
            ("multiplicitiesV", MemberKind::Property),
            ("poles", MemberKind::Property),
            ("weights", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "CurveData",
        kind: DataKind::Class,
        members: &[
            ("kind", MemberKind::Property),
            ("first", MemberKind::Property),
            ("last", MemberKind::Property),
            ("params", MemberKind::Property),
            ("hasBSpline", MemberKind::Property),
            ("bspline", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "SurfaceData",
        kind: DataKind::Class,
        members: &[
            ("kind", MemberKind::Property),
            ("isDirect", MemberKind::Property),
            ("params", MemberKind::Property),
            ("hasBSpline", MemberKind::Property),
            ("bspline", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "EvolutionData",
        kind: DataKind::Class,
        members: &[
            ("resultId", MemberKind::Property),
            ("modified", MemberKind::Property),
            ("generated", MemberKind::Property),
            ("deleted", MemberKind::Property),
            ("edgeModified", MemberKind::Property),
            ("edgeGenerated", MemberKind::Property),
            ("edgeDeleted", MemberKind::Property),
            ("firstShape", MemberKind::Property),
            ("lastShape", MemberKind::Property),
            ("spineGenerated", MemberKind::Property),
            ("inputShapes", MemberKind::Property),
            ("resultCounts", MemberKind::Property),
            ("modifiedIndexed", MemberKind::Property),
            ("generatedIndexed", MemberKind::Property),
            ("deletedIndexed", MemberKind::Property),
            ("edgeModifiedIndexed", MemberKind::Property),
            ("edgeGeneratedIndexed", MemberKind::Property),
            ("edgeDeletedIndexed", MemberKind::Property),
            ("firstShapeIndexed", MemberKind::Property),
            ("lastShapeIndexed", MemberKind::Property),
            ("spineGeneratedIndexed", MemberKind::Property),
        ],
    },
    DataRegistration {
        name: "XCAFLabelInfo",
        kind: DataKind::Value,
        members: &[
            ("labelId", MemberKind::Field),
            ("name", MemberKind::Field),
            ("hasColor", MemberKind::Field),
            ("r", MemberKind::Field),
            ("g", MemberKind::Field),
            ("b", MemberKind::Field),
            ("isAssembly", MemberKind::Field),
            ("isComponent", MemberKind::Field),
            ("shapeId", MemberKind::Field),
        ],
    },
];

/// Register all result structs using the same field/member rules.
fn emit_data_registrations(buf: &mut String) {
    for registration in DATA_REGISTRATIONS {
        let name = registration.name;
        let kind = match registration.kind {
            DataKind::Class => "class_",
            DataKind::Value => "value_object",
        };
        let _ = writeln!(buf, "    // {name}");
        let _ = writeln!(buf, "    {kind}<{name}>(\"{name}\")");
        for (field, member) in registration.members {
            let member = match member {
                MemberKind::Function => "function",
                MemberKind::Property => "property",
                MemberKind::Field => "field",
            };
            let _ = writeln!(buf, "        .{member}(\"{field}\", &{name}::{field})");
        }
        close_embind_chain(buf);
        buf.push('\n');
    }
}

/// Generate the contents of `facade/generated/bindings.cpp`.
///
/// This is a real, compilable file that registers all Embind bindings.
pub fn emit_bindings(methods: &[&MethodSpec]) -> String {
    let mut buf = String::with_capacity(4096);

    let _ = writeln!(
        buf,
        "// AUTO-GENERATED by cargo xtask codegen -- DO NOT EDIT"
    );
    let _ = writeln!(buf);
    let _ = writeln!(buf, "#include \"occt_kernel.h\"");
    let _ = writeln!(buf, "#include <cstdint>");
    let _ = writeln!(buf, "#include <emscripten/bind.h>");
    let _ = writeln!(buf);
    let _ = writeln!(buf, "using namespace emscripten;");
    let _ = writeln!(buf);
    let _ = writeln!(buf, "EMSCRIPTEN_BINDINGS(occt_wasm) {{");

    // Vector types. Each is given a dataPtr() returning the address of its
    // contiguous storage, so JS can read large results through one typed-array
    // view over the heap instead of N per-element get() boundary crossings.
    let _ = writeln!(buf, "    // Vector types");
    for (cpp_ty, js_name) in [
        ("uint32_t", "VectorUint32"),
        ("double", "VectorDouble"),
        ("int", "VectorInt"),
    ] {
        let _ = writeln!(buf, "    register_vector<{cpp_ty}>(\"{js_name}\")");
        let _ = writeln!(
            buf,
            "        .function(\"dataPtr\", +[](const std::vector<{cpp_ty}>& v) {{"
        );
        // unsigned int (not int): a heap address above 2 GB would become a
        // negative JS number, and slice(negativeStart) silently wraps instead
        // of throwing. Bit-identical on wasm32, but unambiguous.
        let _ = writeln!(
            buf,
            "            return static_cast<unsigned int>(reinterpret_cast<uintptr_t>(v.data()));"
        );
        let _ = writeln!(buf, "        }});");
    }
    let _ = writeln!(buf);

    emit_data_registrations(&mut buf);

    // OcctKernel method bindings — auto-generated from specs
    let _ = writeln!(buf, "    // OcctKernel");
    let _ = writeln!(buf, "    class_<OcctKernel>(\"OcctKernel\")");
    let _ = writeln!(buf, "        .constructor<>()");

    let groups = group_by_category(methods);
    for (category, specs) in &groups {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "        // {category}");
        for spec in specs {
            let name = spec.name;
            if spec.return_type == ReturnType::Bytes {
                emit_bytes_binding(&mut buf, spec);
            } else if matches!(
                spec.return_type,
                ReturnType::MeshData | ReturnType::MeshBatchData | ReturnType::EdgeData
            ) {
                let _ = writeln!(
                    buf,
                    "        .function(\"{name}\", &OcctKernel::{name}, return_value_policy::take_ownership())"
                );
            } else {
                let _ = writeln!(buf, "        .function(\"{name}\", &OcctKernel::{name})");
            }
        }
    }

    close_embind_chain(&mut buf);

    let _ = writeln!(buf, "}}");

    buf.trim_end().to_owned() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::types::{FacadeParam, MethodKind, MethodSpec, ReturnType};

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
        category: "Primitives",
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
        category: "Booleans",
    };

    static FILLET: MethodSpec = MethodSpec {
        name: "fillet",
        kind: MethodKind::FilletLike,
        params: &[
            FacadeParam::ShapeId("solidId"),
            FacadeParam::VectorShapeIds("edgeIds"),
            FacadeParam::Double("radius"),
        ],
        return_type: ReturnType::ShapeId,
        occt_class: "BRepFilletAPI_MakeFillet",
        ctor_args: "TopoDS::Solid(get(solidId))",
        setup_code: "",
        includes: &["TopoDS.hxx"],
        category: "Modeling",
    };

    #[test]
    fn kernel_simple_shape_matches_expected() {
        let methods: Vec<&MethodSpec> = vec![&MAKE_BOX];
        let output = emit_kernel(&methods);

        assert!(output.contains("uint32_t OcctKernel::makeBox(double dx, double dy, double dz)"));
        assert!(output.contains("BRepPrimAPI_MakeBox maker(dx, dy, dz)"));
        assert!(output.contains("maker.Build()"));
        assert!(output.contains("store(maker.Shape())"));
        assert!(output.contains("#include <BRepPrimAPI_MakeBox.hxx>"));
        assert!(output.contains("// === Primitives ==="));
    }

    #[test]
    fn kernel_boolean_op_uses_ctor_args() {
        let methods: Vec<&MethodSpec> = vec![&FUSE];
        let output = emit_kernel(&methods);

        assert!(output.contains("BRepAlgoAPI_Fuse op;"));
        assert!(output.contains("buildBoolean(op, get(a), get(b));"));
        assert!(output.contains("recoverLostFuse(op, {get(a), get(b)});"));
        assert_eq!(
            output.matches("op.Build();").count(),
            2,
            "only the helper builds, and the retry of a lost fuse"
        );
        assert!(output.contains("static void buildBoolean("));
        assert!(output.contains("static bool isFiniteShape("));
        assert!(output.contains("op.SetUseOBB(allFinite(args) && allFinite(tools));"));
        assert!(output.contains("op.HasErrors()"));
        assert!(output.contains("boolean operation failed"));
    }

    #[test]
    fn kernel_fillet_like_iterates_edges() {
        let methods: Vec<&MethodSpec> = vec![&FILLET];
        let output = emit_kernel(&methods);

        assert!(output.contains("for (uint32_t eid : edgeIds)"));
        assert!(output.contains("edges.push_back(TopoDS::Edge(get(eid)));"));
        assert!(output.contains("maker.Add(radius, edges.back())"));
        assert!(output.contains("#include <TopoDS.hxx>"));
    }

    #[test]
    fn kernel_fillet_like_names_faulty_edges() {
        let methods: Vec<&MethodSpec> = vec![&FILLET];
        let output = emit_kernel(&methods);

        assert!(output.contains(
            "throw std::runtime_error(filletFailure(maker, edgeIds, edges, \"fillet\"));"
        ));
        assert!(output.contains("static std::string filletFailure("));
        assert!(output.contains("static std::string chamferFailure("));
        assert!(output.contains("[faulty edges: "));
        assert!(output.contains("#include <ChFi3d_ChBuilder.hxx>"));
    }

    #[test]
    fn bindings_emits_compilable_embind() {
        let methods: Vec<&MethodSpec> = vec![&MAKE_BOX, &FUSE, &FILLET];
        let output = emit_bindings(&methods);

        assert!(output.contains("// AUTO-GENERATED by cargo xtask codegen"));
        assert!(output.contains("#include <emscripten/bind.h>"));
        assert!(output.contains("EMSCRIPTEN_BINDINGS(occt_wasm)"));
        assert!(output.contains(".function(\"makeBox\", &OcctKernel::makeBox)"));
        assert!(output.contains(".function(\"fuse\", &OcctKernel::fuse)"));
        assert!(output.contains(".function(\"fillet\", &OcctKernel::fillet)"));
        assert!(output.contains("class_<OcctKernel>(\"OcctKernel\")"));
    }

    #[test]
    fn bytes_return_binds_through_a_uint8array_copy() {
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
        let bindings = emit_bindings(&[&EXPORT_BYTES]);
        assert!(!bindings.contains("&OcctKernel::exportStlBinary"));
        assert!(bindings.contains(
            ".function(\"exportStlBinary\", +[](OcctKernel& kernel, uint32_t id, double linearDeflection) {"
        ));
        assert!(bindings.contains("kernel.exportStlBinary(id, linearDeflection)"));
        assert!(bindings.contains("val::global(\"Uint8Array\").new_(typed_memory_view("));

        let kernel = emit_kernel(&[&EXPORT_BYTES]);
        assert!(kernel.contains(
            "std::string OcctKernel::exportStlBinary(uint32_t id, double linearDeflection)"
        ));
    }

    #[test]
    fn skip_methods_are_excluded() {
        static SKIPPED: MethodSpec = MethodSpec {
            name: "makeEllipsoid",
            kind: MethodKind::Skip,
            params: &[],
            return_type: ReturnType::ShapeId,
            occt_class: "",
            ctor_args: "",
            setup_code: "",
            includes: &[],
            category: "Primitives",
        };
        let methods: Vec<&MethodSpec> = vec![&SKIPPED, &MAKE_BOX];
        let kernel = emit_kernel(&methods);
        let bindings = emit_bindings(&methods);

        assert!(!kernel.contains("makeEllipsoid"));
        assert!(!bindings.contains("makeEllipsoid"));
    }

    #[test]
    fn includes_are_deduplicated_and_sorted() -> anyhow::Result<()> {
        use anyhow::Context as _;
        let methods: Vec<&MethodSpec> = vec![&MAKE_BOX, &FUSE, &FILLET];
        let output = emit_kernel(&methods);

        // All includes should appear exactly once.
        let count = output.matches("#include <TopoDS.hxx>").count();
        assert_eq!(count, 1);

        // Sorted: BRepAlgoAPI_Fuse before BRepPrimAPI_MakeBox.
        let fuse_pos = output
            .find("#include <BRepAlgoAPI_Fuse.hxx>")
            .context("fuse include")?;
        let box_pos = output
            .find("#include <BRepPrimAPI_MakeBox.hxx>")
            .context("box include")?;
        assert!(fuse_pos < box_pos);
        Ok(())
    }

    #[test]
    fn custom_body_bool_return() {
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
        let output = emit_kernel(&[&IS_VALID]);
        assert!(output.contains("bool OcctKernel::isValid(uint32_t id)"));
        assert!(output.contains("BRepCheck_Analyzer checker(get(id))"));
    }

    #[test]
    fn custom_body_void_return() {
        static BUILD_CURVES: MethodSpec = MethodSpec {
            name: "buildCurves3d",
            kind: MethodKind::CustomBody,
            params: &[FacadeParam::ShapeId("wireId")],
            return_type: ReturnType::Void,
            occt_class: "",
            ctor_args: "",
            setup_code: "BRepLib::BuildCurves3d(get(wireId));",
            includes: &["BRepLib.hxx"],
            category: "healing",
        };
        let output = emit_kernel(&[&BUILD_CURVES]);
        assert!(output.contains("void OcctKernel::buildCurves3d(uint32_t wireId)"));
        assert!(output.contains("BRepLib::BuildCurves3d(get(wireId))"));
    }
}
