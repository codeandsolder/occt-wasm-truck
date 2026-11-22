//! Integration tests for the occt-wasm Rust crate.
//!
//! These tests exercise the embedded WASI kernel. Debug builds explicitly
//! mark them ignored because debug wasmtime compilation is prohibitively slow.
//! Release tests always require a working embedded kernel.
//!
//! To run: `cargo test --release -p occt-wasm --test integration`

use std::sync::OnceLock;

use occt_wasm::wasmtime::{Engine, Module, StoreLimitsBuilder};
use occt_wasm::{
    CurveGeometry, DocumentHandle, OcctError, OcctKernel, ShapeHandle, SurfaceData, Vec3,
};

/// Compile once per integration-test process; every kernel has its own store.
fn shared_module() -> &'static Module {
    static MODULE: OnceLock<Module> = OnceLock::new();
    MODULE.get_or_init(|| {
        let engine = Engine::new(&OcctKernel::engine_config()).expect("create WASM engine");
        OcctKernel::compile(&engine).expect("compile embedded OCCT module")
    })
}

fn kernel() -> OcctKernel {
    OcctKernel::from_module(shared_module()).expect("instantiate embedded OCCT module")
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn kernel_init_and_drop() {
    let _kernel = kernel();
    // Kernel initializes and drops cleanly
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn new_compiles_its_own_module() {
    let mut kernel = OcctKernel::new().unwrap();
    let shape = kernel.make_box(1.0, 2.0, 3.0).unwrap();
    assert!((kernel.get_volume(shape).unwrap() - 6.0).abs() < 1e-9);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn one_module_serves_independent_kernels() {
    let module = shared_module();
    let mut a = OcctKernel::from_module(module).unwrap();
    let mut b = OcctKernel::from_module(module).unwrap();
    let in_a = a.make_box(1.0, 2.0, 3.0).unwrap();
    let in_b = b.make_box(2.0, 2.0, 2.0).unwrap();
    // Each kernel has its own memory, so the same arena slot holds different shapes.
    assert_eq!(in_a, in_b);
    a.release(in_a).unwrap();
    assert!((b.get_volume(in_b).unwrap() - 8.0).abs() < 1e-9);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
#[expect(
    unsafe_code,
    reason = "test round-trips a module serialized in this process using the same engine configuration"
)]
fn serialized_module_instantiates() {
    let module = shared_module();
    let bytes = module.serialize().unwrap();
    let engine = Engine::new(&OcctKernel::engine_config()).unwrap();
    // SAFETY: the bytes come from `Module::serialize` in this process, on an
    // engine with the same configuration.
    let restored = unsafe { Module::deserialize(&engine, &bytes) }.unwrap();
    let mut kernel = OcctKernel::from_module(&restored).unwrap();
    let shape = kernel.make_box(1.0, 2.0, 3.0).unwrap();
    assert!((kernel.get_volume(shape).unwrap() - 6.0).abs() < 1e-9);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn memory_limit_below_initial_memory_fails_instantiation() {
    let module = shared_module();
    let limits = StoreLimitsBuilder::new().memory_size(64 << 20).build();
    match OcctKernel::from_module_with_limiter(module, limits) {
        Err(OcctError::MemoryLimit { current, requested }) => {
            assert_eq!(current, 0);
            assert_eq!(requested, 128 << 20);
        }
        Err(e) => panic!("expected a memory limit error, got {e:?}"),
        Ok(_) => panic!("a 64 MiB cap should not fit the module's 128 MiB of initial memory"),
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn memory_limit_stops_the_call_that_exceeds_it() {
    let module = shared_module();
    let cap = 160 << 20;
    let limits = StoreLimitsBuilder::new().memory_size(cap).build();
    let mut kernel = OcctKernel::from_module_with_limiter(module, limits).unwrap();
    let sphere = kernel.make_sphere(10.0).unwrap();
    match kernel.tessellate(sphere, 0.0005, 0.01) {
        Err(OcctError::MemoryLimit { current, requested }) => {
            assert!(
                current <= cap && requested > cap,
                "{current} -> {requested}"
            );
        }
        Err(e) => panic!("expected a memory limit error, got {e:?}"),
        Ok(mesh) => panic!("{} triangles fit under the cap", mesh.indices.len() / 3),
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn make_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    assert!(shape.id() > 0);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn box_volume() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let vol = kernel.get_volume(shape).unwrap();
    assert!((vol - 6000.0).abs() < 1.0, "expected ~6000, got {vol}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn fuse_two_boxes() {
    let mut kernel = kernel();
    let a = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let b = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let result = kernel.fuse(a, b).unwrap();
    let vol = kernel.get_volume(result).unwrap();
    // Fuse of identical boxes at origin = same volume (overlapping)
    assert!((vol - 1000.0).abs() < 1.0, "expected ~1000, got {vol}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn cut_operation() {
    let mut kernel = kernel();
    let box_shape = kernel.make_box(20.0, 20.0, 20.0).unwrap();
    let sphere = kernel.make_sphere(5.0).unwrap();
    let result = kernel.cut(box_shape, sphere).unwrap();
    let vol = kernel.get_volume(result).unwrap();
    // Box volume - sphere volume (partially inside)
    assert!(vol > 0.0 && vol < 8000.0, "unexpected volume: {vol}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn bounding_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let bbox = kernel.get_bounding_box(shape, true).unwrap();
    assert!((bbox.min.x).abs() < 0.01);
    assert!((bbox.min.y).abs() < 0.01);
    assert!((bbox.min.z).abs() < 0.01);
    assert!((bbox.max.x - 10.0).abs() < 0.01);
    assert!((bbox.max.y - 20.0).abs() < 0.01);
    assert!((bbox.max.z - 30.0).abs() < 0.01);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn bounding_box_loose_contains_precise() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let bbox = kernel.get_bounding_box_loose(shape, false).unwrap();
    assert!((bbox.min.x).abs() < 0.01);
    assert!((bbox.max.x - 10.0).abs() < 0.01);
    assert!((bbox.max.z - 30.0).abs() < 0.01);

    let cyl = kernel.make_cylinder(5.0, 10.0).unwrap();
    let precise = kernel.get_bounding_box(cyl, false).unwrap();
    let loose = kernel.get_bounding_box_loose(cyl, false).unwrap();
    assert!(loose.min.x <= precise.min.x && loose.max.x >= precise.max.x);
    assert!(loose.min.y <= precise.min.y && loose.max.y >= precise.max.y);
    assert!(loose.min.z <= precise.min.z && loose.max.z >= precise.max.z);
    assert!((loose.max.z - 10.0).abs() < 0.01);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn tessellate_box() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let mesh = kernel.tessellate(shape, 0.1, 0.5).unwrap();
    // A box has at least 8 vertices and 12 triangles
    assert!(
        mesh.positions.len() >= 24,
        "expected at least 24 position floats, got {}",
        mesh.positions.len()
    );
    assert!(
        mesh.indices.len() >= 36,
        "expected at least 36 index values, got {}",
        mesh.indices.len()
    );
    assert_eq!(mesh.positions.len(), mesh.normals.len());
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn step_roundtrip() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let step_data = kernel.export_step(shape).unwrap();
    assert!(step_data.contains("STEP"), "expected STEP format data");

    let imported = kernel.import_step(&step_data).unwrap();
    let vol = kernel.get_volume(imported).unwrap();
    assert!(
        (vol - 6000.0).abs() < 1.0,
        "roundtrip volume mismatch: {vol}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn brep_binary_roundtrip() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let bytes = kernel.export_brep_binary(shape).unwrap();
    let restored = kernel.import_brep_binary(&bytes).unwrap();
    assert!((kernel.get_volume(restored).unwrap() - 6000.0).abs() < 1e-6);
    assert!(kernel.import_brep_binary(b"not a brep").is_err());
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn stl_binary_roundtrip() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 20.0, 30.0).unwrap();
    let bytes = kernel.export_stl_binary(shape, 0.5).unwrap();
    let triangles = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    assert_eq!(triangles, 12);
    assert_eq!(bytes.len(), 84 + triangles * 50);
    for facet in bytes[84..].chunks_exact(50) {
        let mut normal: Vec<f32> = facet[..12]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()).abs())
            .collect();
        normal.sort_by(f32::total_cmp);
        assert_eq!(normal, [0.0, 0.0, 1.0], "facet normal bytes must survive");
    }

    let imported = kernel.import_stl_binary(&bytes).unwrap();
    let bbox = kernel.get_bounding_box(imported, false).unwrap();
    assert!((bbox.max.x - 10.0).abs() < 1e-3, "bbox: {bbox:?}");
    assert!((bbox.max.y - 20.0).abs() < 1e-3, "bbox: {bbox:?}");
    assert!((bbox.max.z - 30.0).abs() < 1e-3, "bbox: {bbox:?}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn make_cylinder_and_query() {
    let mut kernel = kernel();
    let cyl = kernel.make_cylinder(5.0, 10.0).unwrap();
    let vol = kernel.get_volume(cyl).unwrap();
    let expected = std::f64::consts::PI * 25.0 * 10.0;
    assert!(
        (vol - expected).abs() < 1.0,
        "expected ~{expected:.0}, got {vol}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn make_sphere_surface_area() {
    let mut kernel = kernel();
    let sphere = kernel.make_sphere(5.0).unwrap();
    let area = kernel.get_surface_area(sphere).unwrap();
    let expected = 4.0 * std::f64::consts::PI * 25.0;
    assert!(
        (area - expected).abs() < 1.0,
        "expected ~{expected:.0}, got {area}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn release_shape() {
    let mut kernel = kernel();
    let shape = kernel.make_box(1.0, 1.0, 1.0).unwrap();
    kernel.release(shape).unwrap();
    // After release, the handle is invalid
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn get_shape_type() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let shape_type = kernel.get_shape_type(shape).unwrap();
    assert_eq!(shape_type, "solid", "box should be a solid");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn get_sub_shapes() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let faces = kernel.get_sub_shapes(shape, "face").unwrap();
    assert_eq!(faces.len(), 6, "box should have 6 faces");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn sub_shape_count_and_hashes() {
    let mut kernel = kernel();
    let shape = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    assert_eq!(kernel.sub_shape_count(shape, "face").unwrap(), 6);
    let hashes = kernel.sub_shape_hashes(shape, "face", 1_000_000).unwrap();
    assert_eq!(hashes.len(), 6, "one hash per face, no handles allocated");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn checkpoint_release_since() {
    let mut kernel = kernel();
    let _keep = kernel.make_box(1.0, 1.0, 1.0).unwrap();
    let mark = kernel.checkpoint().unwrap();
    let _a = kernel.make_box(2.0, 2.0, 2.0).unwrap();
    let _b = kernel.make_box(3.0, 3.0, 3.0).unwrap();
    assert_eq!(kernel.get_shape_count().unwrap(), 3);
    kernel.release_since(mark).unwrap();
    // Only the pre-checkpoint handle survives.
    assert_eq!(kernel.get_shape_count().unwrap(), 1);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn extrude_rectangle() {
    let mut kernel = kernel();
    let rect = kernel.make_rectangle(10.0, 20.0).unwrap();
    let solid = kernel.extrude(rect, 0.0, 0.0, 5.0).unwrap();
    let vol = kernel.get_volume(solid).unwrap();
    assert!(
        (vol - 1000.0).abs() < 1.0,
        "expected 10*20*5=1000, got {vol}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn sweep_oriented_auxiliary_keeps_planes_exact() {
    let mut kernel = kernel();
    let corners = [
        (-2.0, -2.0),
        (2.0, -2.0),
        (2.0, 2.0),
        (-2.0, 2.0),
        (-2.0, -2.0),
    ];
    let profile_edges: Vec<_> = corners
        .windows(2)
        .map(|w| {
            kernel
                .make_line_edge(w[0].0, w[0].1, 0.0, w[1].0, w[1].1, 0.0)
                .unwrap()
        })
        .collect();
    let profile = kernel.make_wire(&profile_edges).unwrap();
    let spine_edge = kernel
        .make_line_edge(0.0, 0.0, 0.0, 0.0, 0.0, 20.0)
        .unwrap();
    let spine = kernel.make_wire(&[spine_edge]).unwrap();
    let guide_edge = kernel
        .make_line_edge(5.0, 0.0, 0.0, 5.0, 0.0, 20.0)
        .unwrap();
    let guide = kernel.make_wire(&[guide_edge]).unwrap();

    let solid = kernel
        .sweep_oriented(occt_wasm::SweepOrientedRequest {
            profile_id: profile,
            spine_id: spine,
            mode: 3,
            up_x: 0.0,
            up_y: 0.0,
            up_z: 1.0,
            aux_spine_id: guide,
            curvilinear_equivalence: false,
            contact_mode: 0,
            tol3d: 0.0,
            bound_tol: 0.0,
            tol_angular: 0.0,
        })
        .unwrap();
    let vol = kernel.get_volume(solid).unwrap().abs();
    assert!((vol - 320.0).abs() < 1e-6, "expected 4*4*20=320, got {vol}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn helix_handedness_mirrors_across_the_axis_plane() {
    let mut kernel = kernel();
    // One turn: pitch 5, height 5, radius 3 about +Z through the origin.
    let mut quarter_turn = |left_handed: bool| {
        let wire = kernel
            .make_helix_wire_handed(occt_wasm::MakeHelixWireHandedRequest {
                px: 0.0,
                py: 0.0,
                pz: 0.0,
                dx: 0.0,
                dy: 0.0,
                dz: 1.0,
                pitch: 5.0,
                height: 5.0,
                radius: 3.0,
                left_handed,
            })
            .unwrap();
        let range = kernel.curve_parameters(wire).unwrap();
        let param = range[0] + (range[1] - range[0]) / 4.0;
        kernel.curve_point_at_param(wire, param).unwrap()
    };
    let right = quarter_turn(false);
    let left = quarter_turn(true);

    assert!(
        (right[1] - 3.0).abs() < 1e-6,
        "right-handed quarter turn should reach +Y, got {right:?}"
    );
    assert!(
        (left[1] + 3.0).abs() < 1e-6,
        "left-handed quarter turn should reach -Y, got {left:?}"
    );
    assert!((left[0] - right[0]).abs() < 1e-6);
    assert!((left[2] - right[2]).abs() < 1e-6);
}

/// An XCAF document with a named, coloured housing and one gear component.
fn housing_with_gear(kernel: &mut OcctKernel) -> (DocumentHandle, i32, i32) {
    let doc = kernel.xcaf_new_document().unwrap();
    let housing = kernel.make_box(20.0, 20.0, 20.0).unwrap();
    let gear = kernel.make_cylinder(5.0, 10.0).unwrap();
    let root = kernel.xcaf_add_shape(doc, housing).unwrap();
    kernel.xcaf_set_name(doc, root, "housing").unwrap();
    kernel.xcaf_set_color(doc, root, 0.8, 0.2, 0.1).unwrap();
    let comp = kernel
        .xcaf_add_component(occt_wasm::XcafAddComponentRequest {
            doc_id: doc,
            parent_label_id: root,
            shape_id: gear,
            tx: 10.0,
            ty: 0.0,
            tz: 5.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
        })
        .unwrap();
    kernel.xcaf_set_name(doc, comp, "gear-1").unwrap();
    (doc, root, comp)
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn xcaf_document_assembly() {
    let mut kernel = kernel();
    let (doc, _, comp) = housing_with_gear(&mut kernel);

    let info = kernel.xcaf_get_label_info(doc, comp).unwrap();
    assert!(info.is_component);
    assert_eq!(info.name, "gear-1");
    let proto = kernel.xcaf_get_referred_label(doc, comp).unwrap();
    assert!(proto > 0);
    assert_eq!(kernel.xcaf_get_referred_label(doc, proto).unwrap(), 0);
    let location = kernel.xcaf_get_label_location(doc, comp).unwrap();
    assert_eq!(location.len(), 12);
    assert!((location[3] - 10.0).abs() < 1e-9);
    assert!((location[11] - 5.0).abs() < 1e-9);
    kernel.xcaf_close(doc).unwrap();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn xcaf_export_glb() {
    let mut kernel = kernel();
    let (doc, _, _) = housing_with_gear(&mut kernel);
    let glb = kernel.xcaf_export_glb(doc, 0.5, 0.5).unwrap();
    assert_eq!(&glb[..4], b"glTF");
    assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
    let length = u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize;
    assert_eq!(length, glb.len());
    kernel.xcaf_close(doc).unwrap();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn xcaf_step_roundtrip() {
    let mut kernel = kernel();
    let (doc, _, _) = housing_with_gear(&mut kernel);
    let step = kernel.xcaf_export_step(doc).unwrap();
    assert!(step.contains("STEP"));
    kernel.xcaf_close(doc).unwrap();

    let imported = kernel.xcaf_import_step(&step).unwrap();
    let roots = kernel.xcaf_get_root_labels(imported).unwrap();
    assert_eq!(roots.len(), 1);
    let root_info = kernel.xcaf_get_label_info(imported, roots[0]).unwrap();
    assert_eq!(root_info.name, "housing");
    assert!(root_info.shape_id > 0);
    kernel.xcaf_close(imported).unwrap();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn enumerated_sub_shapes_feed_back_into_modelling() {
    let mut kernel = kernel();
    // get_sub_shapes hands back the same handle type the modelling calls take,
    // so an enumerated edge can be filleted without leaving the type system.
    let box_id = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let edges = kernel.get_sub_shapes(box_id, "edge").unwrap();
    assert_eq!(edges.len(), 12);

    let rounded = kernel.fillet(box_id, &edges[..1], 1.0).unwrap();
    let volume = kernel.get_volume(rounded).unwrap();
    assert!(volume > 990.0 && volume < 1000.0, "volume {volume}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn unbuildable_fillet_reports_an_error_instead_of_trapping() {
    let mut kernel = kernel();
    // A radius equal to the wall thickness drives OCCT's blend solver into an
    // adaptor holding no curve. Through wasmtime that dereference used to be a
    // trap, which no caller can handle (#348).
    let solid = kernel.make_box(40.0, 30.0, 20.0).unwrap();
    let top = kernel
        .get_sub_shapes(solid, "face")
        .unwrap()
        .into_iter()
        .find(|f| kernel.get_bounding_box(*f, false).unwrap().min.z > 19.999)
        .expect("top face");
    let shelled = kernel.shell(solid, &[top], 2.0, 1e-3).unwrap();

    let edges = kernel.get_sub_shapes(shelled, "edge").unwrap();
    let rim = edges
        .iter()
        .copied()
        .find(|e| {
            let b = kernel.get_bounding_box(*e, false).unwrap();
            b.min.z > 19.999 && b.min.x < 0.001 && b.max.x < 0.001
        })
        .expect("rim edge");

    let failure = kernel.fillet(shelled, &[rim], 2.0).unwrap_err();
    assert!(
        format!("{failure}").contains("fillet"),
        "expected a fillet error, got: {failure}"
    );

    // The store survives the failure, so the next build still works.
    let corner = edges
        .iter()
        .copied()
        .find(|e| {
            let b = kernel.get_bounding_box(*e, false).unwrap();
            b.min.x > 39.999 && b.min.y > 29.999
        })
        .expect("corner edge");
    let good = kernel.fillet(shelled, &[corner], 1.0).unwrap();
    let volume = kernel.get_volume(good).unwrap();
    assert!((7140.0..7152.0).contains(&volume), "volume {volume}");
}

/// L-shaped prism, 20 x 20 x 10, with one concave vertical edge at (10, 10).
/// Offsetting inward pulls that edge's two faces apart, so the join type
/// decides whether the gap is bridged by a cylinder (Arc) or planes (Intersection).
fn l_prism(kernel: &mut OcctKernel) -> ShapeHandle {
    let a = kernel.make_box(20.0, 10.0, 10.0).unwrap();
    let b = kernel.make_box(10.0, 20.0, 10.0).unwrap();
    let fused = kernel.fuse(a, b).unwrap();
    kernel.unify_same_domain(fused).unwrap()
}

fn all_planar(kernel: &mut OcctKernel, shape: ShapeHandle) -> bool {
    kernel
        .get_sub_shapes(shape, "face")
        .unwrap()
        .into_iter()
        .all(|f| kernel.surface_type(f).unwrap() == "plane")
}

fn top_face(kernel: &mut OcctKernel, shape: ShapeHandle) -> ShapeHandle {
    kernel
        .get_sub_shapes(shape, "face")
        .unwrap()
        .into_iter()
        .find(|f| kernel.get_bounding_box(*f, false).unwrap().min.z > 9.999)
        .expect("top face")
}

// Raw join codes: 0 = Arc, 1 = Intersection, 2 = Tangent (rejected in 3D).
const ARC: i32 = 0;
const INTERSECTION: i32 = 1;
const TANGENT: i32 = 2;
const BOUND: i32 = 1_000_000;

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn join_type_keeps_the_concave_edge_sharp() {
    let mut kernel = kernel();
    // Intersection keeps a 2 x 2 square at the concave edge where Arc keeps a
    // quarter circle of radius 2.
    let corner = 4.0 - std::f64::consts::PI;

    let solid = l_prism(&mut kernel);
    let arc = kernel.offset_with_join(solid, -2.0, 1e-6, ARC).unwrap();
    let sharp = kernel
        .offset_with_join(solid, -2.0, 1e-6, INTERSECTION)
        .unwrap();
    assert!(!all_planar(&mut kernel, arc));
    assert!(all_planar(&mut kernel, sharp));
    let lost = kernel.get_volume(arc).unwrap() - kernel.get_volume(sharp).unwrap();
    assert!(
        (lost - corner * 6.0).abs() < 1e-2,
        "offset volume difference {lost}"
    );

    let top = top_face(&mut kernel, solid);
    let arc = kernel
        .shell_with_join(solid, &[top], 2.0, 1e-6, ARC)
        .unwrap();
    let sharp = kernel
        .shell_with_join(solid, &[top], 2.0, 1e-6, INTERSECTION)
        .unwrap();
    assert!(kernel.is_valid(sharp).unwrap());
    assert!(all_planar(&mut kernel, sharp));
    let extra = kernel.get_volume(sharp).unwrap() - kernel.get_volume(arc).unwrap();
    assert!(
        (extra - corner * 8.0).abs() < 1e-2,
        "shell volume difference {extra}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn history_join_variants_return_a_usable_result() {
    let mut kernel = kernel();
    let solid = l_prism(&mut kernel);
    let hashes: Vec<i32> = kernel
        .get_sub_shapes(solid, "face")
        .unwrap()
        .into_iter()
        .map(|f| kernel.hash_code(f, BOUND).unwrap())
        .collect();

    let evo = kernel
        .offset_with_history_and_join(solid, -2.0, 1e-6, &hashes, BOUND, INTERSECTION)
        .unwrap();
    assert!(all_planar(&mut kernel, evo.result()));
    assert!(!evo.generated.is_empty());

    let top = top_face(&mut kernel, solid);
    let evo = kernel
        .shell_with_history_and_join(occt_wasm::ShellWithHistoryAndJoinRequest {
            solid_id: solid,
            face_ids: &[top],
            thickness: 2.0,
            tolerance: 1e-6,
            input_face_hashes: &hashes,
            hash_upper_bound: BOUND,
            join_type: INTERSECTION,
        })
        .unwrap();
    assert!(all_planar(&mut kernel, evo.result()));
    assert!(!evo.modified.is_empty());
    let plain = kernel
        .shell_with_join(solid, &[top], 2.0, 1e-6, INTERSECTION)
        .unwrap();
    let with_history = kernel.get_volume(evo.result()).unwrap();
    assert!((with_history - kernel.get_volume(plain).unwrap()).abs() < 1e-6);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn join_type_rejects_tangent() {
    let mut kernel = kernel();
    let solid = l_prism(&mut kernel);
    let err = kernel
        .offset_with_join(solid, -2.0, 1e-6, TANGENT)
        .unwrap_err();
    assert!(format!("{err}").contains("joinType"), "got: {err}");
    let err = kernel
        .offset_with_history_and_join(solid, -2.0, 1e-6, &[], 1, TANGENT)
        .unwrap_err();
    assert!(format!("{err}").contains("joinType"), "got: {err}");

    // The store survives the rejection.
    assert!(kernel.offset_with_join(solid, -2.0, 1e-6, ARC).is_ok());
}

/// The four edges of a 10 mm cube that run along Z.
fn vertical_edges(kernel: &mut OcctKernel, solid: ShapeHandle) -> Vec<ShapeHandle> {
    let edges = kernel.get_sub_shapes(solid, "edge").unwrap();
    edges
        .into_iter()
        .filter(|&edge| {
            let bb = kernel.get_bounding_box(edge, false).unwrap();
            (bb.max.z - bb.min.z) > 1.0
        })
        .collect()
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn fillet_radii_gives_each_edge_its_own_radius() {
    let mut kernel = kernel();
    let cube = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let edges = vertical_edges(&mut kernel, cube);
    assert_eq!(edges.len(), 4);
    let loss = |r: f64| (1.0 - std::f64::consts::FRAC_PI_4) * r * r * 10.0;

    let rounded = kernel
        .fillet_radii(cube, &edges, &[1.0, 1.0, 3.0, 3.0], &[1.0, 1.0, 3.0, 3.0])
        .unwrap();
    let volume = kernel.get_volume(rounded).unwrap();
    let removed: f64 = [1.0, 1.0, 3.0, 3.0].into_iter().map(loss).sum();
    let expected = 1000.0 - removed;
    assert!(
        (volume - expected).abs() < 1e-6,
        "volume {volume}, expected {expected}"
    );

    let history = kernel
        .fillet_radii_with_history(cube, &edges, &[1.0; 4], &[3.0; 4], &[], 1 << 20)
        .unwrap();
    let radius_one: f64 = [1.0; 4].into_iter().map(loss).sum();
    assert!(kernel.get_volume(history.result()).unwrap() < 1000.0 - radius_one);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn chamfer_distances_and_angles_per_edge() {
    let mut kernel = kernel();
    let cube = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let edges = vertical_edges(&mut kernel, cube);

    let symmetric = kernel
        .chamfer_distances(
            cube,
            &edges,
            &[1.0, 1.0, 2.0, 2.0],
            &[1.0, 1.0, 2.0, 2.0],
            &[],
        )
        .unwrap();
    assert!((kernel.get_volume(symmetric).unwrap() - 950.0).abs() < 1e-6);

    let asymmetric = kernel
        .chamfer_distances(cube, &edges, &[1.0; 4], &[3.0; 4], &[])
        .unwrap();
    assert!((kernel.get_volume(asymmetric).unwrap() - 940.0).abs() < 1e-6);

    let angled = kernel
        .chamfer_dist_angles(cube, &edges, &[1.0; 4], &[45.0; 4], &[])
        .unwrap();
    assert!((kernel.get_volume(angled).unwrap() - 980.0).abs() < 1e-6);

    let hashes: Vec<i32> = kernel
        .get_sub_shapes(cube, "face")
        .unwrap()
        .into_iter()
        .map(|f| kernel.hash_code(f, BOUND).unwrap())
        .collect();
    let angled_history = kernel
        .chamfer_dist_angles_with_history(occt_wasm::ChamferDistAnglesWithHistoryRequest {
            solid_id: cube,
            edge_ids: &edges,
            distances: &[1.0; 4],
            angles_deg: &[45.0; 4],
            reference_face_ids: &[],
            input_face_hashes: &hashes,
            hash_upper_bound: BOUND,
        })
        .unwrap();
    assert!((kernel.get_volume(angled_history.result()).unwrap() - 980.0).abs() < 1e-6);
    assert!(!angled_history.modified.is_empty());

    let err = kernel
        .chamfer_distances(cube, &edges, &[1.0; 4], &[1.0; 3], &[])
        .unwrap_err();
    assert!(err.to_string().contains("one entry per edge"), "{err}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn draft_faces_pivot_about_the_neutral_plane() {
    let mut kernel = kernel();
    let cube = kernel
        .make_box_from_corners(0.0, 0.0, 100.0, 10.0, 10.0, 110.0)
        .unwrap();
    let faces = kernel.get_sub_shapes(cube, "face").unwrap();
    let mut sides = Vec::new();
    for face in faces {
        let center = kernel.get_surface_center_of_mass(face).unwrap();
        if (center[2] - 105.0).abs() < 1e-9 {
            sides.push(face);
        }
    }
    assert_eq!(sides.len(), 4);
    let angle = 5f64.to_radians();
    let drafted = kernel
        .draft_faces(occt_wasm::DraftFacesRequest {
            shape_id: cube,
            face_ids: &sides[..1],
            angles_rad: &[angle],
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
            px: 0.0,
            py: 0.0,
            pz: 100.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
        })
        .unwrap();
    let wedge = 500.0 * angle.tan();
    let expected = 1000.0 - wedge;
    let volume = kernel.get_volume(drafted).unwrap();
    assert!(
        (volume - expected).abs() < 1e-6,
        "volume {volume}, expected {expected}"
    );

    let all = kernel
        .draft_faces(occt_wasm::DraftFacesRequest {
            shape_id: cube,
            face_ids: &sides,
            angles_rad: &[angle; 4],
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
            px: 0.0,
            py: 0.0,
            pz: 100.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
        })
        .unwrap();
    assert!(kernel.get_volume(all).unwrap() < volume);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn step_export_writes_the_requested_unit() {
    let mut kernel = kernel();
    let cube = kernel.make_box(25.4, 25.4, 25.4).unwrap();
    let step = kernel.export_step_with_unit(cube, "inch").unwrap();
    assert!(step.contains("CONVERSION_BASED_UNIT('INCH'"));
    let back = kernel.import_step(&step).unwrap();
    let volume = kernel.get_volume(back).unwrap();
    assert!((volume - 25.4f64.powi(3)).abs() < 1e-6, "volume {volume}");
    assert!(kernel.export_step_with_unit(cube, "furlong").is_err());

    let doc = kernel.xcaf_new_document().unwrap();
    let label = kernel.xcaf_add_shape(doc, cube).unwrap();
    kernel.xcaf_set_name(doc, label, "Größe").unwrap();
    kernel
        .xcaf_set_color(doc, label, 128.0 / 255.0, 0.5, 0.5)
        .unwrap();
    let info = kernel.xcaf_get_label_info(doc, label).unwrap();
    assert_eq!(info.name, "Größe");
    assert!((info.r - 128.0 / 255.0).abs() < 1e-6);
    let step = kernel.xcaf_export_step_with_unit(doc, "inch").unwrap();
    assert!(step.contains("CONVERSION_BASED_UNIT('INCH'"));
    assert!(
        step.contains("COLOUR_RGB('',0.50196"),
        "colour not written as sRGB"
    );
    kernel.xcaf_close(doc).unwrap();
}

/// The header's `FILE_SCHEMA` entry. OCCT wraps a long one across lines.
fn file_schema(step: &str) -> &str {
    let start = step.find("FILE_SCHEMA").unwrap();
    let len = step[start..].find(';').unwrap();
    &step[start..start + len]
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn step_export_writes_the_requested_schema() {
    const AP214: &str = "'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'";
    let mut kernel = kernel();
    // The cone's lateral face is dropped unless DirectFaces runs, so it shows
    // that every protocol still writes it.
    let cone = kernel.make_cone(8.0, 0.0, 24.0).unwrap();
    for (schema, protocol) in [
        ("AP203", "'CONFIG_CONTROL_DESIGN'"),
        ("ap214", AP214),
        (
            "AP242",
            "'AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF {1 0 10303 442 1 1 4 }'",
        ),
    ] {
        let step = kernel
            .export_step_with_options(cone, "inch", schema)
            .unwrap();
        assert!(file_schema(&step).contains(protocol), "{schema}: {step}");
        assert!(step.contains("CONVERSION_BASED_UNIT('INCH'"), "{schema}");
        assert_eq!(step.matches("CONICAL_SURFACE").count(), 1, "{schema}");

        let doc = kernel.xcaf_new_document().unwrap();
        kernel.xcaf_add_shape(doc, cone).unwrap();
        let step = kernel
            .xcaf_export_step_with_options(doc, "", schema)
            .unwrap();
        kernel.xcaf_close(doc).unwrap();
        assert!(
            file_schema(&step).contains(protocol),
            "xcaf {schema}: {step}"
        );
        assert!(step.contains(".MILLI."), "xcaf {schema}");
        assert_eq!(step.matches("CONICAL_SURFACE").count(), 1, "xcaf {schema}");
    }

    let step = kernel.export_step(cone).unwrap();
    assert_eq!(file_schema(&step), format!("FILE_SCHEMA(({AP214}))"));
    assert!(step.contains(".MILLI."));

    let err = kernel
        .export_step_with_options(cone, "", "AP209")
        .unwrap_err();
    assert!(
        format!("{err}").contains("unknown STEP schema 'AP209'"),
        "got: {err}"
    );
    let doc = kernel.xcaf_new_document().unwrap();
    assert!(
        kernel
            .xcaf_export_step_with_options(doc, "mm", "AP209")
            .is_err()
    );
    kernel.xcaf_close(doc).unwrap();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn freeform_bounds_and_volume_are_exact() {
    let mut kernel = kernel();
    let ellipsoid = kernel.make_ellipsoid(3.0, 4.0, 5.0).unwrap();
    let bb = kernel.get_bounding_box(ellipsoid, false).unwrap();
    for (got, want) in [
        (bb.min.x, -3.0),
        (bb.min.y, -4.0),
        (bb.min.z, -5.0),
        (bb.max.x, 3.0),
        (bb.max.y, 4.0),
        (bb.max.z, 5.0),
    ] {
        assert!((got - want).abs() < 1e-6, "bound {got}, expected {want}");
    }
    let volume = kernel.get_volume(ellipsoid).unwrap();
    let expected = 4.0 / 3.0 * std::f64::consts::PI * 60.0;
    assert!(
        ((volume - expected) / expected).abs() < 1e-6,
        "volume {volume}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn fill_surface_passes_through_boundary_and_point() {
    let mut kernel = kernel();
    let corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let edges: Vec<ShapeHandle> = (0..4)
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            kernel.make_line_edge(a.0, a.1, 0.0, b.0, b.1, 0.0).unwrap()
        })
        .collect();
    let face = kernel
        .fill_surface(occt_wasm::FillSurfaceRequest {
            free_edge_ids: &edges,
            supported_edge_ids: &[],
            support_face_ids: &[],
            support_orders: &[],
            guide_edge_ids: &[],
            flat_points: &[5.0, 5.0, 2.0],
            init_face_id: 0,
            params: &[],
        })
        .unwrap();
    assert_eq!(kernel.get_shape_type(face).unwrap(), "face");
    let apex = kernel.make_vertex(5.0, 5.0, 2.0).unwrap();
    let gap = kernel.distance_between(face, apex).unwrap();
    assert!(gap < 1e-3, "filling misses its point by {gap}");
    let bad = kernel.fill_surface(occt_wasm::FillSurfaceRequest {
        free_edge_ids: &edges,
        supported_edge_ids: &[],
        support_face_ids: &[],
        support_orders: &[],
        guide_edge_ids: &[],
        flat_points: &[],
        init_face_id: 0,
        params: &[3.0],
    });
    assert!(bad.is_err(), "a short params vector must be rejected");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn edge_on_surface_follows_a_cylinder_across_its_seam() {
    let mut kernel = kernel();
    let cylinder = kernel.make_cylinder(5.0, 10.0).unwrap();
    let face = kernel
        .get_sub_shapes(cylinder, "face")
        .unwrap()
        .into_iter()
        .find(|&f| kernel.surface_type(f).unwrap() == "cylinder")
        .unwrap();
    let points: Vec<f64> = (0..48)
        .flat_map(|i| {
            let t = std::f64::consts::TAU * f64::from(i) / 48.0;
            [5.0 * t.cos(), 5.0 * t.sin(), 5.0]
        })
        .collect();
    let edge = kernel.edge_on_surface(face, &points, 1e-4, true).unwrap();
    let length = kernel.get_length(edge).unwrap();
    let circumference = std::f64::consts::TAU * 5.0;
    assert!((length - circumference).abs() < 1e-2, "length {length}");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn fillet_with_history_reports_each_rounded_face_under_its_edge() {
    let mut kernel = kernel();
    let cube = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let edges = vertical_edges(&mut kernel, cube);
    let evo = kernel
        .fillet_with_history(cube, &edges, 1.0, &[], BOUND)
        .unwrap();

    let mut round_faces = Vec::new();
    for face in kernel.get_sub_shapes(evo.result(), "face").unwrap() {
        if kernel.surface_type(face).unwrap() == "cylinder" {
            round_faces.push(kernel.hash_code(face, BOUND).unwrap());
        }
    }
    assert_eq!(round_faces.len(), 4);

    let mut generated = std::collections::HashMap::new();
    let mut rest = evo.edge_generated.as_slice();
    while let [input, count, tail @ ..] = rest {
        let (results, next) = tail.split_at(usize::try_from(*count).unwrap());
        generated.insert(*input, results.to_vec());
        rest = next;
    }
    let mut from_edges = Vec::new();
    for edge in edges {
        let hash = kernel.hash_code(edge, BOUND).unwrap();
        let faces: Vec<i32> = generated[&hash]
            .iter()
            .copied()
            .filter(|h| round_faces.contains(h))
            .collect();
        assert_eq!(faces.len(), 1, "edge {hash} generated {faces:?}");
        from_edges.extend(faces);
    }
    from_edges.sort_unstable();
    from_edges.dedup();
    assert_eq!(from_edges.len(), 4);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn nurbs_surface_is_the_surface_its_poles_describe() {
    let mut kernel = kernel();
    // A bilinear patch: degree 1 both ways, so its middle is the poles' mean.
    let poles = [
        0.0, 0.0, 0.0, 0.0, 10.0, 2.0, //
        10.0, 0.0, 4.0, 10.0, 10.0, 0.0,
    ];
    let face = kernel
        .nurbs_surface(occt_wasm::NurbsSurfaceRequest {
            flat_poles: &poles,
            weights: &[],
            knots_u: &[0.0, 1.0],
            knots_v: &[0.0, 1.0],
            mults_u: &[2, 2],
            mults_v: &[2, 2],
            layout: &[1, 1, 2, 2, 0, 0],
        })
        .unwrap();
    assert_eq!(kernel.surface_type(face).unwrap(), "bspline");
    let middle = kernel.make_vertex(5.0, 5.0, 1.5).unwrap();
    let gap = kernel.distance_between(face, middle).unwrap();
    assert!(gap < 1e-9, "the patch misses its middle by {gap}");
    let bad = kernel.nurbs_surface(occt_wasm::NurbsSurfaceRequest {
        flat_poles: &poles,
        weights: &[],
        knots_u: &[0.0, 1.0],
        knots_v: &[0.0, 1.0],
        mults_u: &[2, 1],
        mults_v: &[2, 2],
        layout: &[1, 1, 2, 2, 0, 0],
    });
    assert!(
        bad.is_err(),
        "multiplicities short of the poles must be rejected"
    );
}

fn assert_near(actual: Vec3, expected: [f64; 3]) {
    let delta = [
        actual.x - expected[0],
        actual.y - expected[1],
        actual.z - expected[2],
    ];
    assert!(
        delta.iter().all(|d| d.abs() < 1e-9),
        "{actual:?} is not {expected:?}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn curve_data_reads_exact_parameters() {
    let mut kernel = kernel();
    let line = kernel
        .make_line_edge(1.0, 2.0, 3.0, 1.0, 2.0, 13.0)
        .unwrap();
    let data = kernel.curve_data(line).unwrap();
    assert!(data.first.abs() < 1e-9);
    assert!((data.last - 10.0).abs() < 1e-9);
    let CurveGeometry::Line { point, direction } = data.geometry else {
        panic!("expected a line, got {:?}", data.geometry);
    };
    assert_near(point, [1.0, 2.0, 3.0]);
    assert_near(direction, [0.0, 0.0, 1.0]);

    let circle = kernel
        .make_circle_edge(occt_wasm::MakeCircleEdgeRequest {
            cx: 1.0,
            cy: 2.0,
            cz: 3.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            radius: 7.0,
        })
        .unwrap();
    let CurveGeometry::Circle {
        center,
        axis,
        x_direction,
        radius,
    } = kernel.curve_data(circle).unwrap().geometry
    else {
        panic!("expected a circle");
    };
    assert_near(center, [1.0, 2.0, 3.0]);
    assert_near(axis, [0.0, 0.0, 1.0]);
    assert_near(x_direction, [1.0, 0.0, 0.0]);
    assert!((radius - 7.0).abs() < 1e-12);

    let bezier = kernel
        .make_bezier_edge(&[0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0])
        .unwrap();
    let CurveGeometry::Other { kind, bspline } = kernel.curve_data(bezier).unwrap().geometry else {
        panic!("expected another kind");
    };
    assert_eq!(kind, "bezier");
    assert_eq!(bspline.unwrap().degree, 2);

    let solid = kernel.make_box(1.0, 1.0, 1.0).unwrap();
    assert!(kernel.curve_data(solid).is_err());
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn surface_data_reads_exact_parameters() {
    let mut kernel = kernel();
    let cylinder = kernel.make_cylinder(5.0, 10.0).unwrap();
    let faces = kernel.get_sub_shapes(cylinder, "face").unwrap();
    let side = faces
        .into_iter()
        .find(|&f| kernel.surface_type(f).unwrap() == "cylinder")
        .unwrap();
    let SurfaceData::Cylinder { frame, radius } = kernel.surface_data(side).unwrap() else {
        panic!("expected a cylinder");
    };
    assert!((radius - 5.0).abs() < 1e-12);
    assert_near(frame.location, [0.0, 0.0, 0.0]);
    assert_near(frame.axis, [0.0, 0.0, 1.0]);
    assert!(frame.is_direct);

    let matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0];
    let moved = kernel.located(side, &matrix).unwrap();
    let SurfaceData::Cylinder { frame, .. } = kernel.surface_data(moved).unwrap() else {
        panic!("expected a cylinder");
    };
    assert_near(frame.location, [1.0, 2.0, 3.0]);

    let poles = [
        0.0, 0.0, 0.0, 0.0, 10.0, 2.0, //
        10.0, 0.0, 4.0, 10.0, 10.0, 0.0,
    ];
    let face = kernel
        .nurbs_surface(occt_wasm::NurbsSurfaceRequest {
            flat_poles: &poles,
            weights: &[],
            knots_u: &[0.0, 1.0],
            knots_v: &[0.0, 1.0],
            mults_u: &[2, 2],
            mults_v: &[2, 2],
            layout: &[1, 1, 2, 2, 0, 0],
        })
        .unwrap();
    let SurfaceData::BSpline(data) = kernel.surface_data(face).unwrap() else {
        panic!("expected a B-spline");
    };
    assert_eq!((data.degree_u, data.degree_v), (1, 1));
    assert_eq!((data.nb_poles_u, data.nb_poles_v), (2, 2));
    assert_eq!(data.poles, poles);
    assert!(data.weights.is_empty());
    assert_eq!(data.knots_u, [0.0, 1.0]);
    assert_eq!(data.knots_v, [0.0, 1.0]);
    assert_eq!(data.multiplicities_u, [2, 2]);
    assert_eq!(data.multiplicities_v, [2, 2]);
    assert!(!data.periodic_u);
    assert!(!data.periodic_v);

    assert!(kernel.surface_data(cylinder).is_err());
}

/// Decode `[input, count, result...]` runs into `(input, results)` pairs.
fn runs(flat: &[i32]) -> Vec<(i32, Vec<i32>)> {
    let mut decoded = Vec::new();
    let mut rest = flat;
    while let [input, count, tail @ ..] = rest {
        let (results, next) = tail.split_at(usize::try_from(*count).unwrap());
        decoded.push((*input, results.to_vec()));
        rest = next;
    }
    decoded
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn revolve_with_history_names_walls_and_caps() {
    let mut kernel = kernel();
    let points = [
        [5.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [8.0, 0.0, 4.0],
        [5.0, 0.0, 4.0],
    ];
    let edges: Vec<_> = (0..4)
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % 4]);
            kernel
                .make_line_edge(a[0], a[1], a[2], b[0], b[1], b[2])
                .unwrap()
        })
        .collect();
    let wire = kernel.make_wire(&edges).unwrap();
    let face = kernel.make_face(wire).unwrap();
    let evo = kernel
        .revolve_with_history(occt_wasm::RevolveWithHistoryRequest {
            shape_id: face,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
            angle_rad: 1.5,
            input_face_hashes: &[],
            hash_upper_bound: BOUND,
        })
        .unwrap();
    assert_eq!(evo.first_shape.len(), 1);
    assert_eq!(evo.last_shape.len(), 1);
    let walls = runs(&evo.edge_generated);
    assert_eq!(walls.len(), 4);

    let result_faces: Vec<i32> = kernel
        .get_sub_shapes(evo.result(), "face")
        .unwrap()
        .into_iter()
        .map(|f| kernel.hash_code(f, BOUND).unwrap())
        .collect();
    assert_eq!(result_faces.len(), 6);
    for hash in walls
        .iter()
        .flat_map(|(_, faces)| faces)
        .chain(&evo.first_shape)
        .chain(&evo.last_shape)
    {
        assert!(result_faces.contains(hash), "{hash} is not in the result");
    }

    let full = kernel
        .revolve_with_history(occt_wasm::RevolveWithHistoryRequest {
            shape_id: face,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
            angle_rad: std::f64::consts::TAU,
            input_face_hashes: &[],
            hash_upper_bound: BOUND,
        })
        .unwrap();
    assert!(full.first_shape.is_empty());
    assert!(full.last_shape.is_empty());
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn sweep_and_pipe_with_history_pair_spine_and_profile_edges() {
    let mut kernel = kernel();
    let spine_edges = [
        kernel
            .make_line_edge(0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
            .unwrap(),
        kernel
            .make_line_edge(0.0, 0.0, 10.0, 0.0, 10.0, 10.0)
            .unwrap(),
    ];
    let spine = kernel.make_wire(&spine_edges).unwrap();
    let profile_edges = [
        kernel
            .make_line_edge(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            .unwrap(),
        kernel.make_line_edge(1.0, 0.0, 0.0, 1.0, 1.0, 0.0).unwrap(),
    ];
    let profile = kernel.make_wire(&profile_edges).unwrap();

    let piped = kernel
        .pipe_with_history(profile, spine, &[], BOUND)
        .unwrap();
    let swept = kernel
        .sweep_with_history(occt_wasm::SweepWithHistoryRequest {
            profile_id: profile,
            spine_id: spine,
            aux_spine_id: 0,
            support_id: 0,
            options: &[],
            input_face_hashes: &[],
            hash_upper_bound: BOUND,
        })
        .unwrap();
    // Read back from the wires: make_wire can copy an edge to share a vertex.
    let spine_hashes: Vec<i32> = kernel
        .get_sub_shapes(spine, "edge")
        .unwrap()
        .into_iter()
        .map(|e| kernel.hash_code(e, BOUND).unwrap())
        .collect();
    let profile_hashes: Vec<i32> = kernel
        .get_sub_shapes(profile, "edge")
        .unwrap()
        .into_iter()
        .map(|e| kernel.hash_code(e, BOUND).unwrap())
        .collect();
    for evo in [piped, swept] {
        let walls = runs(&evo.edge_generated);
        assert_eq!(walls.len(), 2);
        assert!(evo.first_shape.is_empty());
        let result_faces: Vec<i32> = kernel
            .get_sub_shapes(evo.result(), "face")
            .unwrap()
            .into_iter()
            .map(|f| kernel.hash_code(f, BOUND).unwrap())
            .collect();

        let mut pairs = Vec::new();
        let mut rest = evo.spine_generated.as_slice();
        while let [spine_hash, profile_hash, count, tail @ ..] = rest {
            let (faces, next) = tail.split_at(usize::try_from(*count).unwrap());
            assert!(!faces.is_empty());
            let (_, wall) = walls.iter().find(|(edge, _)| edge == profile_hash).unwrap();
            for face in faces {
                assert!(result_faces.contains(face), "{face} is not in the result");
                assert!(wall.contains(face), "{face} is not under its profile edge");
            }
            pairs.push((*spine_hash, *profile_hash));
            rest = next;
        }
        pairs.sort_unstable();
        let mut expected: Vec<(i32, i32)> = spine_hashes
            .iter()
            .flat_map(|s| profile_hashes.iter().map(|p| (*s, *p)))
            .collect();
        expected.sort_unstable();
        assert_eq!(pairs, expected);
    }

    let short = kernel.sweep_with_history(occt_wasm::SweepWithHistoryRequest {
        profile_id: profile,
        spine_id: spine,
        aux_spine_id: 0,
        support_id: 0,
        options: &[1.0],
        input_face_hashes: &[],
        hash_upper_bound: BOUND,
    });
    assert!(short.is_err(), "a short options list must be rejected");
    let mut options = [0.0; 17];
    options[3] = 1.0;
    options[16] = 1.0;
    for bad in [f64::NAN, f64::INFINITY, 0.5, 3e10] {
        options[0] = bad;
        let swept = kernel.sweep_with_history(occt_wasm::SweepWithHistoryRequest {
            profile_id: profile,
            spine_id: spine,
            aux_spine_id: 0,
            support_id: 0,
            options: &options,
            input_face_hashes: &[],
            hash_upper_bound: BOUND,
        });
        assert!(swept.is_err(), "a mode of {bad} must be rejected");
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn loft_and_unify_with_history_report_their_inputs() {
    let mut kernel = kernel();
    let bottom = kernel
        .make_circle_edge(occt_wasm::MakeCircleEdgeRequest {
            cx: 0.0,
            cy: 0.0,
            cz: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            radius: 5.0,
        })
        .unwrap();
    let top = kernel
        .make_circle_edge(occt_wasm::MakeCircleEdgeRequest {
            cx: 0.0,
            cy: 0.0,
            cz: 10.0,
            nx: 0.0,
            ny: 0.0,
            nz: 1.0,
            radius: 3.0,
        })
        .unwrap();
    let wires = [
        kernel.make_wire(&[bottom]).unwrap(),
        kernel.make_wire(&[top]).unwrap(),
    ];
    let evo = kernel
        .loft_with_history(occt_wasm::LoftWithHistoryRequest {
            wire_ids: &wires,
            is_solid: true,
            ruled: false,
            start_vertex_id: 0,
            end_vertex_id: 0,
            input_face_hashes: &[],
            hash_upper_bound: BOUND,
        })
        .unwrap();
    assert_eq!(evo.first_shape.len(), 1);
    assert_eq!(evo.last_shape.len(), 1);
    assert_eq!(runs(&evo.edge_generated).len(), 2);

    let left = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let right = kernel.make_box(10.0, 10.0, 10.0).unwrap();
    let right = kernel.translate(right, 10.0, 0.0, 0.0).unwrap();
    let fused = kernel.fuse(left, right).unwrap();
    let hashes: Vec<i32> = kernel
        .get_sub_shapes(fused, "face")
        .unwrap()
        .into_iter()
        .map(|f| kernel.hash_code(f, BOUND).unwrap())
        .collect();
    let unified = kernel
        .unify_same_domain_with_history(fused, &hashes, BOUND)
        .unwrap();
    assert_eq!(
        kernel
            .get_sub_shapes(unified.result(), "face")
            .unwrap()
            .len(),
        6
    );
    assert!(!unified.modified.is_empty());

    let matrix = [1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let moved = kernel
        .transform_with_history(left, &matrix, &[], BOUND)
        .unwrap();
    assert!(moved.first_shape.is_empty());
    assert!(!moved.edge_modified.is_empty());
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "requires optimized wasmtime; run in release mode"
)]
fn thread_groove_is_a_valid_solid_of_the_exact_volume() {
    let mut kernel = kernel();
    // An ISO 68-1 M6 x 1 shaft groove, 10 long: the trapezoid in the meridian
    // plane swept round the axis as many turns as the length holds (Pappus).
    let (pitch, length) = (1.0_f64, 10.0_f64);
    let h = 3.0_f64.sqrt() / 2.0 * pitch;
    let (cover, root) = (3.0, (-5.0 / 8.0_f64).mul_add(h, 3.0));
    let (cover_half, root_half) = (7.0 * pitch / 16.0, pitch / 8.0);
    let r = [cover, root, root, cover];
    let z = [-cover_half, -root_half, root_half, cover_half];
    let (mut area, mut moment) = (0.0, 0.0);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let cross = r[i].mul_add(z[j], -(r[j] * z[i]));
        area += cross;
        moment = (r[i] + r[j]).mul_add(cross, moment);
    }
    let exact =
        (area / 2.0).abs() * std::f64::consts::TAU * (moment / (3.0 * area)) * (length / pitch);

    let groove = kernel
        .thread_groove(occt_wasm::ThreadGrooveRequest {
            px: 0.0,
            py: 0.0,
            pz: 0.0,
            dx: 0.0,
            dy: 0.0,
            dz: 1.0,
            pitch,
            length,
            cover_radius: cover,
            root_radius: root,
            cover_half_width: cover_half,
            root_half_width: root_half,
            phase: 0.137,
        })
        .unwrap();
    assert!(kernel.is_valid(groove).unwrap());
    let volume = kernel.get_volume(groove).unwrap();
    assert!(
        (volume / exact - 1.0).abs() < 1e-4,
        "volume {volume}, exact {exact}"
    );
}
