//! Strict regressions for the validation facade. No placeholder/debug skips.
use occt_wasm::{
    OcctKernel, ShapeHandle,
    wasmtime::{Engine, Module},
};
use std::sync::OnceLock;

fn kernel() -> Result<OcctKernel, Box<dyn std::error::Error>> {
    static MODULE: OnceLock<Result<Module, String>> = OnceLock::new();
    let module = MODULE.get_or_init(|| {
        let engine = Engine::new(&OcctKernel::engine_config()).map_err(|e| e.to_string())?;
        OcctKernel::compile(&engine).map_err(|e| e.to_string())
    });
    Ok(OcctKernel::from_module(
        module.as_ref().map_err(String::as_str)?,
    )?)
}

fn volume_at(
    kernel: &mut OcctKernel,
    shape: ShapeHandle,
    point: [f64; 3],
    epsilon: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let [x, y, z] = point;
    let values = kernel.get_signed_face_volume_integrals(shape, x, y, z, epsilon)?;
    assert_eq!(values.len(), 12, "six face mass/error pairs");
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.chunks_exact(2).all(|pair| pair[1] >= 0.0));
    Ok(values.chunks_exact(2).map(|pair| pair[0]).sum())
}

#[test]
fn validation_occurrences_preserve_topology_multiplicity() -> Result<(), Box<dyn std::error::Error>>
{
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    assert_eq!(kernel.sub_shape_count(shape, "edge")?, 12);
    assert_eq!(kernel.sub_shape_occurrence_count(shape, "edge")?, 24);
    assert_eq!(kernel.sub_shape_occurrence_count(shape, "vertex")?, 48);
    assert_eq!(kernel.sub_shape_occurrence_count(shape, "solid")?, 1);
    let edges = kernel.get_sub_shape_occurrences(shape, "edge")?;
    assert_eq!(edges.len(), 24);
    for edge in edges {
        kernel.release(edge)?;
    }
    assert!(kernel.sub_shape_occurrence_count(shape, "invalid").is_err());
    Ok(())
}

#[test]
fn validation_signed_volume_and_bounds_are_reference_independent()
-> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let moved = kernel.translate(shape, 1000., -2000., 3000.)?;
    for (solid, origin) in [(shape, [0., 0., 0.]), (moved, [1000., -2000., 3000.])] {
        let bbox = kernel.get_optimal_bounds(solid, false, false)?;
        assert!((bbox.min.x - origin[0]).abs() < 1e-10);
        assert!((bbox.max.x - origin[0] - 2.).abs() < 1e-10);
        for point in [
            [0., 0., 0.],
            origin,
            [origin[0] + 1., origin[1] + 1.5, origin[2] + 2.],
        ] {
            for epsilon in [1e-12, 1e-13] {
                assert!((volume_at(&mut kernel, solid, point, epsilon)? - 24.).abs() < 1e-8);
            }
        }
    }
    assert!(
        kernel
            .get_signed_face_volume_integrals(shape, 0., 0., 0., 0.)
            .is_err()
    );
    let compound = kernel.make_compound(&[shape, moved])?;
    assert!(
        kernel
            .get_signed_face_volume_integrals(compound, 0., 0., 0., 1e-12)
            .is_err()
    );
    Ok(())
}

#[test]
fn validation_classifier_does_not_coerce_boundary() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    assert_eq!(kernel.classify_solid_point(shape, 1., 1., 1., 1e-8)?, 1);
    assert_eq!(kernel.classify_solid_point(shape, 10., 10., 10., 1e-8)?, 0);
    assert_eq!(kernel.classify_solid_point(shape, 0., 1., 1., 1e-8)?, 2);
    assert!(
        kernel
            .classify_solid_point(shape, f64::NAN, 0., 0., 1e-8)
            .is_err()
    );
    Ok(())
}

#[test]
fn validation_tessellation_retains_f64_world_coordinates() -> Result<(), Box<dyn std::error::Error>>
{
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let origin = 1_000_000.001;
    let moved = kernel.translate(shape, origin, 0., 0.)?;
    let triangles = kernel.tessellate_triangles64(moved, 0.002, 0.2)?;
    assert_eq!(triangles.len(), 12 * 9);
    let x_min = triangles
        .chunks_exact(3)
        .map(|p| p[0])
        .fold(f64::INFINITY, f64::min);
    assert!((x_min - origin).abs() < 1e-9);
    assert!(kernel.tessellate_triangles64(moved, 0., 0.2).is_err());
    Ok(())
}

#[test]
fn validation_step_byte_import_roundtrips() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let step = kernel.export_step(shape)?;
    let imported = kernel.import_step_bytes(step.as_bytes())?;
    assert_eq!(imported.roots, 1);
    assert_eq!(imported.transferred, 1);
    assert!((kernel.get_volume(imported.shape)? - 24.).abs() < 1e-10);
    assert!(kernel.import_step_bytes(b"not STEP\xff").is_err());
    Ok(())
}

#[test]
fn validation_native_mass_property_record() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let properties = kernel.get_b_rep_properties(shape)?;
    assert!((properties.volume - 24.).abs() < 1e-10);
    assert!((properties.area - 52.).abs() < 1e-10);
    assert!((properties.edge_length - 72.).abs() < 1e-10);
    for (actual, expected) in properties.center_of_mass.into_iter().zip([1., 1.5, 2.]) {
        assert!((actual - expected).abs() < 1e-10);
    }
    for (row, expected) in
        properties
            .inertia
            .into_iter()
            .zip([[50., 0., 0.], [0., 40., 0.], [0., 0., 26.]])
    {
        for (actual, expected) in row.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
    Ok(())
}
