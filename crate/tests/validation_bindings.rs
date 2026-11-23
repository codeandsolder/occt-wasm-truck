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

fn entity_id(step: &str, kind: &str) -> Result<u32, Box<dyn std::error::Error>> {
    for record in step.split(';') {
        if let Some((label, body)) = record.trim().split_once('=')
            && body.trim_start().starts_with(kind)
            && let Some(label) = label.trim().strip_prefix('#')
        {
            return Ok(label.parse()?);
        }
    }
    Err(format!("missing exported STEP entity {kind}").into())
}

// This test helper rewrites the box export's known ASCII records, not arbitrary
// input STEP. The production code resolves labels from OCCT's parsed model.
fn relabel_and_reverse_box_step(step: &str) -> Result<String, Box<dyn std::error::Error>> {
    use std::fmt::Write;
    let (header, data) = step.split_once("DATA;").ok_or("missing DATA")?;
    let (records, footer) = data.split_once("ENDSEC;").ok_or("missing ENDSEC")?;
    let mut relabeled = String::new();
    let mut chunks = records.split('#');
    relabeled.push_str(chunks.next().ok_or("missing DATA records")?);
    for chunk in chunks {
        let len = chunk
            .find(|ch: char| !ch.is_ascii_digit())
            .unwrap_or(chunk.len());
        let id = chunk[..len].parse::<u32>()?;
        write!(relabeled, "#{}{}", 10000 + 3 * id, &chunk[len..])?;
    }
    let mut records: Vec<_> = relabeled
        .split_terminator(';')
        .map(str::trim)
        .filter(|record| !record.is_empty())
        .collect();
    assert!(records.iter().all(|record| record.starts_with('#')));
    records.reverse();
    let body = records.join(";\n");
    Ok(format!("{header}DATA;\n{body};\nENDSEC;{footer}"))
}

#[test]
fn validation_step_entity_import_uses_textual_labels_not_rank()
-> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let step = kernel.export_step(shape)?;
    let id = entity_id(&step, "MANIFOLD_SOLID_BREP(")?;
    let original = kernel.import_step_entity_bytes(step.as_bytes(), id)?;
    assert!((kernel.get_b_rep_properties(original.shape)?.volume - 24.).abs() < 1e-10);

    let reordered = relabel_and_reverse_box_step(&step)?;
    let renamed = 10000 + 3 * id;
    let imported = kernel.import_step_entity_bytes(reordered.as_bytes(), renamed)?;
    assert_ne!(imported.rank, renamed);
    assert_ne!(imported.rank, original.rank);
    assert_eq!(
        kernel.sub_shape_occurrence_count(imported.shape, "solid")?,
        1
    );
    assert!((kernel.get_b_rep_properties(imported.shape)?.volume - 24.).abs() < 1e-10);
    for invalid in [0, id, u32::MAX] {
        assert!(
            kernel
                .import_step_entity_bytes(reordered.as_bytes(), invalid)
                .is_err()
        );
    }
    let not_shape = entity_id(&reordered, "APPLICATION_PROTOCOL_DEFINITION(")?;
    assert!(
        kernel
            .import_step_entity_bytes(reordered.as_bytes(), not_shape)
            .is_err()
    );

    // Reusing the same textual label for two different entity records is
    // malformed; it must not silently resolve to whichever record came last.
    let duplicate = reordered
        .split(';')
        .find(|record| record.trim_start().starts_with(&format!("#{renamed} ")))
        .ok_or("missing relabeled solid record")?;
    let (header, data) = reordered.split_once("DATA;").ok_or("missing DATA")?;
    let malformed = format!("{header}DATA;\n{duplicate};{data}");
    assert!(
        kernel
            .import_step_entity_bytes(malformed.as_bytes(), renamed)
            .is_err()
    );
    Ok(())
}

#[test]
fn validation_adaptive_volume_reports_signed_integral_and_error()
-> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = kernel()?;
    let shape = kernel.make_box(2., 3., 4.)?;
    let moved = kernel.translate(shape, 1000., -2000., 3000.)?;
    let outer = kernel.make_box(3., 3., 3.)?;
    let cavity = kernel.make_box_from_corners(1., 1., 1., 2., 2., 2.)?;
    let hollow = kernel.cut(outer, cavity)?;
    for (solid, expected) in [(shape, 24.0), (moved, 24.0), (hollow, 26.0)] {
        for epsilon in [1e-8, 1e-12] {
            for spans in [false, true] {
                let measured = kernel.get_adaptive_volume(solid, epsilon, true, spans)?;
                assert!((measured.volume - expected).abs() < 1e-9);
                assert!(measured.relative_error.is_finite());
                assert!(measured.relative_error >= 0.0);
            }
        }
    }
    for invalid in [0.0, -1e-8, 1.0, f64::INFINITY, f64::NAN] {
        assert!(
            kernel
                .get_adaptive_volume(shape, invalid, true, true)
                .is_err()
        );
    }
    Ok(())
}
