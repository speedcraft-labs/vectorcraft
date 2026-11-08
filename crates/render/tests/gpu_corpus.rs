#![cfg(feature = "gpu")]

use vectorcraft_doc::{Document, NodeKind};
use vectorcraft_render::{RenderOptions, Renderer};

const MAX_CHANNEL_ERROR: u8 = 2;

#[test]
fn gpu_matches_cpu_oracle_on_pinned_vectorcraft_corpus() {
    let Some(corpus) = std::env::var_os("ARTCRAFT_CORPUS") else {
        eprintln!("skipped: ARTCRAFT_CORPUS is not set");
        return;
    };
    let file = std::path::Path::new(&corpus).join("vectorcraft-fixtures/still-life.vectorcraft");
    let bytes = std::fs::read(&file).unwrap_or_else(|error| panic!("{}: {error}", file.display()));
    let doc = vectorcraft_format::load(&bytes).expect("load pinned native VectorCraft fixture");
    let region = doc.artboards.first().expect("fixture artboard").rect;
    let cpu = Renderer::new().render_region(&doc, region, 1.0, true);
    let mut gpu_renderer = Renderer::new_gpu();
    let opts = RenderOptions { background: Some([255; 4]), ..Default::default() };
    let fallback = gpu_renderer.render_region_with(&doc, region, 1.0, &opts);
    assert!(!gpu_renderer.stats.gpu_rasterized, "3D/effect content must stay on the CPU fallback");
    assert_eq!(cpu.pixels, fallback.pixels, "unsupported corpus content must retain CPU oracle output exactly");

    // The fixture contains both plain vector shapes and content outside Vello Hybrid's current
    // feature set. Isolate one plain path from that real file to exercise the GPU backend itself.
    let source = doc
        .layers
        .iter()
        .flat_map(|layer| layer.children().into_iter().flatten())
        .find(|node| {
            !node.appearance.effects.iter().any(|effect| effect.visible) && matches!(node.kind, NodeKind::Path { clipping: false, guide: false, .. })
        })
        .expect("fixture has a plain path");
    let mut subset = Document::new(region.width(), region.height());
    let layer = subset.layers.first().map(|layer| layer.id);
    let mut path = (**source).clone();
    path.id = subset.alloc_id();
    if let NodeKind::Path { live, .. } = &mut path.kind {
        *live = None;
    }
    subset.insert(layer, usize::MAX, path).expect("add corpus path to subset");
    let cpu = Renderer::new().render_region(&subset, region, 1.0, true);
    let gpu = gpu_renderer.render_region_with(&subset, region, 1.0, &opts);
    if !gpu_renderer.gpu_available() {
        eprintln!("skipped: no Vello GPU adapter is available");
        return;
    }
    println!("Vello GPU adapter: {}", gpu_renderer.gpu_adapter_name().as_deref().unwrap_or("unknown"));
    assert!(gpu_renderer.stats.gpu_rasterized, "subset must exercise Vello GPU, not just CPU fallback");
    assert_eq!((cpu.width, cpu.height), (gpu.width, gpu.height));
    assert_eq!(cpu.pixels.len(), gpu.pixels.len());

    let (max, total, over_bound) = cpu.pixels.iter().zip(&gpu.pixels).fold((0u8, 0u64, 0usize), |(max, total, over), (&a, &b)| {
        let error = a.abs_diff(b);
        (max.max(error), total + u64::from(error), over + usize::from(error > MAX_CHANNEL_ERROR))
    });
    let mean = total as f64 / cpu.pixels.len() as f64;
    println!("corpus still-life plain path: max channel error {max}/255, mean {mean:.6}/255, channels over {MAX_CHANNEL_ERROR}/255: {over_bound}");
    assert!(
        max <= MAX_CHANNEL_ERROR,
        "{} plain-path subset: max per-channel error {max}, mean {mean:.4}, {over_bound} channels over {MAX_CHANNEL_ERROR}",
        file.display()
    );
}
