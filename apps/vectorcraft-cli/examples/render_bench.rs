//! Paired 50k-path frame benchmark used by craft-journeys.

use std::time::Instant;

use sha2::{Digest, Sha256};
use vectorcraft_color::{Color, Paint};
use vectorcraft_doc::{Appearance, Document, Node};
use vectorcraft_geom::{Affine, Point, Rect, shapes};
use vectorcraft_render::{RenderOptions, Renderer};

fn synthetic(paths: usize) -> Result<Document, String> {
    let page = (1600.0, 1200.0);
    let mut doc = Document::new(page.0, page.1);
    let layer = doc.layers.first().map(|layer| layer.id);
    for i in 0..paths {
        let x = ((i * 7919) % 160_000) as f64 / 100.0;
        let y = ((i * 3571) % 120_000) as f64 / 100.0;
        let size = 3.0 + (i % 220) as f64 / 10.0;
        let color = Color::rgb((i % 251) as f32 / 250.0, ((i * 7) % 251) as f32 / 250.0, ((i * 13) % 251) as f32 / 250.0);
        let path = if i % 2 == 0 {
            shapes::ellipse(Rect::from_center_size(Point::new(x, y), (2.0 * size, 2.0 * size)))
        } else {
            shapes::rectangle(Rect::new(x, y, x + 2.0 * size, y + size))
        };
        let node = Node::path(doc.alloc_id(), path, Appearance::basic(Paint::solid(color), Paint::None, 0.0));
        doc.insert(layer, usize::MAX, node).map_err(|error| error.to_string())?;
    }
    Ok(doc)
}

fn main() -> Result<(), String> {
    let paths = std::env::args().skip(1).find_map(|arg| arg.strip_prefix("--paths=").and_then(|n| n.parse().ok())).unwrap_or(50_000);
    let doc = synthetic(paths)?;
    #[cfg(feature = "gpu-raster")]
    let mut renderer = Renderer::new_gpu();
    #[cfg(not(feature = "gpu-raster"))]
    let mut renderer = Renderer::new();
    let options = RenderOptions::default();
    let view = Affine::scale(1.6);
    let warmup = renderer.render(&doc, 2880, 1800, view, &options);
    #[cfg(feature = "gpu-raster")]
    if !renderer.gpu_available() {
        return Err("no Vello GPU adapter is available".into());
    }
    #[cfg(feature = "gpu-raster")]
    println!("CRAFT_GPU_ADAPTER={}", renderer.gpu_adapter_name().unwrap_or_else(|| "unknown".into()));
    #[cfg(feature = "gpu-raster")]
    if !renderer.stats.gpu_rasterized {
        return Err("the benchmark workload was not rasterized on Vello GPU".into());
    }
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let rendered = renderer.render(&doc, 2880, 1800, view, &options);
        #[cfg(feature = "gpu-raster")]
        if !renderer.stats.gpu_rasterized {
            return Err("the benchmark workload left the Vello GPU path".into());
        }
        std::hint::black_box(rendered.pixels);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    println!("CRAFT_JOURNEY_NS={}", samples[samples.len() / 2]);
    #[cfg(feature = "gpu-raster")]
    let oracle_pixels = {
        const MAX_CHANNEL_ERROR: u8 = 2;
        let oracle = Renderer::new().render(&doc, 2880, 1800, view, &options);
        if (warmup.width, warmup.height) != (oracle.width, oracle.height) || warmup.pixels.len() != oracle.pixels.len() {
            return Err("Vello GPU output dimensions differ from the CPU oracle".into());
        }
        let max_error = warmup.pixels.iter().zip(&oracle.pixels).map(|(&gpu, &cpu)| gpu.abs_diff(cpu)).max().unwrap_or(0);
        if max_error > MAX_CHANNEL_ERROR {
            return Err(format!("Vello GPU output exceeds the CPU-oracle bound: {max_error}/255 > {MAX_CHANNEL_ERROR}/255"));
        }
        eprintln!("Vello GPU output is within {max_error}/255 per channel of the CPU oracle (bound {MAX_CHANNEL_ERROR}/255)");
        oracle.pixels
    };
    #[cfg(not(feature = "gpu-raster"))]
    let oracle_pixels = warmup.pixels;
    let output_hash = format!("{:x}", Sha256::digest(&oracle_pixels));
    println!("CRAFT_OUTPUT_SHA256={output_hash}");
    Ok(())
}
