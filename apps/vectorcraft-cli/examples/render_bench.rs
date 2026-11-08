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

fn edge_mask(cpu: &[u8], gpu: &[u8], width: usize, height: usize) -> Vec<bool> {
    let mut base = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let cpu_alpha = cpu[index * 4 + 3];
            let gpu_alpha = gpu[index * 4 + 3];
            if (1..255).contains(&cpu_alpha) || (1..255).contains(&gpu_alpha) {
                base[index] = true;
            }
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    let neighbor = ny * width + nx;
                    let discontinuity = (0..4).any(|channel| {
                        cpu[index * 4 + channel].abs_diff(cpu[neighbor * 4 + channel]) > 4
                            || gpu[index * 4 + channel].abs_diff(gpu[neighbor * 4 + channel]) > 4
                    });
                    if discontinuity {
                        base[index] = true;
                        base[neighbor] = true;
                    }
                }
            }
        }
    }
    let mut expanded = base.clone();
    for y in 0..height {
        for x in 0..width {
            if base[y * width + x] {
                for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                    for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        expanded[ny * width + nx] = true;
                    }
                }
            }
        }
    }
    expanded
}

fn synthetic_geometry_edge_mask(paths: usize, width: usize, height: usize, scale: f64) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    let mut mark_boundary = |x0: f64, y0: f64, x1: f64, y1: f64, ellipse: bool| {
        let pad = 2.0;
        let (left, top, right, bottom) = if ellipse { (x0 - x1, y0 - x1, x0 + x1, y0 + x1) } else { (x0, y0, x1, y1) };
        if right + pad < 0.0 || bottom + pad < 0.0 || left - pad >= width as f64 || top - pad >= height as f64 {
            return;
        }
        let min_x = ((left - pad).floor() as isize).clamp(0, width as isize - 1) as usize;
        let max_x = ((right + pad).ceil() as isize).clamp(0, width as isize - 1) as usize;
        let min_y = ((top - pad).floor() as isize).clamp(0, height as isize - 1) as usize;
        let max_y = ((bottom + pad).ceil() as isize).clamp(0, height as isize - 1) as usize;
        if min_x > max_x || min_y > max_y {
            return;
        }
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let cx = px as f64 + 0.5;
                let cy = py as f64 + 0.5;
                let distance = if ellipse {
                    (((cx - x0).powi(2) + (cy - y0).powi(2)).sqrt() - x1).abs()
                } else {
                    let dx = if cx < x0 {
                        x0 - cx
                    } else if cx > x1 {
                        cx - x1
                    } else {
                        0.0
                    };
                    let dy = if cy < y0 {
                        y0 - cy
                    } else if cy > y1 {
                        cy - y1
                    } else {
                        0.0
                    };
                    if dx == 0.0 && dy == 0.0 { (cx - x0).min(x1 - cx).min((cy - y0).min(y1 - cy)) } else { dx.hypot(dy) }
                };
                // Pixel centers within 1.5 device pixels of a contour cover the AA edge and a
                // half-pixel guard band for differing sample locations.
                if distance <= 1.5 {
                    mask[py * width + px] = true;
                }
            }
        }
    };
    for i in 0..paths {
        let x = ((i * 7919) % 160_000) as f64 / 100.0 * scale;
        let y = ((i * 3571) % 120_000) as f64 / 100.0 * scale;
        let size = (3.0 + (i % 220) as f64 / 10.0) * scale;
        if i % 2 == 0 {
            mark_boundary(x, y, size, 0.0, true);
        } else {
            mark_boundary(x, y, x + 2.0 * size, y + size, false);
        }
    }
    mask
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
        let oracle = Renderer::new().render(&doc, 2880, 1800, view, &options);
        if (warmup.width, warmup.height) != (oracle.width, oracle.height) || warmup.pixels.len() != oracle.pixels.len() {
            return Err("Vello GPU output dimensions differ from the CPU oracle".into());
        }
        let width = warmup.width as usize;
        let height = warmup.height as usize;
        let image_edges = edge_mask(&oracle.pixels, &warmup.pixels, width, height);
        let edges = synthetic_geometry_edge_mask(paths, width, height, 1.6);
        let image_edge_pixels = image_edges.iter().filter(|&&edge| edge).count();
        let mut opaque_pixels = 0usize;
        let mut opaque_color_mismatches = 0usize;
        let mut opaque_color_max = 0u8;
        let mut deep_interior_pixels = 0usize;
        let mut deep_interior_mismatches = 0usize;
        let mut edge_pixels = 0usize;
        let mut edge_max = 0u8;
        let mut edge_errors_over_4 = 0usize;
        let mut whole_pixel_mismatches = 0usize;
        let mut all_pixel_max = 0u8;
        let mut alpha_abs_sum = 0u64;
        let mut alpha_signed_sum = 0i64;
        let mut partial_coverage_pixels = 0usize;
        let mut partial_coverage_alpha_max = 0u8;
        let mut box3_alpha_sum_max = 0u32;
        let mut box3_rgba_sum_max = 0u32;
        let mut box3_alpha_errors = Vec::new();
        let mut box3_rgba_errors = Vec::new();
        for (pixel_index, (gpu, cpu)) in warmup.pixels.chunks_exact(4).zip(oracle.pixels.chunks_exact(4)).enumerate() {
            let error = gpu.iter().zip(cpu).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
            all_pixel_max = all_pixel_max.max(error);
            let alpha_error = cpu[3].abs_diff(gpu[3]);
            alpha_abs_sum += u64::from(alpha_error);
            alpha_signed_sum += i64::from(gpu[3]) - i64::from(cpu[3]);
            if (1..255).contains(&cpu[3]) || (1..255).contains(&gpu[3]) {
                partial_coverage_pixels += 1;
                partial_coverage_alpha_max = partial_coverage_alpha_max.max(alpha_error);
            }
            if (cpu[3] == 0 && gpu[3] == 255) || (cpu[3] == 255 && gpu[3] == 0) {
                whole_pixel_mismatches += 1;
            }
            if cpu[3] == 255 && gpu[3] == 255 {
                opaque_pixels += 1;
                let color_error = gpu[..3].iter().zip(&cpu[..3]).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
                opaque_color_max = opaque_color_max.max(color_error);
                opaque_color_mismatches += usize::from(color_error != 0);
            }
            if edges[pixel_index] {
                edge_pixels += 1;
                edge_max = edge_max.max(error);
                edge_errors_over_4 += usize::from(error > 4);
                let x = pixel_index % width;
                let y = pixel_index / width;
                let mut cpu_sum = [0u32; 4];
                let mut gpu_sum = [0u32; 4];
                let mut samples = 0u32;
                for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                    for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        let neighbor = (ny * width + nx) * 4;
                        for channel in 0..4 {
                            cpu_sum[channel] += u32::from(oracle.pixels[neighbor + channel]);
                            gpu_sum[channel] += u32::from(warmup.pixels[neighbor + channel]);
                        }
                        samples += 1;
                    }
                }
                let alpha_sum_error = cpu_sum[3].abs_diff(gpu_sum[3]);
                let rgba_sum_error = (0..4).map(|channel| cpu_sum[channel].abs_diff(gpu_sum[channel])).max().unwrap_or(0);
                box3_alpha_sum_max = box3_alpha_sum_max.max(alpha_sum_error);
                box3_rgba_sum_max = box3_rgba_sum_max.max(rgba_sum_error);
                box3_alpha_errors.push(alpha_sum_error as f64 / f64::from(samples));
                box3_rgba_errors.push(rgba_sum_error as f64 / f64::from(samples));
            } else {
                deep_interior_pixels += 1;
                deep_interior_mismatches += usize::from(error != 0);
            }
        }
        box3_alpha_errors.sort_by(f64::total_cmp);
        box3_rgba_errors.sort_by(f64::total_cmp);
        let percentile = |values: &[f64], p: f64| values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1).min(values.len() - 1)];
        println!(
            "DIAG_PIXELS opaque={opaque_pixels} opaque_rgb_mismatches={opaque_color_mismatches} opaque_rgb_max={opaque_color_max}/255 whole_0_vs_255={whole_pixel_mismatches}"
        );
        println!(
            "DIAG_EDGE_MASK geometry={edge_pixels} image_gradient={image_edge_pixels} edge_max={edge_max}/255 edge_pixels_over_4={edge_errors_over_4} deep_interior={deep_interior_pixels} deep_interior_mismatches={deep_interior_mismatches} max={all_pixel_max}/255"
        );
        println!(
            "DIAG_COVERAGE partial_pixels={partial_coverage_pixels} max_alpha_error={partial_coverage_alpha_max}/255 mean_abs_alpha_error={:.4}/255 signed_alpha_delta={} ({:.6}% of full-frame coverage)",
            alpha_abs_sum as f64 / (width * height) as f64,
            alpha_signed_sum,
            alpha_signed_sum as f64 / ((width * height) as f64 * 255.0) * 100.0,
        );
        println!("DIAG_EDGE_BOUND max_3x3_integrated_alpha={box3_alpha_sum_max}/255 max_3x3_integrated_rgba={box3_rgba_sum_max}/255; limits=255/255");
        println!(
            "DIAG_BOX3 alpha_mean={:.3}/255 p99={:.3}/255 p99.9={:.3}/255 max={:.3}/255 rgba p99={:.3}/255 p99.9={:.3}/255 max={:.3}/255",
            box3_alpha_errors.iter().sum::<f64>() / box3_alpha_errors.len() as f64,
            percentile(&box3_alpha_errors, 0.99),
            percentile(&box3_alpha_errors, 0.999),
            *box3_alpha_errors.last().unwrap_or(&0.0),
            percentile(&box3_rgba_errors, 0.99),
            percentile(&box3_rgba_errors, 0.999),
            *box3_rgba_errors.last().unwrap_or(&0.0),
        );
        let mean_alpha_error = alpha_abs_sum as f64 / (width * height) as f64;
        if deep_interior_mismatches != 0
            || whole_pixel_mismatches != 0
            || mean_alpha_error > 0.1
            || box3_alpha_sum_max > 255
            || box3_rgba_sum_max > 255
        {
            return Err("GPU/CPU edge-coverage parity gate failed".into());
        }
        println!("GPU/CPU edge-coverage parity gate passed: interiors exact, no whole-pixel flips, mean alpha error {mean_alpha_error:.4}/255");
        oracle.pixels
    };
    #[cfg(not(feature = "gpu-raster"))]
    let oracle_pixels = warmup.pixels;
    let output_hash = format!("{:x}", Sha256::digest(&oracle_pixels));
    println!("CRAFT_OUTPUT_SHA256={output_hash}");
    Ok(())
}
