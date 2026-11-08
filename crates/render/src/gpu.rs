use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

struct GpuTarget {
    texture: wgpu_gpu::Texture,
    view: wgpu_gpu::TextureView,
    readback: wgpu_gpu::Buffer,
    bytes_per_row: u32,
}

impl GpuTarget {
    fn new(device: &wgpu_gpu::Device, width: u16, height: u16) -> Self {
        let size = wgpu_gpu::Extent3d { width: u32::from(width), height: u32::from(height), depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu_gpu::TextureDescriptor {
            label: Some("VectorCraft Vello compute frame"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu_gpu::TextureDimension::D2,
            format: wgpu_gpu::TextureFormat::Rgba8Unorm,
            usage: wgpu_gpu::TextureUsages::STORAGE_BINDING | wgpu_gpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu_gpu::TextureViewDescriptor::default());
        let bytes_per_row = (u32::from(width) * 4).next_multiple_of(wgpu_gpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu_gpu::BufferDescriptor {
            label: Some("VectorCraft Vello readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu_gpu::BufferUsages::COPY_DST | wgpu_gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self { texture, view, readback, bytes_per_row }
    }
}

/// A headless Vello compute device shared by a renderer's frames. It is requested lazily, so
/// machines without a usable adapter continue through vello_cpu without changing the caller API.
pub(crate) struct GpuRuntime {
    _instance: wgpu_gpu::Instance,
    device: wgpu_gpu::Device,
    queue: wgpu_gpu::Queue,
    renderer: vello::Renderer,
    adapter_name: String,
    failed: Arc<AtomicBool>,
    targets: HashMap<(u16, u16), GpuTarget>,
}

impl GpuRuntime {
    pub(crate) fn new() -> Option<Self> {
        let instance = wgpu_gpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu_gpu::RequestAdapterOptions {
            power_preference: wgpu_gpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu_gpu::DeviceDescriptor {
            label: Some("VectorCraft Vello compute renderer"),
            required_features: wgpu_gpu::Features::empty(),
            ..Default::default()
        }))
        .ok()?;
        let renderer = vello::Renderer::new(&device, vello::RendererOptions::default()).ok()?;
        let failed = Arc::new(AtomicBool::new(false));
        let lost = Arc::clone(&failed);
        device.set_device_lost_callback(move |_reason, _message| {
            lost.store(true, Ordering::Release);
        });
        Some(Self { _instance: instance, device, queue, renderer, adapter_name: info.name, failed, targets: HashMap::new() })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub(crate) fn render_into(&mut self, scene: &vello::Scene, width: u16, height: u16, pixels: &mut [u8]) -> bool {
        if self.failed.load(Ordering::Acquire) || pixels.len() != usize::from(width) * usize::from(height) * 4 {
            return false;
        }
        let target = self.targets.entry((width, height)).or_insert_with(|| GpuTarget::new(&self.device, width, height));
        if self
            .renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                scene,
                &target.view,
                &vello::RenderParams {
                    base_color: vello::peniko::Color::TRANSPARENT,
                    width: u32::from(width),
                    height: u32::from(height),
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .is_err()
        {
            return false;
        }

        // Renderer::render returns a CPU pixmap today; keep the readback isolated in this backend
        // until the egui canvas can consume the wgpu texture directly.
        let row_bytes = usize::from(width) * 4;
        let mut encoder = self.device.create_command_encoder(&wgpu_gpu::CommandEncoderDescriptor { label: Some("VectorCraft Vello readback") });
        encoder.copy_texture_to_buffer(
            wgpu_gpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu_gpu::Origin3d::ZERO,
                aspect: wgpu_gpu::TextureAspect::All,
            },
            wgpu_gpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu_gpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(target.bytes_per_row), rows_per_image: None },
            },
            wgpu_gpu::Extent3d { width: u32::from(width), height: u32::from(height), depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        let slice = target.readback.slice(..);
        let (send, recv) = std::sync::mpsc::channel();
        slice.map_async(wgpu_gpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
        if self.device.poll(wgpu_gpu::PollType::wait_indefinitely()).is_err()
            || self.failed.load(Ordering::Acquire)
            || !matches!(recv.recv(), Ok(Ok(())))
        {
            return false;
        }
        let Ok(mapped) = slice.get_mapped_range() else {
            return false;
        };
        for (output_row, source_row) in pixels.chunks_exact_mut(row_bytes).zip(mapped.chunks_exact(target.bytes_per_row as usize)) {
            output_row.copy_from_slice(&source_row[..row_bytes]);
        }
        drop(mapped);
        target.readback.unmap();
        true
    }
}
