#[cfg(feature = "gpu")]
use super::gpu::GpuRuntime;
#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex};
use vello_cpu::{Pixmap, RasterizerSettings, Resources};

/// Records a frame once for both Vello backends. CPU rendering remains the fallback and oracle.
pub(crate) struct RenderContext {
    cpu: vello_cpu::RenderContext,
    record_cpu: bool,
    #[cfg(feature = "gpu")]
    gpu: Option<(vello::Scene, Arc<Mutex<GpuRuntime>>)>,
    #[cfg(feature = "gpu")]
    gpu_supported: bool,
    #[cfg(feature = "gpu")]
    gpu_transform: super::Affine,
    #[cfg(feature = "gpu")]
    gpu_paint: vello::peniko::Brush,
    #[cfg(feature = "gpu")]
    gpu_paint_transform: Option<super::Affine>,
    #[cfg(feature = "gpu")]
    gpu_fill_rule: vello_cpu::peniko::Fill,
    #[cfg(feature = "gpu")]
    gpu_stroke: vello_cpu::kurbo::Stroke,
    #[cfg(feature = "gpu")]
    gpu_layers: Vec<bool>,
    gpu_used: bool,
    width: u16,
    height: u16,
}

impl RenderContext {
    pub(crate) fn new_with(width: u16, height: u16, settings: vello_cpu::RenderSettings) -> Self {
        Self {
            cpu: vello_cpu::RenderContext::new_with(width, height, settings),
            record_cpu: true,
            #[cfg(feature = "gpu")]
            gpu: None,
            #[cfg(feature = "gpu")]
            gpu_supported: false,
            #[cfg(feature = "gpu")]
            gpu_transform: super::Affine::IDENTITY,
            #[cfg(feature = "gpu")]
            gpu_paint: vello::peniko::Brush::default(),
            #[cfg(feature = "gpu")]
            gpu_paint_transform: None,
            #[cfg(feature = "gpu")]
            gpu_fill_rule: vello_cpu::peniko::Fill::NonZero,
            #[cfg(feature = "gpu")]
            gpu_stroke: vello_cpu::kurbo::Stroke::new(1.0),
            #[cfg(feature = "gpu")]
            gpu_layers: Vec::new(),
            gpu_used: false,
            width,
            height,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn with_gpu(width: u16, height: u16, threads: u16, runtime: Arc<Mutex<GpuRuntime>>) -> Self {
        Self {
            cpu: vello_cpu::RenderContext::new_with(width, height, vello_cpu::RenderSettings { num_threads: threads, ..Default::default() }),
            record_cpu: false,
            gpu: Some((vello::Scene::new(), runtime)),
            gpu_supported: true,
            gpu_transform: super::Affine::IDENTITY,
            gpu_paint: vello::peniko::Brush::default(),
            gpu_paint_transform: None,
            gpu_fill_rule: vello_cpu::peniko::Fill::NonZero,
            gpu_stroke: vello_cpu::kurbo::Stroke::new(1.0),
            gpu_layers: Vec::new(),
            gpu_used: false,
            width,
            height,
        }
    }

    pub(crate) fn width(&self) -> u16 {
        self.width
    }
    pub(crate) fn height(&self) -> u16 {
        self.height
    }
    pub(crate) fn gpu_used(&self) -> bool {
        self.gpu_used
    }
    pub(crate) fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
    pub(crate) fn transform(&self) -> &super::Affine {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            return &self.gpu_transform;
        }
        self.cpu.transform()
    }

    pub(crate) fn reset(&mut self) {
        if self.record_cpu {
            self.cpu.reset();
        }
        self.gpu_used = false;
        #[cfg(feature = "gpu")]
        if let Some((scene, _)) = &mut self.gpu {
            scene.reset();
            self.gpu_supported = true;
            self.gpu_transform = super::Affine::IDENTITY;
            self.gpu_paint = vello::peniko::Brush::default();
            self.gpu_paint_transform = None;
            self.gpu_fill_rule = vello_cpu::peniko::Fill::NonZero;
            self.gpu_stroke = vello_cpu::kurbo::Stroke::new(1.0);
            self.gpu_layers.clear();
        }
    }

    pub(crate) fn flush(&mut self) {
        if self.record_cpu {
            self.cpu.flush();
        }
    }

    pub(crate) fn render_with(&mut self, pixmap: &mut Pixmap, resources: &mut Resources, settings: RasterizerSettings) {
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, runtime)) = &self.gpu
        {
            if let Ok(mut runtime) = runtime.lock()
                && runtime.render_into(scene, self.width, self.height, pixmap.data_as_u8_slice_mut())
            {
                self.gpu_used = true;
                return;
            }
        }
        if self.record_cpu {
            self.cpu.render_with(pixmap, resources, settings);
        }
    }

    pub(crate) fn render(&mut self, pixmap: &mut Pixmap, resources: &mut Resources) {
        self.render_with(pixmap, resources, RasterizerSettings::default());
    }

    pub(crate) fn set_aliasing_threshold(&mut self, threshold: Option<u8>) {
        if self.record_cpu {
            self.cpu.set_aliasing_threshold(threshold);
        }
        #[cfg(feature = "gpu")]
        if threshold.is_some() {
            self.gpu_supported = false;
        }
    }

    pub(crate) fn set_transform(&mut self, transform: super::Affine) {
        if self.record_cpu {
            self.cpu.set_transform(transform);
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            self.gpu_transform = transform;
        }
    }

    pub(crate) fn set_paint<P>(&mut self, paint: P)
    where
        P: Into<vello_cpu::PaintType>,
    {
        let cpu_paint: vello_cpu::PaintType = paint.into();
        if self.record_cpu {
            self.cpu.set_paint(cpu_paint.clone());
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            match cpu_paint {
                vello_cpu::PaintType::Solid(color) => self.gpu_paint = color.into(),
                vello_cpu::PaintType::Gradient(gradient) => self.gpu_paint = gradient.into(),
                vello_cpu::PaintType::Image(_) => self.gpu_supported = false,
            }
        }
    }

    pub(crate) fn set_paint_transform(&mut self, transform: super::Affine) {
        if self.record_cpu {
            self.cpu.set_paint_transform(transform);
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            self.gpu_paint_transform = Some(transform);
        }
    }

    pub(crate) fn reset_paint_transform(&mut self) {
        if self.record_cpu {
            self.cpu.reset_paint_transform();
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            self.gpu_paint_transform = None;
        }
    }

    pub(crate) fn set_fill_rule(&mut self, rule: vello_cpu::peniko::Fill) {
        if self.record_cpu {
            self.cpu.set_fill_rule(rule);
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            self.gpu_fill_rule = rule;
        }
    }

    pub(crate) fn set_stroke(&mut self, stroke: vello_cpu::kurbo::Stroke) {
        if self.record_cpu {
            self.cpu.set_stroke(stroke.clone());
        }
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            self.gpu_stroke = stroke;
        }
    }

    pub(crate) fn set_blend_mode(&mut self, mode: vello_cpu::peniko::BlendMode) {
        if self.record_cpu {
            self.cpu.set_blend_mode(mode);
        }
        #[cfg(feature = "gpu")]
        if mode != vello_cpu::peniko::BlendMode::default() {
            self.gpu_supported = false;
        }
    }

    pub(crate) fn fill_path(&mut self, path: &super::BezPath) {
        if self.record_cpu {
            self.cpu.fill_path(path);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.fill(self.gpu_fill_rule, self.gpu_transform, &self.gpu_paint, self.gpu_paint_transform, path);
        }
    }

    pub(crate) fn stroke_path(&mut self, path: &super::BezPath) {
        if self.record_cpu {
            self.cpu.stroke_path(path);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.stroke(&self.gpu_stroke, self.gpu_transform, &self.gpu_paint, self.gpu_paint_transform, path);
        }
    }

    pub(crate) fn fill_rect(&mut self, rect: &vello_cpu::kurbo::Rect) {
        if self.record_cpu {
            self.cpu.fill_rect(rect);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.fill(self.gpu_fill_rule, self.gpu_transform, &self.gpu_paint, self.gpu_paint_transform, rect);
        }
    }

    pub(crate) fn push_clip_path(&mut self, path: &super::BezPath) {
        if self.record_cpu {
            self.cpu.push_clip_path(path);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.push_clip_layer(self.gpu_fill_rule, self.gpu_transform, path);
            self.gpu_layers.push(true);
        }
    }

    pub(crate) fn pop_clip_path(&mut self) {
        if self.record_cpu {
            self.cpu.pop_clip_path();
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && self.gpu_layers.pop().unwrap_or(false)
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.pop_layer();
        }
    }

    pub(crate) fn push_clip_layer(&mut self, path: &super::BezPath) {
        if self.record_cpu {
            self.cpu.push_clip_layer(path);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.push_clip_layer(self.gpu_fill_rule, self.gpu_transform, path);
            self.gpu_layers.push(true);
        }
    }

    pub(crate) fn push_layer(
        &mut self,
        clip: Option<&super::BezPath>,
        blend: Option<vello_cpu::peniko::BlendMode>,
        opacity: Option<f32>,
        mask: Option<vello_cpu::Mask>,
        filter: Option<vello_common::filter_effects::Filter>,
    ) {
        if self.record_cpu {
            self.cpu.push_layer(clip, blend, opacity, mask.clone(), filter.clone());
        }
        #[cfg(feature = "gpu")]
        if let Some((scene, _)) = &mut self.gpu {
            let blend_supported = blend.is_none_or(|mode| mode == vello_cpu::peniko::BlendMode::default());
            let opacity_supported = opacity.is_none_or(|alpha| alpha == 1.0);
            let mut pushed = false;
            if mask.is_some() || filter.is_some() || !blend_supported || !opacity_supported {
                self.gpu_supported = false;
            } else if self.gpu_supported
                && let Some(clip) = clip
            {
                scene.push_clip_layer(self.gpu_fill_rule, self.gpu_transform, clip);
                pushed = true;
            }
            self.gpu_layers.push(pushed);
        }
    }

    pub(crate) fn pop_layer(&mut self) {
        if self.record_cpu {
            self.cpu.pop_layer();
        }
        #[cfg(feature = "gpu")]
        if self.gpu_supported
            && self.gpu_layers.pop().unwrap_or(false)
            && let Some((scene, _)) = &mut self.gpu
        {
            scene.pop_layer();
        }
    }
}
