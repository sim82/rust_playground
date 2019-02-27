use rust_playground::render_bits;

use crate::render_bits::InputState;
use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;

use vulkano::buffer::{cpu_pool::CpuBufferPoolChunk, BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::image::ImmutableImage;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, Vector3};

use image::ImageBuffer;
use render_bits::{TexCoord, Vertex};
use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glyph_brush::{rusttype, BrushAction, BrushError, GlyphBrush, GlyphBrushBuilder, Section};

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

struct TestDelgate {
    last_time: std::time::Instant,

    text_console: Option<render_bits::text_console::TextConsole>,
}
impl TestDelgate {
    fn new() -> TestDelgate {
        // let dejavu: &[u8] = include_bytes!("DejaVuSansMono.ttf");
        // let glyph_brush = GlyphBrushBuilder::using_font_bytes(dejavu).build();

        TestDelgate {
            // glyph_brush: glyph_brush,
            // data: None,
            last_time: Instant::now(),
            // text_lines: Vec::new(),
            text_console: None,
        }
    }
}

impl RenderDelegate for TestDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        self.text_console = Some(render_bits::text_console::TextConsole::new(render_test));
        Box::new(vulkano::sync::now(render_test.device.clone()))
    }
    fn shutdown(self) {}

    fn create_pipeline(
        &self,
        render_test: &RenderTest,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
        let vs = vs::Shader::load(render_test.device.clone()).unwrap();
        let fs = fs::Shader::load(render_test.device.clone()).unwrap();
        let dimensions = render_test.dimension();
        // In the triangle example we use a dynamic viewport, as its a simple example.
        // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
        // This allows the driver to optimize things, at the cost of slower window resizes.
        // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
        Arc::new(
            GraphicsPipeline::start()
                .vertex_input(TwoBuffersDefinition::<Vertex, TexCoord>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .front_face_clockwise()
                .cull_mode_back()
                .viewports_dynamic_scissors_irrelevant(1)
                .viewports(iter::once(Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }))
                .blend_alpha_blending()
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(Subpass::from(render_test.render_pass.clone(), 0).unwrap())
                .build(render_test.device.clone())
                .unwrap(),
        )
    }

    fn update(&mut self, render_test: &RenderTest, _input_state: &InputState) -> Box<GpuFuture> {
        if let Some(text_console) = &mut self.text_console {
            text_console.add_line(&format!("time: {:?}\n", Instant::now()));
            text_console.update(render_test)
        } else {
            Box::new(vulkano::sync::now(render_test.device.clone()))
        }
    }

    fn frame(
        &mut self,
        render_test: &RenderTest,
        // input_state: &InputState,
        framebuffer: Arc<FramebufferAbstract + Send + Sync>,
        pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> Option<
        Box<
            vulkano::command_buffer::CommandBuffer<
                PoolAlloc = vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc,
            >,
        >,
    > {
        let builder = AutoCommandBufferBuilder::primary_one_time_submit(
            render_test.device.clone(),
            render_test.queue.family(),
        )
        .unwrap();

        Some(Box::new(
            if let Some(text_console) = &mut self.text_console {
                text_console
                    .render(builder, framebuffer, pipeline)
                    .build()
                    .unwrap()
            } else {
                builder.build().unwrap()
            },
        ))
    }
}

fn main() {
    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    let mut delegate = TestDelgate::new();

    render_bits::render_test(&mut delegate);
    delegate.shutdown();
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/font/debug_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/font/debug_frag.glsl"
    }
}
