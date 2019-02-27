use crate::render_bits;
use crate::render_bits::{RenderTest, TexCoord, Vertex};
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
use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glyph_brush::{rusttype, BrushAction, BrushError, GlyphBrush, GlyphBrushBuilder, Section};

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);
trait Canvas {
    fn blit(&mut self, rect: glyph_brush::rusttype::Rect<u32>, data: &[u8]);
}

impl Canvas for (&mut [u8], (u32, u32)) {
    fn blit(&mut self, rect: glyph_brush::rusttype::Rect<u32>, data: &[u8]) {
        let (mydata, (w, h)) = self;

        if rect.max.x > *w || rect.max.y > *h {
            return;
        }
        for y in 0..rect.height() {
            let ydest = y + rect.min.y;
            let dest = (ydest * *w + rect.min.x) as usize;
            let destx = dest + rect.width() as usize;
            let src = (y * rect.width()) as usize;
            let srcx = src + rect.width() as usize;

            mydata[dest..destx].copy_from_slice(&data[src..srcx]);
        }
    }
}

type GlyphVertexData = (
    rusttype::Point<i32>,
    rusttype::Point<i32>,
    rusttype::Point<f32>,
    rusttype::Point<f32>,
);

struct TextConsoleBuffers {
    vb: Arc<CpuBufferPoolChunk<Vertex, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    tb: Arc<CpuBufferPoolChunk<TexCoord, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    ib: Arc<CpuBufferPoolChunk<u32, Arc<vulkano::memory::pool::StdMemoryPool>>>,
}

pub struct TextConsole {
    vb_pool: CpuBufferPool<Vertex>,
    tb_pool: CpuBufferPool<TexCoord>,
    ib_pool: CpuBufferPool<u32>,
    ub_pool: CpuBufferPool<vs::ty::Data>,
    buffers: Option<TextConsoleBuffers>,
    glyph_image: Option<Arc<ImmutableImage<vulkano::format::Format>>>,
    glyph_data: Vec<u8>,
    brush: GlyphBrush<'static, GlyphVertexData>,
    sampler: Arc<vulkano::sampler::Sampler>,

    text_lines: Vec<String>,
}

impl TextConsole {
    pub fn new(render_test: &RenderTest) -> Self {
        TextConsole {
            vb_pool: CpuBufferPool::<Vertex>::new(render_test.device.clone(), BufferUsage::all()),
            tb_pool: CpuBufferPool::<TexCoord>::new(render_test.device.clone(), BufferUsage::all()),
            ib_pool: CpuBufferPool::<u32>::new(render_test.device.clone(), BufferUsage::all()),
            ub_pool: CpuBufferPool::<vs::ty::Data>::new(
                render_test.device.clone(),
                BufferUsage::all(),
            ),
            buffers: None,
            glyph_image: None,
            glyph_data: vec![0u8; 256 * 256],
            brush: GlyphBrushBuilder::using_font_bytes(&include_bytes!("DejaVuSansMono.ttf")[..])
                .build(),
            text_lines: Vec::new(),
            // glyph_image: gi,
            sampler: Sampler::new(
                render_test.device.clone(),
                Filter::Linear,
                Filter::Linear,
                MipmapMode::Nearest,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .unwrap(),
        }
    }

    pub fn add_line(&mut self, line: &str) {
        self.text_lines.push(line.into());

        while self.text_lines.len() > 20 {
            self.text_lines.remove(0);
        }
    }

    pub fn update(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        let font_size = (16f64 * render_test.surface.window().get_hidpi_factor()) as f32;
        for (i, line) in self.text_lines.iter().enumerate() {
            self.brush.queue(Section {
                //text: "MMMHello qwertyuiopasdfghjklzxcvbnmQWERTYUIOASDFGHJKLZXCVBNM",
                text: line,
                scale: glyph_brush::rusttype::Scale::uniform(font_size),
                screen_position: (0f32, font_size * i as f32),
                ..Section::default()
            });

            // println!("queue: {}", line);
        }
        // println!("done");
        let dimensions = render_test.dimension();
        let mut verts = Vec::new();
        let mut tex = Vec::new();
        let mut indices = Vec::new();
        let mut glyph_updated = false;

        loop {
            let size = self.brush.texture_dimensions();
            let glyph_data = &mut self.glyph_data;

            match self.brush.process_queued(
                (dimensions[0], dimensions[1]),
                |rect, tex_data| {
                    (&mut glyph_data[..], size).blit(rect, tex_data);
                    glyph_updated = true;
                    println!("blit");
                },
                |vertex_data| {
                    (
                        vertex_data.pixel_coords.min,
                        vertex_data.pixel_coords.max,
                        vertex_data.tex_coords.min,
                        vertex_data.tex_coords.max,
                    )
                },
            ) {
                Ok(BrushAction::Draw(vertices)) => {
                    for (vmin, vmax, tmin, tmax) in vertices {
                        let v = verts.len() as u32;
                        indices.append(&mut vec![v, v + 1, v + 2, v, v + 2, v + 3]);

                        verts.push(Vertex {
                            position: (vmin.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmax.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmin.x as f32, vmax.y as f32, 0f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmax.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmax.y as f32),
                        });
                    }

                    self.buffers = Some(TextConsoleBuffers {
                        vb: Arc::new(self.vb_pool.chunk(verts.iter().cloned()).unwrap()),
                        tb: Arc::new(self.tb_pool.chunk(tex.iter().cloned()).unwrap()),
                        ib: Arc::new(self.ib_pool.chunk(indices.iter().cloned()).unwrap()),
                    });
                    // println!("draw");

                    break;
                }
                Ok(BrushAction::ReDraw) => {
                    // Re-draw last frame's vertices unmodified.
                    // println!("redraw");
                    break;
                }
                Err(BrushError::TextureTooSmall { suggested }) => {
                    println!("too small {:?}", suggested);
                    // Enlarge texture + glyph_brush texture cache and retry.

                    self.glyph_data = vec![0u8; (suggested.0 * suggested.1) as usize];
                    self.brush.resize_texture(suggested.0, suggested.1);
                }
            }
        }

        if glyph_updated {
            let (width, height) = self.brush.texture_dimensions();

            let (gi, gi_fut) = ImmutableImage::from_iter(
                self.glyph_data.iter().cloned(),
                vulkano::image::Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                vulkano::format::Format::R8Srgb,
                render_test.queue.clone(),
            )
            .unwrap();

            // let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(
            //     test_data.width,
            //     test_data.height,
            //     &test_data.glyph_data[..],
            // )
            // .unwrap();
            // image.save("font.png").unwrap();
            self.glyph_image = Some(gi);
            Box::new(gi_fut)
        } else {
            Box::new(vulkano::sync::now(render_test.device.clone()))
        }
    }

    pub fn render<PoolBuilder>(
        &mut self,
        builder: AutoCommandBufferBuilder<PoolBuilder>,
        framebuffer: Arc<FramebufferAbstract + Send + Sync>,
        pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> AutoCommandBufferBuilder<PoolBuilder> {
        if let Some(buffers) = &self.buffers {
            let uniform_buffer_subbuffer = {
                let width = framebuffer.width();
                let height = framebuffer.height();

                let unity = Matrix4::from_scale(1f32);
                let uniform_data = vs::ty::Data {
                    view: unity.into(),
                    world: unity.into(),
                    proj: render_bits::math::canvas_projection(width, height).into(),
                };

                self.ub_pool.next(uniform_data).unwrap()
            };

            let set = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 0)
                    .add_buffer(uniform_buffer_subbuffer)
                    .unwrap()
                    .add_sampled_image(
                        self.glyph_image.as_mut().unwrap().clone(),
                        self.sampler.clone(),
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
            );

            builder
                .begin_render_pass(
                    framebuffer.clone(),
                    false,
                    vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .draw_indexed(
                    pipeline.clone(),
                    &DynamicState::none(),
                    vec![buffers.vb.clone(), buffers.tb.clone()],
                    buffers.ib.clone(),
                    set.clone(),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
        } else {
            builder
        }
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_bits/text_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_bits/text_frag.glsl"
    }
}