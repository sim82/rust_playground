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
use std::time::Instant;

use glyph_brush::{rusttype, BrushAction, BrushError, GlyphBrush, GlyphBrushBuilder, Section};

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

struct GlyphBuffers {
    vb: Arc<CpuBufferPoolChunk<Vertex, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    tb: Arc<CpuBufferPoolChunk<TexCoord, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    ib: Arc<CpuBufferPoolChunk<u32, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    glyph_image: Arc<ImmutableImage<vulkano::format::Format>>,
}

struct TestData {
    vb_pool: CpuBufferPool<Vertex>,
    tb_pool: CpuBufferPool<TexCoord>,
    ib_pool: CpuBufferPool<u32>,

    buffers: Option<GlyphBuffers>,

    width: u32,
    height: u32,

    glyph_data: Vec<u8>,

    // vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    // tex_buffer: Arc<ImmutableBuffer<[TexCoord]>>,

    // colors_buffer_gpu: Arc<ImmutableBuffer<[Color]>>,
    // index_buffer: Arc<ImmutableBuffer<[u32]>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,

    sampler: Arc<Sampler>,
}

type GlyphVertexData = (
    rusttype::Point<i32>,
    rusttype::Point<i32>,
    rusttype::Point<f32>,
    rusttype::Point<f32>,
);

struct TestDelgate {
    data: Option<TestData>,
    glyph_brush: GlyphBrush<'static, GlyphVertexData>,
    last_time: std::time::Instant,
}
impl TestDelgate {
    fn new() -> TestDelgate {
        let dejavu: &[u8] = include_bytes!("DejaVuSansMono.ttf");
        let glyph_brush = GlyphBrushBuilder::using_font_bytes(dejavu).build();

        TestDelgate {
            glyph_brush: glyph_brush,
            data: None,
            last_time: Instant::now(),
        }
    }
    fn update_text(
        &mut self,
        render_test: &RenderTest,
        text: &str,
    ) -> Box<vulkano::sync::GpuFuture> {
        if let Some(ref mut test_data) = &mut self.data {
            self.glyph_brush.queue(Section {
                //text: "MMMHello qwertyuiopasdfghjklzxcvbnmQWERTYUIOASDFGHJKLZXCVBNM",
                text: &text,
                scale: glyph_brush::rusttype::Scale::uniform(64f32),
                ..Section::default()
            });

            let dimensions = render_test.dimension();
            let mut verts = Vec::new();
            let mut tex = Vec::new();
            let mut indices = Vec::new();
            loop {
                match self.glyph_brush.process_queued(
                    (dimensions[0], dimensions[1]),
                    |rect, tex_data| {
                        (
                            &mut test_data.glyph_data[..],
                            (test_data.width, test_data.height),
                        )
                            .blit(rect, tex_data);
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

                        break;
                    }
                    Ok(BrushAction::ReDraw) => {
                        // Re-draw last frame's vertices unmodified.
                        println!("redraw");
                        break;
                    }
                    Err(BrushError::TextureTooSmall { suggested }) => {
                        println!("too small {:?}", suggested);
                        // Enlarge texture + glyph_brush texture cache and retry.
                        test_data.width = suggested.0;
                        test_data.height = suggested.1;

                        test_data.glyph_data =
                            vec![0u8; (test_data.width * test_data.height) as usize];
                        self.glyph_brush
                            .resize_texture(test_data.width, test_data.height);
                    }
                }
            }

            let (gi, gi_fut) = ImmutableImage::from_iter(
                test_data.glyph_data.iter().cloned(),
                vulkano::image::Dimensions::Dim2d {
                    width: test_data.width,
                    height: test_data.height,
                },
                vulkano::format::Format::R8Srgb,
                render_test.queue.clone(),
            )
            .unwrap();

            test_data.buffers = Some(GlyphBuffers {
                vb: Arc::new(test_data.vb_pool.chunk(verts.iter().cloned()).unwrap()),
                tb: Arc::new(test_data.tb_pool.chunk(tex.iter().cloned()).unwrap()),
                ib: Arc::new(test_data.ib_pool.chunk(indices.iter().cloned()).unwrap()),

                glyph_image: gi,
            });

            Box::new(gi_fut)
        } else {
            unimplemented!();
        }
    }
}

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

impl RenderDelegate for TestDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        //self.vertex_buffer = vulkano::buffer::CpuAccessibleBuffer::from_data(device: Arc<Device>, usage: BufferUsage, data: T)

        // let dejavu: &[u8] = include_bytes!("DejaVuSansMono.ttf");
        // let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(dejavu).build();

        // let text = String::from_utf8(include_bytes!("test.txt")[..].into()).unwrap();
        // glyph_brush.queue(Section {
        //     //text: "MMMHello qwertyuiopasdfghjklzxcvbnmQWERTYUIOASDFGHJKLZXCVBNM",
        //     text: &text,
        //     scale: glyph_brush::rusttype::Scale::uniform(64f32),
        //     ..Section::default()
        // });
        // // glyph_brush.queue(some_other_section);
        // let dimensions = render_test.dimension();
        // // let mut x = 0;

        // let mut size = (256u32, 256u32);
        // let mut glyph_data = vec![0u8; (size.0 * size.1) as usize];
        // let mut verts = Vec::new();
        // let mut tex = Vec::new();
        // let mut indices = Vec::new();
        // loop {
        //     match glyph_brush.process_queued(
        //         (dimensions[0], dimensions[1]),
        //         |rect, tex_data| {
        //             (&mut glyph_data[..], (size.0, size.1)).blit(rect, tex_data);
        //         },
        //         |vertex_data| {
        //             (
        //                 vertex_data.pixel_coords.min,
        //                 vertex_data.pixel_coords.max,
        //                 vertex_data.tex_coords.min,
        //                 vertex_data.tex_coords.max,
        //             )
        //         },
        //     ) {
        //         Ok(BrushAction::Draw(vertices)) => {
        //             for (vmin, vmax, tmin, tmax) in vertices {
        //                 let v = verts.len() as u32;
        //                 indices.append(&mut vec![v, v + 1, v + 2, v, v + 2, v + 3]);

        //                 verts.push(Vertex {
        //                     position: (vmin.x as f32, vmin.y as f32, 0f32),
        //                 });
        //                 verts.push(Vertex {
        //                     position: (vmax.x as f32, vmin.y as f32, 0f32),
        //                 });
        //                 verts.push(Vertex {
        //                     position: (vmax.x as f32, vmax.y as f32, 0f32),
        //                 });
        //                 verts.push(Vertex {
        //                     position: (vmin.x as f32, vmax.y as f32, 0f32),
        //                 });
        //                 tex.push(TexCoord {
        //                     tex: (tmin.x as f32, tmin.y as f32),
        //                 });
        //                 tex.push(TexCoord {
        //                     tex: (tmax.x as f32, tmin.y as f32),
        //                 });
        //                 tex.push(TexCoord {
        //                     tex: (tmax.x as f32, tmax.y as f32),
        //                 });
        //                 tex.push(TexCoord {
        //                     tex: (tmin.x as f32, tmax.y as f32),
        //                 });
        //             }

        //             break;
        //         }
        //         Ok(BrushAction::ReDraw) => {
        //             // Re-draw last frame's vertices unmodified.
        //             println!("redraw");
        //             break;
        //         }
        //         Err(BrushError::TextureTooSmall { suggested }) => {
        //             println!("too small {:?}", suggested);
        //             // Enlarge texture + glyph_brush texture cache and retry.
        //             size = suggested;
        //             glyph_data = vec![0u8; (size.0 * size.1) as usize];
        //             glyph_brush.resize_texture(size.0, size.1);
        //         }
        //     }
        // }
        // let image =
        //     ImageBuffer::<image::Luma<u8>, _>::from_raw(size.0, size.1, &glyph_data[..]).unwrap();
        // image.save("font.png").unwrap();

        // self.glyph_image = Some(
        //     StorageImage::new(
        //         render_test.device.clone(),
        //         vulkano::image::Dimensions::Dim2d {
        //             width: 256,
        //             height: 256,
        //         },
        //         vulkano::format::R8Uint,
        //         render_test.device.active_queue_families(),
        //     )
        //     .unwrap(),
        // );
        // let (gi, gi_fut) = ImmutableImage::from_iter(
        //     glyph_data.iter().cloned(),
        //     vulkano::image::Dimensions::Dim2d {
        //         width: size.0,
        //         height: size.1,
        //     },
        //     vulkano::format::Format::R8Srgb,
        //     render_test.queue.clone(),
        // )
        // .unwrap();

        // let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
        //     verts.iter().cloned(),
        //     vulkano::buffer::BufferUsage::vertex_buffer(),
        //     render_test.queue.clone(),
        // )
        // .unwrap();
        // let (tb, tb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
        //     tex.iter().cloned(),
        //     vulkano::buffer::BufferUsage::vertex_buffer(),
        //     render_test.queue.clone(),
        // )
        // .unwrap();

        // println!("indices {:?}", indices);
        // // let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
        //     [
        //         Vertex {
        //             position: (0f32, 0f32, 0f32),
        //         },
        //         Vertex {
        //             position: (100f32, 0f32, 0f32),
        //         },
        //         Vertex {
        //             position: (100f32, 100f32, 0f32),
        //         },
        //         Vertex {
        //             position: (0f32, 100f32, 0f32),
        //         },
        //         Vertex {
        //             position: (150f32, 0f32, 0f32),
        //         },
        //         Vertex {
        //             position: (250f32, 0f32, 0f32),
        //         },
        //         Vertex {
        //             position: (250f32, 100f32, 0f32),
        //         },
        //         Vertex {
        //             position: (150f32, 100f32, 0f32),
        //         },
        //     ]
        //     .iter()
        //     .cloned(),
        //     vulkano::buffer::BufferUsage::vertex_buffer(),
        //     render_test.queue.clone(),
        // )
        // .unwrap();

        // hollow half cube
        // let (ib, ib_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
        //     indices.iter().cloned(),
        //     vulkano::buffer::BufferUsage::index_buffer(),
        //     render_test.queue.clone(),
        // )
        // .unwrap();

        let (cb, cb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                Color {
                    color: (0f32, 0f32, 0f32), // 0
                },
                Color {
                    color: (1f32, 0f32, 0f32), // 1
                },
                Color {
                    color: (1f32, 1f32, 0f32), // 2
                },
                Color {
                    color: (0f32, 1f32, 0f32), // 3
                },
                Color {
                    color: (0f32, 0f32, 1f32), // 4
                },
                Color {
                    color: (1f32, 0f32, 0f32), // 5
                },
                Color {
                    color: (1f32, 1f32, 0f32), // 6
                },
                Color {
                    color: (0f32, 1f32, 0f32), // 7
                },
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            render_test.queue.clone(),
        )
        .unwrap();

        self.data = Some(TestData {
            vb_pool: CpuBufferPool::<Vertex>::new(render_test.device.clone(), BufferUsage::all()),
            tb_pool: CpuBufferPool::<TexCoord>::new(render_test.device.clone(), BufferUsage::all()),
            ib_pool: CpuBufferPool::<u32>::new(render_test.device.clone(), BufferUsage::all()),

            buffers: None,

            width: 256,
            height: 256,
            glyph_data: vec![0u8; 256 * 256],

            // vertex_buffer: vb,
            // tex_buffer: tb,
            // colors_buffer_gpu: cb,
            // index_buffer: ib,
            uniform_buffer: CpuBufferPool::<vs::ty::Data>::new(
                render_test.device.clone(),
                BufferUsage::all(),
            ),
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
        });

        self.update_text(render_test, "bla bla bla")

        //Box::new(vb_fut.join(ib_fut.join(cb_fut.join(gi_fut.join(tb_fut)))))
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
        // let now = Instant::now();
        // let d_time = now - self.last_time;
        self.last_time = Instant::now();

        Box::new(vulkano::sync::now(render_test.device.clone()))
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
        if let Some(data) = &self.data {
            if let Some(buffers) = &data.buffers {
                let uniform_buffer_subbuffer = {
                    let dimensions = render_test.dimension();
                    let unity = Matrix4::from_scale(1f32);
                    let uniform_data = vs::ty::Data {
                        view: unity.into(),
                        world: unity.into(),
                        proj: render_bits::math::canvas_projection(dimensions[0], dimensions[1])
                            .into(),
                    };

                    data.uniform_buffer.next(uniform_data).unwrap()
                };

                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(uniform_buffer_subbuffer)
                        .unwrap()
                        .add_sampled_image(buffers.glyph_image.clone(), data.sampler.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
                    render_test.device.clone(),
                    render_test.queue.family(),
                )
                .unwrap()
                // .update_buffer(colors_buffer_gpu.clone(), self.colors_cpu[..]).unwrap()
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
                .build()
                .unwrap();
                Some(Box::new(command_buffer))
            } else {
                Some(Box::new(
                    AutoCommandBufferBuilder::primary_one_time_submit(
                        render_test.device.clone(),
                        render_test.queue.family(),
                    )
                    .unwrap()
                    .build()
                    .unwrap(),
                ))
            }
        } else {
            println!("not initialized");
            //None

            Some(Box::new(
                AutoCommandBufferBuilder::primary_one_time_submit(
                    render_test.device.clone(),
                    render_test.queue.family(),
                )
                .unwrap()
                .build()
                .unwrap(),
            ))
        }
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
