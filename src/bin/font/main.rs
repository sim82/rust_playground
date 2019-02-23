use rust_playground::render_bits;

use crate::render_bits::InputState;
use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;

use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::image::StorageImage;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, Vector3};

use render_bits::Vertex;
use std::iter;
use std::sync::Arc;
use std::time::Instant;

use glyph_brush::{BrushAction, BrushError, GlyphBrushBuilder, Section};

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

struct TestDelgate {
    vertex_buffer: Option<Arc<ImmutableBuffer<[Vertex]>>>,
    colors_buffer_gpu: Option<Arc<ImmutableBuffer<[Color]>>>,
    index_buffer: Option<Arc<ImmutableBuffer<[u32]>>>,
    uniform_buffer: Option<CpuBufferPool<vs::ty::Data>>,
    glyph_image: Option<Arc<StorageImage<vulkano::format::R8Uint>>>,
    last_time: std::time::Instant,
}
impl TestDelgate {
    fn new() -> TestDelgate {
        TestDelgate {
            vertex_buffer: None,
            colors_buffer_gpu: None,
            index_buffer: None,
            uniform_buffer: None,
            glyph_image: None,
            last_time: Instant::now(),
        }
    }
}

impl RenderDelegate for TestDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        //self.vertex_buffer = vulkano::buffer::CpuAccessibleBuffer::from_data(device: Arc<Device>, usage: BufferUsage, data: T)

        let dejavu: &[u8] = include_bytes!("DejaVuSans.ttf");
        let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(dejavu).build::<u32>();

        glyph_brush.queue(Section {
            text: "MMMHello glyph_brush",
            scale: glyph_brush::rusttype::Scale::uniform(64f32),
            ..Section::default()
        });
        // glyph_brush.queue(some_other_section);
        let dimensions = render_test.dimension();
        // let mut x = 0;

        self.glyph_image = Some(
            StorageImage::new(
                render_test.device.clone(),
                vulkano::image::Dimensions::Dim2d {
                    width: 256,
                    height: 256,
                },
                vulkano::format::R8Uint,
                render_test.device.active_queue_families(),
            )
            .unwrap(),
        );

        let verts = std::cell::RefCell::new(Vec::new());
        let mut indices = Vec::new();
        match glyph_brush.process_queued(
            (dimensions[0], dimensions[1]),
            |rect, tex_data| {
                if let Some(image) = &self.glyph_image {
                    // image.
                    println!(
                        "blit: {:?}",
                        (image as &vulkano::image::ImageAccess).supports_blit_destination()
                    );
                }
            },
            |vertex_data| {
                let minx = vertex_data.pixel_coords.min.x as f32;
                let miny = vertex_data.pixel_coords.min.y as f32;
                let maxx = vertex_data.pixel_coords.max.x as f32;
                let maxy = vertex_data.pixel_coords.max.y as f32;

                println!("to_vertex: {:?}", vertex_data);
                // x += 1;
                let mut verts = verts.borrow_mut();
                let first = verts.len();
                verts.push(Vertex {
                    position: (minx, miny, 0f32),
                });
                verts.push(Vertex {
                    position: (maxx, miny, 0f32),
                });
                verts.push(Vertex {
                    position: (maxx, maxy, 0f32),
                });
                verts.push(Vertex {
                    position: (minx, maxy, 0f32),
                });

                first as u32
            },
        ) {
            Ok(BrushAction::Draw(vertices)) => {
                // Draw new vertices.
                //println!("draw {:?}", vertices);

                indices.append(
                    &mut vertices
                        .iter()
                        .flat_map(|v| vec![*v, *v + 1, *v + 2, *v, *v + 2, *v + 3])
                        .collect(),
                )
            }
            Ok(BrushAction::ReDraw) => {
                // Re-draw last frame's vertices unmodified.
                println!("redraw");
            }
            Err(BrushError::TextureTooSmall { suggested }) => {
                println!("too small {:?}", suggested);
                // Enlarge texture + glyph_brush texture cache and retry.
            }
        }
        let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            verts.borrow().iter().cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            render_test.queue.clone(),
        )
        .unwrap();
        println!("indices {:?}", indices);
        // let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
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
        self.vertex_buffer = Some(vb);

        // hollow half cube
        let (ib, ib_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            vulkano::buffer::BufferUsage::index_buffer(),
            render_test.queue.clone(),
        )
        .unwrap();
        self.index_buffer = Some(ib);

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
        self.colors_buffer_gpu = Some(cb);

        self.uniform_buffer = Some(CpuBufferPool::<vs::ty::Data>::new(
            render_test.device.clone(),
            BufferUsage::all(),
        ));

        Box::new(vb_fut.join(ib_fut.join(cb_fut)))
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
                .vertex_input(TwoBuffersDefinition::<Vertex, Color>::new())
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
        match (
            &self.vertex_buffer,
            // &self.colors_buffer,
            &self.colors_buffer_gpu,
            &self.index_buffer,
            &self.uniform_buffer,
        ) {
            (
                Some(vertex_buffer),
                // Some(colors_buffer),
                Some(colors_buffer_gpu),
                Some(index_buffer),
                Some(uniform_buffer),
            ) => {
                let uniform_buffer_subbuffer = {
                    let dimensions = render_test.dimension();
                    let unity = Matrix4::from_scale(1f32);
                    let uniform_data = vs::ty::Data {
                        view: unity.into(),
                        world: unity.into(),
                        proj: render_bits::math::canvas_projection(dimensions[0], dimensions[1])
                            .into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(uniform_buffer_subbuffer)
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
                    vec![vertex_buffer.clone(), colors_buffer_gpu.clone()],
                    index_buffer.clone(),
                    set.clone(),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();
                Some(Box::new(command_buffer))
            }

            _ => {
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
