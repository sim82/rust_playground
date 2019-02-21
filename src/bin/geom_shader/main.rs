use rust_playground::render_bits;

use crate::render_bits::InputState;
use crate::render_bits::PlayerFlyModel;
use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;

use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, Vector3};

use rand::prelude::*;
use render_bits::{Normal, Vertex};
use std::iter;
use std::sync::{
    mpsc::{channel, sync_channel, Receiver, Sender},
    Arc,
};
use std::thread::spawn;
use std::thread::JoinHandle;
use std::time::Instant;

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}
type Point3 = cgmath::Point3<f32>;

vulkano::impl_vertex!(Color, color);

struct TestDelgate {
    player_model: PlayerFlyModel,
    vertex_buffer: Option<Arc<ImmutableBuffer<[Vertex]>>>,
    colors_buffer_gpu: Option<Arc<ImmutableBuffer<[Color]>>>,
    index_buffer: Option<Arc<ImmutableBuffer<[u32]>>>,
    uniform_buffer: Option<CpuBufferPool<vs::ty::Data>>,
    last_time: std::time::Instant,
}
impl TestDelgate {
    fn new() -> TestDelgate {
        TestDelgate {
            player_model: PlayerFlyModel::new(
                Point3::new(17f32, 14f32, 27f32),
                cgmath::Deg(13f32),
                cgmath::Deg(-22f32),
            ),
            vertex_buffer: None,
            colors_buffer_gpu: None,
            // colors_buffer_pool: None,
            // colors_buffer: None,
            index_buffer: None,
            uniform_buffer: None,
            // colors_cpu: Vec::new(),
            last_time: Instant::now(),
            // scene: None,
        }
    }
}

impl RenderDelegate for TestDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        //self.vertex_buffer = vulkano::buffer::CpuAccessibleBuffer::from_data(device: Arc<Device>, usage: BufferUsage, data: T)

        let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                Vertex {
                    position: (0f32, 0f32, 0f32),
                },
                Vertex {
                    position: (1f32, 0f32, 0f32),
                },
                Vertex {
                    position: (1f32, 1f32, 0f32),
                },
                Vertex {
                    position: (0f32, 1f32, 0f32),
                },
                Vertex {
                    position: (0f32, 1f32, 1f32),
                },
                Vertex {
                    position: (0f32, 0f32, 1f32),
                },
                Vertex {
                    position: (1f32, 0f32, 1f32),
                },
                Vertex {
                    position: (1f32, 1f32, 1f32),
                },
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            render_test.queue.clone(),
        )
        .unwrap();
        self.vertex_buffer = Some(vb);

        let (ib, ib_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1, 7, 6, 5, 7, 5, 4, 7, 4, 3, 7,
                3, 2, 7, 2, 1, 7, 1, 6,
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::index_buffer(),
            render_test.queue.clone(),
        )
        .unwrap();
        self.index_buffer = Some(ib);

        let (cb, cb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                Color {
                    color: (0f32, 0f32, 0f32),
                },
                Color {
                    color: (1f32, 0f32, 0f32),
                },
                Color {
                    color: (1f32, 1f32, 0f32),
                },
                Color {
                    color: (0f32, 1f32, 0f32),
                },
                Color {
                    color: (0f32, 1f32, 1f32),
                },
                Color {
                    color: (0f32, 0f32, 1f32),
                },
                Color {
                    color: (1f32, 0f32, 1f32),
                },
                Color {
                    color: (1f32, 1f32, 1f32),
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

        Box::new(vb_fut.join(ib_fut))
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

    fn update(&mut self, render_test: &RenderTest, input_state: &InputState) -> Box<GpuFuture> {
        // let now = Instant::now();
        // let d_time = now - self.last_time;
        self.last_time = Instant::now();

        self.player_model.apply_delta_lon(input_state.d_lon);
        self.player_model.apply_delta_lat(input_state.d_lat);

        const FORWARD_VEL: f32 = 1.0 / 60.0 * 2.0;
        let boost = if input_state.run { 3.0 } else { 1.0 };
        if input_state.forward {
            self.player_model.apply_move_forward(FORWARD_VEL * boost);
        }
        if input_state.backward {
            self.player_model.apply_move_forward(-FORWARD_VEL * boost);
        }
        if input_state.left {
            self.player_model.apply_move_right(-FORWARD_VEL * boost);
        }
        if input_state.right {
            self.player_model.apply_move_right(FORWARD_VEL * boost);
        }

        // println!("{:?}", self.player_model);

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
                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    // let proj = cgmath::perspective(
                    //     Rad(std::f32::consts::FRAC_PI_2),
                    //     aspect_ratio,
                    //     0.01,
                    //     100.0,
                    // ) * Matrix4::from_nonuniform_scale(1f32, 1f32, -1f32);
                    let proj =
                        perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0)
                            * Matrix4::from_nonuniform_scale(1f32, 1f32, 1f32);

                    println!("{:?}", self.player_model);
                    let unity = Matrix4::from_scale(1f32);
                    let transform = Matrix4::from_translation(Vector3::new(0f32, 0f32, -5f32));
                    let uniform_data = vs::ty::Data {
                        //world: <Matrix4<f32> as Transform<Point3>>::one().into(), // from(rotation).into(),
                        view: self.player_model.get_view_matrix().into(), //(view * scale).into(),
                        world: unity.into(),
                        // view: transform.into(),
                        proj: proj.into(),
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

fn perspective(fov: Rad<f32>, a: f32, zn: f32, zf: f32) -> Matrix4<f32> {
    #[rustfmt::skip]
    let clip = Matrix4::new(
        1.0f32,  0.0f32, 0.0f32, 0.0f32, 
        0.0f32, -1.0f32, 0.0f32, 0.0f32, 
        0.0f32,  0.0f32, 0.5f32, 0.0f32, 
        0.0f32,  0.0f32, 0.5f32, 1.0f32,
    );

    let f = Rad::cot(fov / 2f32);

    #[rustfmt::skip]
    let proj = Matrix4::new(
        f / a,    0f32,    0f32,                          0f32,
        0f32,     -f,       0f32,                          0f32, 
        0f32,     0f32,    zf / (zn - zf),         -1f32, 
        0f32,     0f32,    (zn * zf) / (zn - zf),  0f32,
    );
    // let proj = cgmath::perspective(fov, aspect_ratio, near_plane, far_plane);
    // clip * proj
    proj
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
        path: "src/bin/geom_shader/debug_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/geom_shader/debug_frag.glsl"
    }
}
