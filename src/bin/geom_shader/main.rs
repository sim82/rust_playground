use rust_playground::{render_bits, script};

use render_bits::{
    InputStateEventDispatcher, PlayerFlyModel, RenderDelegate, RenderTest, VulcanoState,
};
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::{Matrix4, Rad};

use render_bits::Vertex;
use std::iter;
use std::sync::Arc;

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
    pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>>,
    last_time: std::time::Instant,

    input_state: InputStateEventDispatcher,
}
impl TestDelgate {
    fn new(vk_state: &VulcanoState, _script_env: &mut script::Environment) -> TestDelgate {
        let (vb, vb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                Vertex {
                    position: (0f32, 0f32, 0f32), // 0
                },
                Vertex {
                    position: (1f32, 0f32, 0f32), // 1
                },
                Vertex {
                    position: (0f32, 1f32, 0f32), // 2
                },
                Vertex {
                    position: (1f32, 1f32, 0f32), // 3
                },
                Vertex {
                    position: (0f32, 0f32, 1f32), // 4
                },
                Vertex {
                    position: (1f32, 0f32, 1f32), // 5
                },
                Vertex {
                    position: (0f32, 1f32, 1f32), // 6
                },
                Vertex {
                    position: (1f32, 1f32, 1f32), // 7
                },
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            vk_state.queue(),
        )
        .unwrap();
        // self.vertex_buffer = Some(vb);

        // hollow half cube
        let (ib, ib_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                /*xy*/ 0, 2, 3, 0, 3, 1, /*yz*/ 0, 4, 6, 0, 6, 2, /*xz*/ 0, 1, 5, 0,
                5, 4,
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::index_buffer(),
            vk_state.queue(),
        )
        .unwrap();
        // self.index_buffer = Some(ib);

        let (cb, cb_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            [
                Color {
                    color: (0f32, 0f32, 0f32), // 0
                },
                Color {
                    color: (1f32, 0f32, 0f32), // 1
                },
                Color {
                    color: (0f32, 1f32, 0f32), // 2
                },
                Color {
                    color: (1f32, 1f32, 0f32), // 3
                },
                Color {
                    color: (0f32, 0f32, 1f32), // 4
                },
                Color {
                    color: (1f32, 0f32, 1f32), // 5
                },
                Color {
                    color: (0f32, 1f32, 1f32), // 6
                },
                Color {
                    color: (1f32, 1f32, 1f32), // 7
                },
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            vk_state.queue(),
        )
        .unwrap();
        // self.colors_buffer_gpu = Some(cb);

        // render_test.set_player_input_sink(self.input_state.sink.clone());

        vb_fut
            .join(ib_fut)
            .join(cb_fut)
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        TestDelgate {
            player_model: PlayerFlyModel::new(
                Point3::new(17f32, 14f32, 27f32),
                cgmath::Deg(13f32),
                cgmath::Deg(-22f32),
            ),
            vertex_buffer: Some(vb),
            colors_buffer_gpu: Some(cb),
            index_buffer: Some(ib),
            uniform_buffer: Some(CpuBufferPool::<vs::ty::Data>::new(
                vk_state.device(),
                BufferUsage::all(),
            )),
            pipeline: None,
            last_time: Instant::now(),
            input_state: InputStateEventDispatcher::new(),
        }
    }
}

impl RenderDelegate for TestDelgate {
    fn shutdown(self) {}

    fn get_input_sink(&self) -> std::sync::mpsc::Sender<render_bits::InputEvent> {
        self.input_state.sink.clone()
    }
    fn framebuffer_changed(&mut self, vk_state: &VulcanoState) {
        let vs = vs::Shader::load(vk_state.device()).unwrap();
        let fs = fs::Shader::load(vk_state.device()).unwrap();
        let dimensions = vk_state.dimension();
        // In the triangle example we use a dynamic viewport, as its a simple example.
        // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
        // This allows the driver to optimize things, at the cost of slower window resizes.
        // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
        self.pipeline = Some(Arc::new(
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
                .blend_alpha_blending()
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(vk_state.main_subpass())
                .build(vk_state.device())
                .unwrap(),
        ));
    }

    fn update(
        &mut self,
        vk_state: &VulcanoState,
        _env: &mut script::Environment,
    ) -> Box<GpuFuture> {
        // let now = Instant::now();
        // let d_time = now - self.last_time;
        self.last_time = Instant::now();

        self.input_state.update();
        let input_state = &mut self.input_state.input_state;

        self.player_model.apply_delta_lon(input_state.delta_lon());
        self.player_model.apply_delta_lat(input_state.delta_lat());

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

        Box::new(vulkano::sync::now(vk_state.device()))
    }

    fn render_frame(
        &mut self,
        vk_state: &VulcanoState,
        builder: AutoCommandBufferBuilder,
    ) -> Result<AutoCommandBufferBuilder, render_bits::Error> {
        match (
            &self.vertex_buffer,
            &self.colors_buffer_gpu,
            &self.index_buffer,
            &self.uniform_buffer,
            &self.pipeline,
        ) {
            (
                Some(vertex_buffer),
                Some(colors_buffer_gpu),
                Some(index_buffer),
                Some(uniform_buffer),
                Some(pipeline),
            ) => {
                let uniform_buffer_subbuffer = {
                    let dimensions = vk_state.dimension();
                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let unity = Matrix4::from_scale(1f32);
                    let uniform_data = vs::ty::Data {
                        //world: <Matrix4<f32> as Transform<Point3>>::one().into(), // from(rotation).into(),
                        view: self.player_model.get_view_matrix().into(), //(view * scale).into(),
                        world: unity.into(),
                        // view: transform.into(),
                        proj: render_bits::math::perspective_projection(
                            Rad(std::f32::consts::FRAC_PI_2),
                            aspect_ratio,
                            0.01,
                            100.0,
                        )
                        // proj: render_bits::math::orthograpic_projection(
                        //     -10f32, 10f32, -10f32, 10f32, 0.01f32, 100.0f32,
                        // )
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

                Ok(builder
                    .draw_indexed(
                        pipeline.clone(),
                        &DynamicState::none(),
                        vec![vertex_buffer.clone(), colors_buffer_gpu.clone()],
                        index_buffer.clone(),
                        set.clone(),
                        (),
                    )
                    .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                        error: "draw indexed failed for crystal planes".into(),
                    })?)
            }

            _ => {
                println!("not initialized");
                Ok(builder)
            }
        }
    }
}

fn main() {
    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    let mut render_test = render_bits::RenderTest::new(false).unwrap();
    let mut delegate = TestDelgate::new(&render_test.vk_state(), render_test.script_env());
    render_test.main_loop(&mut delegate).unwrap();
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
