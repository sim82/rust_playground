extern crate cgmath;
extern crate crystal;
extern crate render_bits;
extern crate vulkano;

use render_bits::InputState;
use render_bits::PlayerFlyModel;
use render_bits::RenderDelegate;
use render_bits::RenderTest;

use crystal::Planes;
use std::cell::RefCell;
use std::iter;
use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Point3, Rad};

use render_bits::{Normal, Vertex};

struct CrystalRenderDelgate {
    player_model: PlayerFlyModel,
    vertex_buffer: Option<Arc<ImmutableBuffer<[Vertex]>>>,
    normals_buffer: Option<Arc<ImmutableBuffer<[Normal]>>>,
    index_buffer: Option<Arc<ImmutableBuffer<[u16]>>>,
    uniform_buffer: Option<CpuBufferPool<render_bits::vs::ty::Data>>,
}

impl CrystalRenderDelgate {
    fn new() -> Arc<RefCell<CrystalRenderDelgate>> {
        Arc::new(RefCell::new(CrystalRenderDelgate {
            player_model: PlayerFlyModel::new(),
            vertex_buffer: None,
            normals_buffer: None,
            index_buffer: None,
            uniform_buffer: None,
        }))
    }
}

impl RenderDelegate for CrystalRenderDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");
        let mut planes = crystal::PlanesSep::new();
        planes.create_planes(&bm);
        planes.print();

        let mut x: Vec<(&crystal::Point3i, i32)> = planes.vertex_iter().collect();
        x.sort_by_key(|(_, v)| *v);

        let vertices: Vec<_> = x
            .iter()
            .map(|(plane, _)| Vertex {
                position: (plane.x as f32, plane.y as f32, plane.z as f32),
            })
            .collect();
        let normals: Vec<_> = planes
            .dir_iter()
            .map(|dir| Normal::from(dir.get_normal::<f32>()))
            .collect();
        assert!(vertices.len() == normals.len());

        let indices: Vec<_> = planes
            .planes_iter()
            .flat_map(|plane| {
                [plane[0], plane[1], plane[2], plane[0], plane[2], plane[3]]
                    .iter()
                    .map(|y| *y as u16)
                    .collect::<Vec<_>>()
            })
            .collect();
        let (vertex_buffer, vb_future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            BufferUsage::all(),
            render_test.queue.clone(),
        )
        .unwrap();
        // let (vertex_buffer, vb_future) =
        //     ImmutableBuffer::from_iter(vx.iter().cloned(), BufferUsage::all(), queue.clone()).unwrap();

        let normals = normals.iter().cloned();
        let (normals_buffer, nb_future) =
            ImmutableBuffer::from_iter(normals, BufferUsage::all(), render_test.queue.clone())
                .unwrap();

        let indices = indices.iter().cloned();
        let (index_buffer, ib_future) =
            ImmutableBuffer::from_iter(indices, BufferUsage::all(), render_test.queue.clone())
                .unwrap();

        let uniform_buffer = CpuBufferPool::<render_bits::vs::ty::Data>::new(
            render_test.device.clone(),
            BufferUsage::all(),
        );
        self.vertex_buffer = Some(vertex_buffer);
        self.normals_buffer = Some(normals_buffer);
        self.index_buffer = Some(index_buffer);
        self.uniform_buffer = Some(uniform_buffer);

        Box::new(vb_future.join(nb_future.join(ib_future)))
    }

    fn create_pipeline(
        &self,
        render_test: &RenderTest,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
        let vs = render_bits::vs::Shader::load(render_test.device.clone()).unwrap();
        let fs = render_bits::fs::Shader::load(render_test.device.clone()).unwrap();
        let dimensions = render_test.dimension();
        // In the triangle example we use a dynamic viewport, as its a simple example.
        // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
        // This allows the driver to optimize things, at the cost of slower window resizes.
        // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
        Arc::new(
            GraphicsPipeline::start()
                .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
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

    fn frame(
        &mut self,
        render_test: &RenderTest,
        input_state: &InputState,
        framebuffer: Arc<FramebufferAbstract + Send + Sync>,
        pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> Box<
        vulkano::command_buffer::CommandBuffer<
            PoolAlloc = vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc,
        >,
    > {
        match (
            &self.vertex_buffer,
            &self.normals_buffer,
            &self.index_buffer,
            &self.uniform_buffer,
        ) {
            (
                Some(vertex_buffer),
                Some(normals_buffer),
                Some(index_buffer),
                Some(uniform_buffer),
            ) => {
                let uniform_buffer_subbuffer = {
                    let dimensions = render_test.dimension();
                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    ) * Matrix4::from_nonuniform_scale(1f32, 1f32, -1f32);

                    self.player_model.apply_delta_lon(input_state.d_lon);
                    self.player_model.apply_delta_lat(input_state.d_lat);

                    const FORWARD_VEL: f32 = 1.0 / 60.0 * 2.0;
                    if input_state.forward {
                        self.player_model.apply_move_forward(FORWARD_VEL);
                    }
                    if input_state.backward {
                        self.player_model.apply_move_forward(-FORWARD_VEL);
                    }
                    if input_state.left {
                        self.player_model.apply_move_right(-FORWARD_VEL);
                    }
                    if input_state.right {
                        self.player_model.apply_move_right(FORWARD_VEL);
                    }

                    println!("{:?}", self.player_model);

                    let uniform_data = render_bits::vs::ty::Data {
                        world: <Matrix4<f32> as Transform<Point3<f32>>>::one().into(), // from(rotation).into(),
                        view: self.player_model.get_view_matrix().into(), //(view * scale).into(),
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
                .begin_render_pass(
                    framebuffer.clone(),
                    false,
                    vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .draw_indexed(
                    pipeline.clone(),
                    &DynamicState::none(),
                    vec![vertex_buffer.clone(), normals_buffer.clone()],
                    index_buffer.clone(),
                    set.clone(),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();
                Box::new(command_buffer)
            }

            _ => panic!("not initialized"),
        }
    }
}

fn main() {
    let delegate = CrystalRenderDelgate::new();

    render_bits::render_test(delegate);
}
