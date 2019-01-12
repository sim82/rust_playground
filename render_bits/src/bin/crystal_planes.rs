extern crate crystal;
extern crate render_bits;
extern crate vulkano;
extern crate cgmath;

use render_bits::RenderDelegate;
use render_bits::RenderTest;

use std::sync::Arc;
use std::cell::RefCell;
use std::iter;
use crystal::Planes;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, CpuBufferPool};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::impl_vertex;
use vulkano::pipeline::viewport::Viewport;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::sync;
use vulkano::sync::GpuFuture;


use cgmath::prelude::*;
use cgmath::{Deg, Matrix3, Matrix4, Point3, Rad, Vector3, Vector4};

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: (f32, f32, f32),
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Copy, Clone)]
pub struct Normal {
    pub normal: (f32, f32, f32),
}

impl_vertex!(Normal, normal);


struct CrystalRenderDelgate {}

impl CrystalRenderDelgate {
    fn new() -> Arc<RefCell<CrystalRenderDelgate>> {
        Arc::new(CrystalRenderDelgate {})
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
            .map(|(plane, _)| render_bits::Vertex {
                position: (plane.x as f32, plane.y as f32, plane.z as f32),
            })
            .collect();
        let normals: Vec<_> = planes
            .dir_iter()
            .map(|dir| render_bits::Normal::from(dir.get_normal::<f32>()))
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

        let uniform_buffer =
            CpuBufferPool::<render_bits::vs::ty::Data>::new(render_test.device.clone(), BufferUsage::all());
    
        vb_future
                .join(nb_future.join(ib_future))
                .unwrap()
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


    fn frame(&mut self, render_test : &RenderTest) -> Box<vulkano::command_buffer::CommandBuffer<PoolAlloc = vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder>> {

        let uniform_buffer_subbuffer = {
            let dimensions = render_test.dimensions();
            // note: this teapot was meant for OpenGL where the origin is at the lower left
            //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
            let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
            let proj =
                cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0)
                    * Matrix4::from_nonuniform_scale(1f32, 1f32, -1f32);

            const FORWARD_VEL: f32 = 1.0 / 60.0 * 2.0;
            if input_state.forward {
                player_model.apply_move_forward(FORWARD_VEL);
            }
            if input_state.backward {
                player_model.apply_move_forward(-FORWARD_VEL);
            }
            if input_state.left {
                player_model.apply_move_right(-FORWARD_VEL);
            }
            if input_state.right {
                player_model.apply_move_right(FORWARD_VEL);
            }

            println!("{:?}", player_model);

            let uniform_data = vs::ty::Data {
                world: <Matrix4<f32> as Transform<Point3<f32>>>::one().into(), // from(rotation).into(),
                view: player_model.get_view_matrix().into(), //(view * scale).into(),
                proj: proj.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            render_test.device.clone(),
            render_test.queue.family(),
        )
        .unwrap()
        .begin_render_pass(
            framebuffers[image_num].clone(),
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
    }
}

fn main() {
    let delegate = CrystalRenderDelgate::new();

    render_bits::render_test(delegate);
}
