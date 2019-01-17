extern crate cgmath;
extern crate crystal;
extern crate rand;
extern crate render_bits;
extern crate vulkano;

use render_bits::InputState;
use render_bits::PlayerFlyModel;
use render_bits::RenderDelegate;
use render_bits::RenderTest;

use crystal::rad::Scene;
use crystal::PlanesSep;
use std::cell::RefCell;
use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::buffer::{
    BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, ImmutableBuffer,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Point3, Rad, Vector3};

use rand::prelude::*;
use render_bits::{Normal, Vertex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::spawn;
use std::thread::JoinHandle;

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

struct RadWorker {
    rx: Receiver<
        vulkano::buffer::cpu_pool::CpuBufferPoolChunk<
            Color,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
    >,

    join_handle: JoinHandle<()>,
}

impl RadWorker {
    fn start(
        mut scene: Scene,
        mut colors_buffer_pool: CpuBufferPool<Color>,
        mut colors_cpu: Vec<Color>,
        rx_pos: Receiver<crystal::Point3>,
    ) -> RadWorker {
        let (tx, rx) = channel();

        let join_handle = spawn(move || {
            // let mut light_pos = crystal::Point3::new(120f32, 32f32, 80f32);
            loop {
                while let Ok(pos) = rx_pos.try_recv() {
                    // FIXME: this is crappy...
                    scene.apply_light(pos, crystal::Vec3::new(1f32, 1f32, 1f32));
                }
                scene.do_rad();
                for (i, plane) in colors_cpu.chunks_mut(4).enumerate() {
                    // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 0.5, 1.0); //random::<f32>(), 1.0, 1.0);

                    let color = (
                        scene.rad_front[i].x,
                        scene.rad_front[i].y,
                        scene.rad_front[i].z,
                    );

                    plane[0].color = color;
                    plane[1].color = color;
                    plane[2].color = color;
                    plane[3].color = color;
                }

                let chunk = colors_buffer_pool
                    .chunk(colors_cpu.iter().map(|x| *x))
                    .unwrap();

                tx.send(chunk);
            }
        });
        RadWorker {
            rx: rx,
            join_handle: join_handle,
        }
    }
}

struct CrystalRenderDelgate {
    player_model: PlayerFlyModel,
    vertex_buffer: Option<Arc<ImmutableBuffer<[Vertex]>>>,
    normals_buffer: Option<Arc<ImmutableBuffer<[Normal]>>>,
    colors_buffer_gpu:
        Option<Arc<CpuBufferPoolChunk<Color, Arc<vulkano::memory::pool::StdMemoryPool>>>>,
    // colors_buffer_pool: Option<CpuBufferPool<Color>>,

    // colors_buffer: Option<Arc<CpuAccessibleBuffer<[Color]>>>,
    index_buffer: Option<Arc<ImmutableBuffer<[u32]>>>,
    uniform_buffer: Option<CpuBufferPool<vs::ty::Data>>,

    // colors_cpu: Vec<Color>,
    last_time: std::time::Instant,

    light_pos: Point3<f32>,

    rad_worker: Option<RadWorker>,
    tx_pos: Option<Sender<crystal::Point3>>,
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let mut hh = h;
    if hh >= 360.0 {
        hh = 0.0;
    }
    hh /= 60.0;
    let i = hh as i32; //.into();
    let ff = hh - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - (s * ff));
    let t = v * (1.0 - (s * (1.0 - ff)));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

impl CrystalRenderDelgate {
    fn new() -> Arc<RefCell<CrystalRenderDelgate>> {
        Arc::new(RefCell::new(CrystalRenderDelgate {
            player_model: PlayerFlyModel::new(
                crystal::Point3::new(31f32, 12f32, 4f32),
                cgmath::Deg(-65f32),
                cgmath::Deg(35f32),
            ),
            vertex_buffer: None,
            normals_buffer: None,
            colors_buffer_gpu: None,
            // colors_buffer_pool: None,
            // colors_buffer: None,
            index_buffer: None,
            uniform_buffer: None,
            // colors_cpu: Vec::new(),
            last_time: Instant::now(),
            // scene: None,
            light_pos: Point3::new(120.0, 32.0, 120.0),

            rad_worker: None,
            tx_pos: None,
        }))
    }
}

impl RenderDelegate for CrystalRenderDelgate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture> {
        let scene;
        {
            let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");
            let mut planes = PlanesSep::new();
            planes.create_planes(&bm);
            // planes.print();
            scene = Scene::new(planes, bm);
        }

        let future;
        let mut colors_cpu = Vec::new();
        let colors_buffer_pool;
        {
            let planes = &scene.planes;

            let mut x: Vec<(&crystal::Point3i, i32)> = planes.vertex_iter().collect();
            x.sort_by_key(|(_, v)| *v);
            let scale = 0.25f32;

            let vertices: Vec<_> = x
                .iter()
                .map(|(plane, _)| Vertex {
                    position: (
                        plane.x as f32 * scale,
                        plane.y as f32 * scale,
                        plane.z as f32 * scale,
                    ),
                })
                .collect();
            let normals: Vec<_> = planes
                .planes_iter()
                .flat_map(|plane| {
                    let normal = Normal::from(plane.dir.get_normal::<f32>());
                    vec![normal, normal, normal, normal]
                    // .iter()
                    // .map(|y| *y)
                    // .collect::<Vec<_>>()
                })
                .collect();
            assert!(vertices.len() == normals.len());

            let indices: Vec<_> = planes
                .planes_iter()
                .flat_map(|plane| {
                    vec![
                        plane.vertices[0] as u32,
                        plane.vertices[1] as u32,
                        plane.vertices[2] as u32,
                        plane.vertices[0] as u32,
                        plane.vertices[2] as u32,
                        plane.vertices[3] as u32,
                    ]
                    //[plane[0], plane[1], plane[2]]
                })
                .collect();
            let (vertex_buffer, vb_future) = ImmutableBuffer::from_iter(
                vertices.iter().cloned(),
                BufferUsage::all(),
                render_test.queue.clone(),
            )
            .unwrap();

            // let mut colors_tmp = Vec::new();
            for _ in 0..vertices.len() {
                colors_cpu.push(Color {
                    color: (
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                        rand::random::<f32>(),
                    ),
                });
            }

            let normals = normals.iter().cloned();
            let (normals_buffer, nb_future) =
                ImmutableBuffer::from_iter(normals, BufferUsage::all(), render_test.queue.clone())
                    .unwrap();

            let indices = indices.iter().cloned();
            let (index_buffer, ib_future) =
                ImmutableBuffer::from_iter(indices, BufferUsage::all(), render_test.queue.clone())
                    .unwrap();

            let uniform_buffer =
                CpuBufferPool::<vs::ty::Data>::new(render_test.device.clone(), BufferUsage::all());
            self.vertex_buffer = Some(vertex_buffer);
            self.normals_buffer = Some(normals_buffer);
            self.index_buffer = Some(index_buffer);
            self.uniform_buffer = Some(uniform_buffer);

            colors_buffer_pool = CpuBufferPool::new(render_test.device.clone(), BufferUsage::all());
            future = Box::new(vb_future.join(nb_future.join(ib_future)))
        }

        let (tx, rx) = channel();
        tx.send(self.light_pos.clone()); // send initial update

        self.tx_pos = Some(tx);
        self.rad_worker = Some(RadWorker::start(scene, colors_buffer_pool, colors_cpu, rx));

        future
    }

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
                .front_face_counter_clockwise() // face winding swapped by y-swap in vertex shader...
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
        let now = Instant::now();
        let d_time = now - self.last_time;
        self.last_time = now;

        println!("time: {:?}", d_time);

        // if input_state.action1 || !self.colors_buffer_gpu.is_some() {
        let mut rng = rand::thread_rng();

        // let scene = &mut self.scene.as_mut().unwrap();
        let mut light_update = false;
        if input_state.z_neg {
            self.light_pos -= Vector3::<f32>::unit_z();
            light_update = true;
        }
        if input_state.z_pos {
            self.light_pos += Vector3::<f32>::unit_z();
            light_update = true;
        }
        if input_state.x_neg {
            self.light_pos -= Vector3::<f32>::unit_x();
            light_update = true;
        }
        if input_state.x_pos {
            self.light_pos += Vector3::<f32>::unit_x();
            light_update = true;
        }

        if light_update {
            if let Some(tx_pos) = &self.tx_pos {
                tx_pos.send(self.light_pos.clone());
            }
        }

        // println!("light pos: {:?}", self.light_pos);

        // if input_state.action1 {
        //     scene.clear_emit();
        // } else {
        //     scene.apply_light(self.light_pos, crystal::Vec3::new(1f32, 1f32, 1f32));
        // }
        // scene.do_rad();
        // for (i, plane) in self.colors_cpu.chunks_mut(4).enumerate() {
        //     // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 0.5, 1.0); //random::<f32>(), 1.0, 1.0);

        //     let color = (
        //         scene.rad_front[i].x,
        //         scene.rad_front[i].y,
        //         scene.rad_front[i].z,
        //     );

        //     plane[0].color = color;
        //     plane[1].color = color;
        //     plane[2].color = color;
        //     plane[3].color = color;
        // }

        // match self.colors_buffer_pool {
        //     Some(ref colors_buffer_pool) => {
        //         let chunk = colors_buffer_pool
        //             .chunk(self.colors_cpu.iter().map(|x| *x))
        //             .unwrap();
        //         self.colors_buffer_gpu = Some(Arc::new(chunk));
        //     }
        //     _ => panic!("panic"),
        // };
        // }

        if let Some(rad_worker) = &self.rad_worker {
            if let Ok(buf) = rad_worker.rx.try_recv() {
                self.colors_buffer_gpu = Some(Arc::new(buf));
            }
        }

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

        println!("{:?}", self.player_model);

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
            &self.normals_buffer,
            // &self.colors_buffer,
            &self.colors_buffer_gpu,
            &self.index_buffer,
            &self.uniform_buffer,
        ) {
            (
                Some(vertex_buffer),
                Some(normals_buffer),
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
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    ) * Matrix4::from_nonuniform_scale(1f32, 1f32, -1f32);

                    let uniform_data = vs::ty::Data {
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
    let delegate = CrystalRenderDelgate::new();

    render_bits::render_test(delegate);
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/crystal_planes/debug_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/crystal_planes/debug_frag.glsl"
    }
}
