use rust_playground::{crystal, render_bits};

use crate::render_bits::text_console::TextConsole;
use crate::render_bits::InputState;
use crate::render_bits::PlayerFlyModel;
use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;

use crystal::rad::Scene;
use crystal::{Bitmap, PlanesSep};
use crystal::{Point3, Point3i, Vec3};

use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use render_bits::Vertex;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::thread::spawn;
use std::thread::JoinHandle;

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

enum GameEvent {
    UpdateLightPos(Point3),
    DoAction1,
    DoAction2,
    DoAction3,
    Stop,
}

struct RadWorker {
    rx: Receiver<
        vulkano::buffer::cpu_pool::CpuBufferPoolChunk<
            Color,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
    >,

    join_handle: JoinHandle<()>,
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
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
        0 => Vec3::new(v, t, p),
        1 => Vec3::new(q, v, p),
        2 => Vec3::new(p, v, t),
        3 => Vec3::new(p, q, v),
        4 => Vec3::new(t, p, v),
        _ => Vec3::new(v, p, q),
    }
}
impl RadWorker {
    fn start(
        mut scene: Scene,
        colors_buffer_pool: CpuBufferPool<Color>,
        mut colors_cpu: Vec<Color>,
        rx_event: Receiver<GameEvent>,
        tx_console: Sender<String>,
    ) -> RadWorker {
        let (tx, rx) = sync_channel(2);

        let join_handle = spawn(move || {
            let mut light_pos = crystal::Point3::new(120f32, 32f32, 80f32);
            let mut light_update = false;
            let mut last_stat = Instant::now();
            let mut do_stop = false;
            // let mut offs = 0;
            while !do_stop {
                while let Ok(event) = rx_event.try_recv() {
                    match event {
                        GameEvent::Stop => do_stop = true,
                        GameEvent::UpdateLightPos(pos) => {
                            light_pos = pos;
                            light_update = true;
                            // }
                            // GameEvent::DoAction1 => {

                            // let color1 = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                            let color1 = Vector3::new(1f32, 0.5f32, 0f32);
                            // let color2 = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                            let color2 = Vector3::new(0f32, 1f32, 0f32);
                            for (i, plane) in scene.planes.planes_iter().enumerate() {
                                if ((plane.cell.y) / 2) % 2 == 1 {
                                    continue;
                                }
                                scene.diffuse[i] = match plane.dir {
                                    crystal::Dir::XyPos => color1,
                                    crystal::Dir::XyNeg => color2,
                                    crystal::Dir::YzPos | crystal::Dir::YzNeg => {
                                        Vector3::new(0.8f32, 0.8f32, 0.8f32)
                                    }
                                    _ => Vector3::new(1f32, 1f32, 1f32),
                                    // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                                    // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                                }
                            }
                        }
                        GameEvent::DoAction1 => {
                            let mut rng = thread_rng();

                            let color1 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            let color2 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);
                            let color3 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            let color4 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);

                            for (i, plane) in scene.planes.planes_iter().enumerate() {
                                scene.diffuse[i] = Vector3::new(1f32, 1f32, 1f32);

                                let up = plane.cell + crystal::Dir::ZxPos.get_normal::<i32>();
                                let not_edge = (&scene.bitmap as &Bitmap).get(up);

                                scene.emit[i] = if not_edge {
                                    Vector3::zero()
                                } else {
                                    match plane.dir {
                                        crystal::Dir::YzPos => color1,
                                        crystal::Dir::YzNeg => color2,
                                        crystal::Dir::XyPos => color3,
                                        crystal::Dir::XyNeg => color4,
                                        // crystal::Dir::XyPos | crystal::Dir::XyNeg => {
                                        //     Vector3::new(0.8f32, 0.8f32, 0.8f32)
                                        // }
                                        _ => Vector3::zero(),
                                        // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                                        // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                                    }
                                }
                            }
                        }
                        GameEvent::DoAction2 => {
                            let mut rng = thread_rng();

                            let color1 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            // let color1 = Vector3::new(1f32, 0.5f32, 0f32);
                            let color2 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);

                            for (i, plane) in scene.planes.planes_iter().enumerate() {
                                scene.diffuse[i] = Vector3::new(1f32, 1f32, 1f32);
                                scene.emit[i] = if (plane.cell.y) % 3 != 0 {
                                    Vector3::zero()
                                } else {
                                    match plane.dir {
                                        crystal::Dir::XyPos => color1,
                                        crystal::Dir::XyNeg => color2,
                                        // crystal::Dir::XyPos | crystal::Dir::XyNeg => {
                                        //     Vector3::new(0.8f32, 0.8f32, 0.8f32)
                                        // }
                                        _ => Vector3::zero(),
                                        // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                                        // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                                    }
                                }
                            }
                            //offs += 1;
                        }
                        GameEvent::DoAction3 => {
                            let mut rng = thread_rng();

                            for i in 0..scene.planes.num_planes() {
                                // seriously, there is no Vec.fill?
                                scene.diffuse[i] = Vec3::new(1f32, 1f32, 1f32);
                                scene.emit[i] = Vec3::zero();
                            }

                            let num_dots = 1000;
                            for _ in 0..num_dots {
                                let i = rng.gen_range(0, scene.planes.num_planes());
                                scene.emit[i] = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                            }
                        }
                    }
                }

                if light_update {
                    scene.clear_emit();
                    scene.apply_light(light_pos, Vec3::new(1f32, 0.8f32, 0.6f32));
                    light_update = false;
                }
                scene.do_rad();
                for (i, plane) in scene.planes.planes_iter().enumerate() {
                    for v in plane.vertices.iter() {
                        colors_cpu[*v as usize].color = (
                            scene.rad_front.r[i],
                            scene.rad_front.g[i],
                            scene.rad_front.b[i],
                        );
                    }
                }
                let chunk = colors_buffer_pool
                    .chunk(colors_cpu.iter().cloned())
                    .unwrap();
                // println!("size: {} -> {}", old_cap, colors_buffer_pool.capacity());

                if tx.send(chunk).is_err() {
                    println!("send failed.");
                }
                // println!("send");

                let d_time = last_stat.elapsed();
                if d_time >= Duration::from_secs(1) {
                    let pintss = scene.pints as f64
                        / (d_time.as_secs() as f64 + d_time.subsec_nanos() as f64 * 1e-9);
                    scene.pints = 0;

                    println!("pint/s: {:e}", pintss);
                    tx_console.send(format!("bounces/s: {:e}", pintss));
                    last_stat = Instant::now();
                }
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
    colors_buffer_gpu:
        Option<Arc<CpuBufferPoolChunk<Color, Arc<vulkano::memory::pool::StdMemoryPool>>>>,
    // colors_buffer_pool: Option<CpuBufferPool<Color>>,

    // colors_buffer: Option<Arc<CpuAccessibleBuffer<[Color]>>>,
    index_buffer: Option<Arc<ImmutableBuffer<[u32]>>>,
    uniform_buffer: Option<CpuBufferPool<vs::ty::Data>>,

    // colors_cpu: Vec<Color>,
    last_time: std::time::Instant,

    light_pos: Point3,

    rad_worker: Option<RadWorker>,
    tx_pos: Option<Sender<GameEvent>>,

    text_console: Option<TextConsole>,
    pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>>,
}

impl CrystalRenderDelgate {
    fn new() -> CrystalRenderDelgate {
        CrystalRenderDelgate {
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
            light_pos: Point3::new(120.0, 48.0, 120.0),

            rad_worker: None,
            tx_pos: None,

            text_console: None,
            pipeline: None,
        }
    }
}

impl RenderDelegate for CrystalRenderDelgate {
    fn init(&mut self, render_test: &mut RenderTest) -> Box<vulkano::sync::GpuFuture> {
        let scene;
        {
            let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

            let mut planes = PlanesSep::new();
            planes.create_planes(&bm);
            // planes.print();
            scene = Scene::new(planes, bm);
            scene.print_stat();
            // panic!("exit");
        }

        let future;
        let mut colors_cpu = Vec::new();
        let colors_buffer_pool;
        {
            let planes = &scene.planes;

            let mut x: Vec<(&Point3i, i32)> = planes.vertex_iter().collect();
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

            let indices = indices.iter().cloned();
            let (index_buffer, ib_future) =
                ImmutableBuffer::from_iter(indices, BufferUsage::all(), render_test.queue.clone())
                    .unwrap();

            let uniform_buffer =
                CpuBufferPool::<vs::ty::Data>::new(render_test.device.clone(), BufferUsage::all());
            self.vertex_buffer = Some(vertex_buffer);
            self.index_buffer = Some(index_buffer);
            self.uniform_buffer = Some(uniform_buffer);

            colors_buffer_pool = CpuBufferPool::new(render_test.device.clone(), BufferUsage::all());
            future = Box::new(vb_future.join(ib_future))
        }

        let (tx, rx) = channel();
        tx.send(GameEvent::UpdateLightPos(self.light_pos.clone()))
            .unwrap(); // send initial update

        // println!("send");
        self.tx_pos = Some(tx);

        self.text_console = Some(TextConsole::new(render_test));

        self.rad_worker = Some(RadWorker::start(
            scene,
            colors_buffer_pool,
            colors_cpu,
            rx,
            self.text_console.as_ref().unwrap().get_sender(),
        ));

        future
    }
    fn shutdown(self) {
        if let Some(tx_pos) = &self.tx_pos {
            tx_pos.send(GameEvent::Stop).unwrap();
        }

        if let Some(rad_worker) = self.rad_worker {
            drop(rad_worker.rx); // unblock rad_worker if necessary

            print!("joining rad_worker ...");
            rad_worker.join_handle.join().unwrap();
            println!(" done");
        }
    }
    fn framebuffer_changed(&mut self, render_test: &RenderTest) {
        let vs = vs::Shader::load(render_test.device.clone()).unwrap();
        let fs = fs::Shader::load(render_test.device.clone()).unwrap();
        let dimensions = render_test.dimension();
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
                .render_pass(Subpass::from(render_test.render_pass.clone(), 0).unwrap())
                .build(render_test.device.clone())
                .unwrap(),
        ));
        if let Some(text_console) = &mut self.text_console {
            text_console.framebuffer_changed(render_test);
        }
    }
    fn update(&mut self, render_test: &RenderTest, input_state: &InputState) -> Box<GpuFuture> {
        // let now = Instant::now();
        // let d_time = now - self.last_time;
        self.last_time = Instant::now();

        // println!("time: {:?}", d_time);

        // if input_state.action1 || !self.colors_buffer_gpu.is_some() {
        // let mut rng = rand::thread_rng();

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
        if input_state.action1 {
            if let Some(tx_pos) = &self.tx_pos {
                tx_pos.send(GameEvent::DoAction1).unwrap();
            }
        }
        if input_state.action2 {
            if let Some(tx_pos) = &self.tx_pos {
                tx_pos.send(GameEvent::DoAction2).unwrap();
            }
        }
        if light_update {
            if let Some(tx_pos) = &self.tx_pos {
                tx_pos
                    .send(GameEvent::UpdateLightPos(self.light_pos.clone()))
                    .unwrap();
            }
        }

        if let Some(rad_worker) = &self.rad_worker {
            if let Ok(buf) = rad_worker.rx.try_recv() {
                // println!("receive");
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

        // println!("{:?}", self.player_model);

        if let Some(text_console) = &mut self.text_console {
            // text_console.add_line(&format!("{:?}", Instant::now()));
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
            &self.pipeline,
        ) {
            (
                Some(vertex_buffer),
                // Some(colors_buffer),
                Some(colors_buffer_gpu),
                Some(index_buffer),
                Some(uniform_buffer),
                Some(pipeline),
            ) => {
                let uniform_buffer_subbuffer = {
                    let dimensions = render_test.dimension();
                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;

                    let uniform_data = vs::ty::Data {
                        world: <Matrix4<f32> as Transform<Point3>>::one().into(), // from(rotation).into(),
                        view: self.player_model.get_view_matrix().into(), //(view * scale).into(),
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

                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
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
                .unwrap();

                if let Some(text_console) = &mut self.text_console {
                    builder = text_console.render(builder, framebuffer);
                }

                let builder = builder.end_render_pass().unwrap();

                Some(Box::new(builder.build().unwrap()))
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
    let mut delegate = CrystalRenderDelgate::new();

    render_bits::render_test(&mut delegate);
    delegate.shutdown();
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
