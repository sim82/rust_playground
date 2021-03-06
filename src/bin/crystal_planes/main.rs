use rust_playground::{crystal, render_bits, script};

use render_bits::Error;
use render_bits::PlayerFlyModel;
use render_bits::RenderDelegate;

use render_bits::{InputStateEventDispatcher, VulcanoState};

use crystal::rad::Scene;
use crystal::{Bitmap, PlanesSep};
use crystal::{Point3, Point3i, Vec3};

use script::ToValue;

use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sync::GpuFuture;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, Vector3};

use rand::prelude::*;
use render_bits::Vertex;

use clap::{App, Arg};

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
    binding_tx: Sender<script::BindingAction>,
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
        tx_sync: Sender<()>,
        script_lines_sink: Sender<String>,
    ) -> RadWorker {
        let (tx, rx) = sync_channel(2);
        let (btx, brx) = channel();

        // let scene = Arc::new(scene);
        // let scene_thread = scene.clone();
        let join_handle = spawn(move || {
            let mut binding_dispatcher = script::BindingDispatcher::new(brx);

            let mut light_pos = crystal::Point3::new(120f32, 32f32, 80f32);
            let mut light_update = false;
            let mut last_stat = Instant::now();
            let mut do_stop = false;

            let light_mode = script::ValueWatch::new();
            binding_dispatcher.bind_value("light_mode", light_mode.clone());

            let light_pos_watch = script::ValueWatch::new();
            binding_dispatcher.bind_value("light_pos", light_pos_watch.clone());
            // let light_mode = Rc::new(RefCell::new(0));
            // let mut last_light_mode = -1;
            // binding_dispatcher.bind_i32("light_mode", light_mode.clone());
            tx_sync.send(()).unwrap();
            // let mut offs = 0;
            while !do_stop {
                binding_dispatcher.dispatch();

                if let Some(light_mode) = light_mode.borrow_mut().get_update::<i32>() {
                    match light_mode {
                        1 => {
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
                        2 => {
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
                        }
                        3 => {
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
                        _ => {}
                    }
                }

                if let Some(pos) = light_pos_watch.borrow_mut().get_update::<Point3>() {
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

                while let Ok(event) = rx_event.try_recv() {
                    match event {
                        GameEvent::Stop => do_stop = true,
                        _ => (),
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
                    // log::info!("bounces/s: {:e}", pintss);

                    script_lines_sink
                        .send(format!("set rad_bps {:e}", pintss))
                        .expect("script_lines_sink send failed");

                    last_stat = Instant::now();
                }
            }
        });
        RadWorker {
            rx: rx,
            join_handle: join_handle,
            binding_tx: btx,
        }
    }
}

struct CrystalRenderDelgate {
    player_model: PlayerFlyModel,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    colors_buffer_gpu:
        Option<Arc<CpuBufferPoolChunk<Color, Arc<vulkano::memory::pool::StdMemoryPool>>>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    last_time: std::time::Instant,

    light_pos: Point3,

    rad_worker: RadWorker,
    tx_pos: Sender<GameEvent>,

    pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>>,

    input_state: InputStateEventDispatcher,
}

impl CrystalRenderDelgate {
    fn new(
        vk_state: &VulcanoState,
        script_env: &mut script::Environment,
        script_lines_sink: Sender<String>,
    ) -> CrystalRenderDelgate {
        script_env.set("light_mode", 1i32.to_value());
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

        let mut colors_cpu = Vec::new();
        let colors_buffer_pool;

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
            vk_state.queue(),
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
            ImmutableBuffer::from_iter(indices, BufferUsage::all(), vk_state.queue()).unwrap();

        colors_buffer_pool = CpuBufferPool::new(vk_state.device(), BufferUsage::all());
        let future = Box::new(vb_future.join(ib_future));

        let (tx, rx) = channel();
        let light_pos = Point3::new(120.0, 48.0, 120.0);
        tx.send(GameEvent::UpdateLightPos(light_pos.clone()))
            .unwrap(); // send initial update

        let (tx_sync, rx_sync) = channel(); // used as semaphore to sync with thread start

        let rad_worker = RadWorker::start(
            scene,
            colors_buffer_pool,
            colors_cpu,
            rx,
            tx_sync,
            script_lines_sink,
        );
        script_env.subscribe(rad_worker.binding_tx.clone());

        rx_sync.recv().unwrap(); // sync with thread startup (e.g., it has subscribed to script variables)
        script_env.set("light_pos", light_pos.to_value());

        // render_test.set_player_input_sink(self.input_state.sink.clone());
        future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        CrystalRenderDelgate {
            player_model: PlayerFlyModel::new(
                Point3::new(17f32, 14f32, 27f32),
                cgmath::Deg(13f32),
                cgmath::Deg(-22f32),
            ),
            vertex_buffer: vertex_buffer,
            colors_buffer_gpu: None,
            index_buffer: index_buffer,
            uniform_buffer: CpuBufferPool::<vs::ty::Data>::new(
                vk_state.device(),
                BufferUsage::all(),
            ),
            last_time: Instant::now(),
            light_pos: light_pos,

            rad_worker: rad_worker,
            tx_pos: tx,

            // text_console: None,
            pipeline: None,

            input_state: InputStateEventDispatcher::new(),
        }
    }
}

impl RenderDelegate for CrystalRenderDelgate {
    fn get_input_sink(&self) -> Sender<render_bits::InputEvent> {
        self.input_state.sink.clone()
    }

    fn shutdown(self) {
        self.tx_pos.send(GameEvent::Stop).unwrap();

        drop(self.rad_worker.rx); // unblock rad_worker if necessary

        print!("joining rad_worker ...");
        self.rad_worker.join_handle.join().unwrap();
        println!(" done");
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
    fn update(&mut self, vk_state: &VulcanoState, env: &mut script::Environment) -> Box<GpuFuture> {
        self.last_time = Instant::now();
        self.input_state.update();
        // println!("time: {:?}", d_time);

        // if input_state.action1 || !self.colors_buffer_gpu.is_some() {
        // let mut rng = rand::thread_rng();

        // let scene = &mut self.scene.as_mut().unwrap();
        let input_state = &mut self.input_state.input_state;
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
            script::parse(&"set light_mode 1", env);
        }
        if input_state.action2 {
            script::parse(&"set light_mode 2", env);
        }
        if light_update {
            env.set("light_pos", self.light_pos.to_value());
        }

        if let Ok(buf) = self.rad_worker.rx.try_recv() {
            // println!("receive");
            self.colors_buffer_gpu = Some(Arc::new(buf));
        }

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

        Box::new(vulkano::sync::now(vk_state.device()))
    }

    fn render_frame(
        &mut self,
        vk_state: &VulcanoState,
        builder: AutoCommandBufferBuilder,
    ) -> render_bits::Result<AutoCommandBufferBuilder> {
        if let (Some(colors_buffer_gpu), Some(pipeline)) = (&self.colors_buffer_gpu, &self.pipeline)
        {
            let dimensions = vk_state.dimension();

            let uniform_buffer_subbuffer = {
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
                    .into(),
                };
                self.uniform_buffer.next(uniform_data).unwrap()
            };

            let set = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 0)
                    .add_buffer(uniform_buffer_subbuffer)
                    .map_err(|_| Error::UnhandledVulkanoError {
                        error: "failed to add buffer to descriptor set".into(),
                    })?
                    .build()
                    .map_err(|_| Error::UnhandledVulkanoError {
                        error: "failed to build descriptor set".into(),
                    })?,
            );

            Ok(builder
                .draw_indexed(
                    pipeline.clone(),
                    &DynamicState::none(),
                    vec![self.vertex_buffer.clone(), colors_buffer_gpu.clone()],
                    self.index_buffer.clone(),
                    set.clone(),
                    (),
                )
                .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                    error: "draw indexed failed for crystal planes".into(),
                })?)
        } else {
            Ok(builder)
        }
    }
}

fn main() {
    let matches = App::new("crystal_planes")
        .version("1.0")
        .about("Realime Radiosity test")
        .arg(
            Arg::with_name("timed")
                .help("use time-based frame sync")
                .long("timed"),
        )
        .arg(
            Arg::with_name("threads")
                .help("set number of rayon threads")
                .long("threads")
                .takes_value(true),
        )
        .get_matches();

    let timed = matches.is_present("timed");
    if let Some(threads) = matches.value_of("threads") {
        if let Ok(num_threads) = threads.parse::<usize>() {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
        }
    }

    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }

    let mut render_test = render_bits::RenderTest::new(timed).unwrap();

    let sink = render_test.script_lines_sink();
    let mut delegate =
        CrystalRenderDelgate::new(&render_test.vk_state(), render_test.script_env(), sink);
    render_test.main_loop(&mut delegate).unwrap();
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
