// Copyright (c) 2019 Simon A. Berger
// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// inital vulkano initialization based on the teapot.rs example from vulkano-rs

extern crate cgmath;
extern crate num_traits;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use crate::render_bits::text_console::{TextConsole, TextConsoleLog};
use crate::script;
use custom_error;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;

use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;
use winit::Window;

use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Point3, Vector4};

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

pub mod frame;
pub mod math;
// pub mod render_test2;
pub mod text_console;

lazy_static! {
    pub static ref LOGGER: TextConsoleLog = TextConsoleLog::empty();
}

custom_error::custom_error! { pub Error
    ChannelCommunicationError = "communication error on mpsc channel",

    UnhandledVulkanoError{ error: String } = "unhandled vulkano error {error}",

    NotInitialized = "Not initialized",
    NotImplemented = "Not implemented",
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: (f32, f32, f32),
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Copy, Clone)]
pub struct TexCoord {
    pub tex: (f32, f32),
}

vulkano::impl_vertex!(TexCoord, tex);

#[derive(Copy, Clone)]
pub struct Normal {
    pub normal: (f32, f32, f32),
}

impl_vertex!(Normal, normal);

impl From<cgmath::Vector3<f32>> for Vertex {
    fn from(v: cgmath::Vector3<f32>) -> Vertex {
        Vertex {
            position: (v.x, v.y, v.z),
        }
    }
}

impl From<cgmath::Vector3<f32>> for Normal {
    fn from(v: cgmath::Vector3<f32>) -> Normal {
        Normal {
            normal: (v.x, v.y, v.z),
        }
    }
}

type Vec4 = Vector4<f32>;

#[derive(Copy, Clone)]
pub enum InputEvent {
    Key(winit::VirtualKeyCode, winit::ElementState),
    KeyFocus(bool),
    Character(char),
    PointerDelta(f32, f32),
}

pub struct PlayerFlyModel {
    lon: Deg<f32>,
    lat: Deg<f32>,
    pos: Point3<f32>,
}
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub action1: bool,
    pub action2: bool,
    pub run: bool,
    pub d_lon: Deg<f32>,
    pub d_lat: Deg<f32>,

    pub z_neg: bool,
    pub z_pos: bool,
    pub x_neg: bool,
    pub x_pos: bool,
}

impl InputState {
    fn new() -> Self {
        InputState {
            forward: false,
            backward: false,
            left: false,
            right: false,
            action1: false,
            action2: false,
            run: false,
            d_lon: Deg(0f32),
            d_lat: Deg(0f32),
            z_neg: false,
            z_pos: false,
            x_neg: false,
            x_pos: false,
        }
    }

    pub fn delta_lon(&mut self) -> Deg<f32> {
        let ret = self.d_lon;
        self.d_lon = Deg(0f32);
        ret
    }

    pub fn delta_lat(&mut self) -> Deg<f32> {
        let ret = self.d_lat;
        self.d_lat = Deg(0f32);
        ret
    }
}

impl PlayerFlyModel {
    pub fn new(pos: Point3<f32>, lon: Deg<f32>, lat: Deg<f32>) -> Self {
        PlayerFlyModel {
            lon: lon,
            lat: lat,
            pos: pos,
        }
    }

    pub fn apply_delta_lon(&mut self, d: Deg<f32>) {
        self.lon += -d; // rhs coordinate system -> positive lat means turn left
    }
    pub fn apply_delta_lat(&mut self, d: Deg<f32>) {
        self.lat = num_traits::clamp(self.lat - d, Deg(-90.0), Deg(90.0));
    }
    pub fn get_rotation_lon(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(self.lon)
    }
    pub fn get_rotation_lat(&self) -> Matrix4<f32> {
        Matrix4::from_angle_x(self.lat)
    }
    pub fn apply_move_forward(&mut self, d: f32) {
        self.pos +=
            (self.get_rotation_lon() * self.get_rotation_lat() * Vec4::new(0.0, 0.0, -d, 0.0)) // rhs coord system -> forward is negative-z
                .truncate();
    }
    pub fn apply_move_right(&mut self, d: f32) {
        self.pos +=
            (self.get_rotation_lon() * self.get_rotation_lat() * Vec4::new(d, 0.0, 0.0, 0.0))
                .truncate();
    }
    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        // (self.get_rotation_lat(true) * self.get_rotation_lon(true) * self.get_translation(true)).invert().unwrap()
        if let Some(mat) =
            (self.get_translation() * self.get_rotation_lon() * self.get_rotation_lat()).invert()
        {
            mat
        } else {
            <Matrix4<f32> as One>::one()
        }
    }
    pub fn get_translation(&self) -> Matrix4<f32> {
        // Matrix4::from_translation( Point3::<f32>::to_vec(self.pos) )
        Matrix4::from_translation(self.pos.to_vec())
    }
}

pub struct InputStateEventDispatcher {
    pub input_state: InputState,
    pub sink: Sender<InputEvent>,
    source: Receiver<InputEvent>,
}

impl InputStateEventDispatcher {
    pub fn new() -> InputStateEventDispatcher {
        let (tx, rx) = std::sync::mpsc::channel();
        InputStateEventDispatcher {
            input_state: InputState::new(),
            sink: tx,
            source: rx,
        }
    }

    pub fn update(&mut self) {
        loop {
            match self.source.try_recv() {
                Ok(InputEvent::Key(keycode, state)) => {
                    let down = state == winit::ElementState::Pressed;

                    match keycode {
                        winit::VirtualKeyCode::W => self.input_state.forward = down,
                        winit::VirtualKeyCode::S => self.input_state.backward = down,
                        winit::VirtualKeyCode::A => self.input_state.left = down,
                        winit::VirtualKeyCode::D => self.input_state.right = down,
                        winit::VirtualKeyCode::I => self.input_state.z_neg = down,
                        winit::VirtualKeyCode::K => self.input_state.z_pos = down,
                        winit::VirtualKeyCode::J => self.input_state.x_neg = down,
                        winit::VirtualKeyCode::L => self.input_state.x_pos = down,
                        winit::VirtualKeyCode::Q => self.input_state.action1 = down,
                        winit::VirtualKeyCode::E => self.input_state.action2 = down,
                        winit::VirtualKeyCode::LShift | winit::VirtualKeyCode::RShift => {
                            self.input_state.run = down
                        }
                        _ => {}
                    }
                }
                Ok(InputEvent::PointerDelta(x, y)) => {
                    self.input_state.d_lon += Deg(x);
                    self.input_state.d_lat += Deg(y);
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
}

impl std::fmt::Debug for PlayerFlyModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "pos: [{} {} {}] rot: {:?} {:?}",
            self.pos.x, self.pos.y, self.pos.z, self.lon, self.lat
        )
    }
}

pub struct ConsoleInputMultiplexer {
    pub sink: Sender<InputEvent>,
    source: Receiver<InputEvent>,

    pub mx_sink1: Option<Sender<InputEvent>>,
    pub mx_sink2: Option<Sender<InputEvent>>,

    s2_active: bool,
    key: winit::VirtualKeyCode,
}

impl ConsoleInputMultiplexer {
    pub fn new() -> ConsoleInputMultiplexer {
        let (tx, rx) = std::sync::mpsc::channel();
        ConsoleInputMultiplexer {
            sink: tx,
            source: rx,
            mx_sink1: None,
            mx_sink2: None,
            s2_active: false,
            key: winit::VirtualKeyCode::Grave,
        }
    }

    pub fn update(&mut self) {
        loop {
            match self.source.try_recv() {
                Ok(InputEvent::Key(key, state)) => {
                    if key == self.key && state == winit::ElementState::Released {
                        self.dispatch(InputEvent::KeyFocus(false));
                        self.s2_active = !self.s2_active;
                        self.dispatch(InputEvent::KeyFocus(true));
                    } else {
                        self.dispatch(InputEvent::Key(key, state));
                    }
                }

                Ok(event) => self.dispatch(event),
                Err(_) => break,
            }
        }
    }

    fn dispatch(&mut self, event: InputEvent) {
        if self.s2_active {
            if let Some(mx_sink2) = &self.mx_sink2 {
                let ret = mx_sink2.send(event);

                if ret.is_err() {
                    self.mx_sink2 = None;
                }
            }
        } else {
            if let Some(mx_sink1) = &self.mx_sink1 {
                let ret = mx_sink1.send(event);
                if ret.is_err() {
                    self.mx_sink1 = None;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct VulcanoState {
    surface: Arc<vulkano::swapchain::Surface<Window>>,
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,

    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
}

impl VulcanoState {
    pub fn dimension(&self) -> [u32; 2] {
        let window = self.surface.window();
        if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            panic!("panic")
        }
    }

    pub fn queue(&self) -> Arc<vulkano::device::Queue> {
        self.queue.clone()
    }

    pub fn device(&self) -> Arc<Device> {
        self.queue.device().clone()
    }

    pub fn main_subpass(
        &self,
    ) -> vulkano::framebuffer::Subpass<Arc<RenderPassAbstract + Send + Sync>> {
        vulkano::framebuffer::Subpass::<_>::from(self.render_pass.clone(), 0).unwrap()
    }
}

pub struct RenderTest {
    vk_state: VulcanoState,
    events_loop: winit::EventsLoop,

    text_console: TextConsole,
    script_env: script::Environment,
    input_multiplexer: ConsoleInputMultiplexer,

    script_lines_sink: Sender<String>,
    script_lines_source: Receiver<String>,

    timed: bool,
}

impl RenderTest {
    pub fn new(timed: bool) -> Result<RenderTest> {
        let extensions = vulkano_win::required_extensions();
        let instance =
            Instance::new(None, &extensions, None).map_err(|_| Error::UnhandledVulkanoError {
                error: "Instance::new failed".into(),
            })?;

        let events_loop = winit::EventsLoop::new();
        let surface = winit::WindowBuilder::new()
            .build_vk_surface(&events_loop, instance.clone())
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "failed to build vk surface".into(),
            })?;

        let dimensions = {
            let window = surface.window();
            if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                panic!("panic")
            }
        };

        let (device, mut queues) = {
            // for phys in PhysicalDevice::enumerate(&instance) {
            //     let caps = surface.capabilities(phys.clone());
            //     println!("caps: {:?}", caps);
            // }
            let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

            println!(
                "Using device: {} (type: {:?})",
                physical.name(),
                physical.ty()
            );
            let queue_family = physical
                .queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                .unwrap();

            let device_ext = DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            };

            Device::new(
                physical,
                physical.supported_features(),
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "Device::new failed".into(),
            })?
        };
        let queue = queues.next().unwrap();
        // let window = surface.window();
        let (swapchain, images) = {
            let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

            // for p in PhysicalDevice::enumerate(&instance) {
            //     println!("{:?}", p);
            // }
            let caps = surface.capabilities(physical).unwrap();

            let usage = caps.supported_usage_flags;
            let mut format = caps.supported_formats[0].0;
            for (f, c) in caps.supported_formats.iter().cloned() {
                if c == vulkano::swapchain::ColorSpace::SrgbNonLinear
                    && f == vulkano::format::Format::B8G8R8A8Unorm
                {
                    format = f;
                }
            }

            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            let present_mode = if timed {
                PresentMode::Immediate
            } else {
                PresentMode::Fifo
            };

            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                usage,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                present_mode,
                true,
                None,
            )
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "Swapchain::new failed".into(),
            })?
        };

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D32Sfloat,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            )
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "failed to create single pass render pass".into(),
            })?,
        );

        let vk_state = VulcanoState {
            device: device,
            surface: surface,
            queue: queue,
            swapchain: swapchain,
            images: images,
            render_pass: render_pass,
        };

        let (btx, brx) = channel();
        let mut text_console = TextConsole::new(&vk_state, script::BindingDispatcher::new(brx));

        let mut script_env = script::Environment::new();
        script_env.subscribe(btx);

        text_console.watch_value("light_mode");
        text_console.watch_value("light_pos");
        text_console.watch_value("rad_bps");

        LOGGER.set_sink(text_console.get_sender());
        log::set_logger(&*LOGGER)
            .map(|()| log::set_max_level(log::LevelFilter::Info))
            .unwrap();

        let (script_lines_sink, script_lines_source) = channel();

        Ok(RenderTest {
            vk_state: vk_state,

            events_loop: events_loop,
            text_console: text_console,
            input_multiplexer: ConsoleInputMultiplexer::new(),
            script_env: script_env,
            timed: timed,
            script_lines_sink: script_lines_sink,
            script_lines_source: script_lines_source,
        })
    }

    fn window(&self) -> &Window {
        self.vk_state.surface.window()
    }
    // pub fn dimension(&self) -> [u32; 2] {
    //     let window = self.window();
    //     if let Some(dimensions) = window.get_inner_size() {
    //         let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
    //         [dimensions.0, dimensions.1]
    //     } else {
    //         panic!("panic")
    //     }
    // }
    fn recreate_swapchain(&mut self) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        let dimensions = self.vk_state.dimension();
        let (new_swapchain, new_images) =
            match self.vk_state.swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // Err(SwapchainCreationError::UnsupportedDimensions) => panic!("{:?}", err),
                Err(err) => panic!("{:?}", err),
            };
        self.vk_state.swapchain = new_swapchain;
        self.vk_state.images = new_images;

        self.window_size_dependent_setup()
    }

    fn window_size_dependent_setup(&self) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        let dimensions = self.vk_state.images[0].dimensions();

        let depth_buffer =
            AttachmentImage::transient(self.vk_state.device.clone(), dimensions, Format::D32Sfloat)
                .unwrap();

        let framebuffers = self
            .vk_state
            .images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(self.vk_state.render_pass.clone())
                        .add(image.clone())
                        .unwrap()
                        .add(depth_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>();

        framebuffers
    }

    pub fn main_loop(&mut self, delegate: &mut dyn RenderDelegate) -> Result<()> {
        self.input_multiplexer.mx_sink1 = Some(delegate.get_input_sink());

        let framebuffers = self.window_size_dependent_setup();
        delegate.framebuffer_changed(&self.vk_state);
        // let mut pipeline = delegate.create_pipeline(&self);
        let mut recreate_swapchain = false;

        self.input_multiplexer.mx_sink2 = Some(self.text_console.get_input_sink().clone());

        //let mut previous_frame = delegate.init(&self.vk_state, &mut self.script_env);
        let mut previous_frame = Box::new(vulkano::sync::now(self.vk_state.device.clone()))
            as Box<vulkano::sync::GpuFuture>;
        let mut old_pos = None as Option<winit::dpi::LogicalPosition>;

        self.text_console
            .set_input_line_sink(self.script_lines_sink.clone());

        let mut last_frame_start = std::time::Instant::now();
        let target_frame_time = std::time::Duration::from_micros(1000000 / 60);
        loop {
            if self.timed {
                let elapsed = last_frame_start.elapsed();
                if elapsed < target_frame_time {
                    std::thread::sleep(target_frame_time - elapsed);
                }
                last_frame_start = std::time::Instant::now();
            }
            previous_frame.cleanup_finished();

            if recreate_swapchain {
                // framebuffers = self.recreate_swapchain();
                delegate.framebuffer_changed(&self.vk_state);
                self.text_console.framebuffer_changed(&self.vk_state);

                // pipeline = delegate.create_pipeline(&self);
                recreate_swapchain = false;
            }
            let (image_num, acquire_future) =
                match swapchain::acquire_next_image(self.vk_state.swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        continue;
                    }
                    Err(err) => panic!("{:?}", err),
                };
            self.text_console.receive_input(&self.script_env);
            // self.text_console.receive_input(self);
            self.text_console.update(&self.vk_state);
            let update_fut = delegate.update(&self.vk_state, &mut self.script_env);

            loop {
                if let Ok(line) = self.script_lines_source.try_recv() {
                    script::parse(&line, &mut self.script_env);
                } else {
                    break;
                }
            }

            let builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.vk_state.device.clone(),
                self.vk_state.queue.family(),
            )
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "failed to create auto command buffer buidlder in mainloop".into(),
            })?
            // .update_buffer(colors_buffer_gpu.clone(), self.colors_cpu[..]).unwrap()
            .begin_render_pass(
                framebuffers[image_num].clone(),
                false,
                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
            )
            .map_err(|_| Error::UnhandledVulkanoError {
                error: "failed begin render pass in mainloop".into(),
            })?;

            // let builder1 = AutoCommandBufferBuilder::secondary_graphics_one_time_submit(
            //     self.vk_state.device.clone(),
            //     self.vk_state.queue.family(),
            //     Subpass::from(
            // )
            // .unwrap();

            let builder = match delegate.render_frame(&self.vk_state, builder) {
                Ok(builder) => builder,
                Err(error) => panic!("error: {}", error),
            };

            let builder = match self.text_console.render(builder) {
                Ok(builder) => builder,
                Err(error) => panic!("error: {}", error),
            };
            let command_buffer = builder
                .end_render_pass()
                .map_err(|_| Error::UnhandledVulkanoError {
                    error: "failed end render pass in mainloop".into(),
                })?
                .build();

            if let Ok(command_buffer) = command_buffer {
                let future = previous_frame
                    .join(update_fut)
                    .join(acquire_future)
                    .then_execute(self.vk_state.queue.clone(), command_buffer)
                    .map_err(|_| Error::UnhandledVulkanoError {
                        error: "failed to execute render command buffer in mainloop".into(),
                    })?
                    .then_swapchain_present(
                        self.vk_state.queue.clone(),
                        self.vk_state.swapchain.clone(),
                        image_num,
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame = Box::new(future) as Box<_>;
                    }
                    Err(sync::FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame =
                            Box::new(sync::now(self.vk_state.device.clone())) as Box<_>;
                    }
                    Err(e) => {
                        println!("{:?}", e);
                        previous_frame =
                            Box::new(sync::now(self.vk_state.device.clone())) as Box<_>;
                    }
                }
            };

            let mut done = false;
            let mut events = Vec::new();

            // let input_sinks = self.input_sinks.clone();
            // let mut down_keys = HashSet::new();
            // let mut tmp = Vec::new();

            self.events_loop.poll_events(|ev| {
                // tmp.push(format!("{:?}", ev));
                match ev {
                    winit::Event::WindowEvent {
                        event: winit::WindowEvent::CloseRequested,
                        ..
                    } => done = true,
                    winit::Event::WindowEvent {
                        event: winit::WindowEvent::Resized(_),
                        ..
                    } => recreate_swapchain = true,
                    winit::Event::WindowEvent {
                        event: winit::WindowEvent::CursorMoved { position: pos, .. },
                        ..
                    } => {
                        if let Some(op) = old_pos {
                            // input_state.d_lon = Deg((pos.x - op.x) as f32);
                            // input_state.d_lat = Deg((pos.y - op.y) as f32);

                            events.push(InputEvent::PointerDelta(
                                (pos.x - op.x) as f32,
                                (pos.y - op.y) as f32,
                            ));
                        }

                        old_pos = Some(pos);
                    }
                    winit::Event::WindowEvent {
                        event:
                            winit::WindowEvent::KeyboardInput {
                                input:
                                    winit::KeyboardInput {
                                        state,
                                        virtual_keycode: Some(keycode),
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        events.push(InputEvent::Key(keycode, state));
                        match keycode {
                            winit::VirtualKeyCode::F3 => done = true,
                            _ => {}
                        }
                    }
                    winit::Event::WindowEvent {
                        event: winit::WindowEvent::ReceivedCharacter(c),
                        ..
                    } => {
                        // ignore 'grave' since it is used for console switching (and no one will ever use it...)
                        if c != '`' {
                            events.push(InputEvent::Character(c));
                        }
                    }
                    _ => (),
                }
            });

            // for t in tmp {
            //     self.text_console.add_line(&t);
            // }

            for event in events {
                // self.input_sinks
                //     .drain_filter(|sink| sink.send(event.clone()).is_err());
                self.input_multiplexer
                    .sink
                    .send(event)
                    .map_err(|_| Error::ChannelCommunicationError)?;
            }
            self.input_multiplexer.update();

            if done {
                return Ok(());
            }
        }
    }
    // pub fn event(ev: winit::Event) {}
    pub fn set_player_input_sink(&mut self, sender: Sender<InputEvent>) {
        self.input_multiplexer.mx_sink1 = Some(sender);
    }

    pub fn vk_state(&self) -> VulcanoState {
        self.vk_state.clone()
    }

    pub fn script_env(&mut self) -> &mut script::Environment {
        &mut self.script_env
    }
    pub fn script_lines_sink(&self) -> Sender<String> {
        self.script_lines_sink.clone()
    }
}

// fn get_completion_query(template: &String) -> CompletionQuery {
//     let tokens = template.split_whitespace().collect::<Vec<_>>();

//     if tokens.len() == 2 && tokens[0] == "set" {
//         CompletionQuery::Variable(tokens[1].into())
//     } else {
//         CompletionQuery::None
//     }
// }

impl text_console::CompletionProvider for script::Environment {
    fn complete(&self, template: &str) -> Vec<String> {
        // let mut completions = Vec::new();
        // match template {
        //     text_console::CompletionQuery::Variable(name) => {
        //         for (key, _) in &self.variables {
        //             if key.len() < name.len() {
        //                 continue;
        //             }

        //             if key[..name.len()] == name {
        //                 completions.push(key.clone());
        //             }
        //         }
        //     }
        //     _ => (),
        // }

        // completions
        script::complete(template, self)
        //        vec![template.into()]
    }
}

pub trait RenderDelegate {
    // fn init(
    //     &mut self,
    //     vk_state: &VulcanoState,
    //     script_env: &mut script::Environment,
    // ) -> Box<vulkano::sync::GpuFuture>;
    // fn init(&mut self, render_test: &mut RenderTest) -> Box<vulkano::sync::GpuFuture>;
    fn shutdown(self);
    fn framebuffer_changed(&mut self, vk_state: &VulcanoState);
    fn update(
        &mut self,
        vk_state: &VulcanoState,
        script_env: &mut script::Environment,
    ) -> Box<GpuFuture>;
    fn render_frame(
        &mut self,
        vk_state: &VulcanoState,
        builder: AutoCommandBufferBuilder,
    ) -> Result<AutoCommandBufferBuilder>;

    fn get_input_sink(&self) -> Sender<InputEvent>;
}

pub fn render_test(delegate: &mut RenderDelegate, timed: bool) -> Result<()> {
    let mut render_test = RenderTest::new(timed)?;
    //let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();

    render_test.main_loop(delegate)
}

pub fn render_test2<F: FnOnce(&VulcanoState) -> Box<RenderDelegate>>(
    factory: F,
    timed: bool,
) -> Result<()> {
    let mut render_test = RenderTest::new(timed)?;
    let mut delegate = factory(&render_test.vk_state());
    render_test.main_loop(&mut *delegate)?;
    //delegate.shutdown();
    drop(delegate);
    Ok(())
}
