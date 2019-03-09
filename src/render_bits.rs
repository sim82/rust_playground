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

use crate::render_bits::text_console::TextConsole;
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

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

pub mod math;
pub mod text_console;

custom_error::custom_error! { pub Error

    BeginRenderPassError{ source : vulkano::command_buffer::BeginRenderPassError } = "begin render pass",
    DrawIndexedError{ source : vulkano::command_buffer::DrawIndexedError } = "draw indexed",
    BuildError{ source : vulkano::command_buffer::BuildError} = "build error",

    NotInitialized = "Not initialized",
    NotImplemented = "Not implemented",
}

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
        (self.get_translation() * self.get_rotation_lon() * self.get_rotation_lat())
            .invert()
            .unwrap()
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
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
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

pub struct VulcanoState {
    pub surface: Arc<vulkano::swapchain::Surface<Window>>,
    pub device: Arc<Device>,
    pub queue: Arc<vulkano::device::Queue>,

    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    pub render_pass: Arc<RenderPassAbstract + Send + Sync>,
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
}

pub struct RenderTest {
    pub vk_state: VulcanoState,
    events_loop: winit::EventsLoop,

    pub text_console: RefCell<TextConsole>,
    pub script_env: script::Environment,
    input_multiplexer: ConsoleInputMultiplexer,
}

impl RenderTest {
    fn new() -> RenderTest {
        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &extensions, None).unwrap();
        let events_loop = winit::EventsLoop::new();
        let surface = winit::WindowBuilder::new()
            .build_vk_surface(&events_loop, instance.clone())
            .unwrap();

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
            .unwrap()
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
                    || f == vulkano::format::Format::B8G8R8A8Unorm
                {
                    format = f;
                }
            }
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

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
                PresentMode::Fifo,
                true,
                None,
            )
            .unwrap()
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
            .unwrap(),
        );

        let vk_state = VulcanoState {
            device: device,
            surface: surface,
            queue: queue,
            swapchain: swapchain,
            images: images,
            render_pass: render_pass,
        };

        let text_console = TextConsole::new(&vk_state);

        RenderTest {
            vk_state: vk_state,

            events_loop: events_loop,
            text_console: RefCell::new(text_console),
            input_multiplexer: ConsoleInputMultiplexer::new(),
            script_env: script::Environment::new(),
        }
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

    fn main_loop(&mut self, delegate: &mut dyn RenderDelegate) {
        let framebuffers = self.window_size_dependent_setup();
        delegate.framebuffer_changed(&self.vk_state);
        // let mut pipeline = delegate.create_pipeline(&self);
        let mut recreate_swapchain = false;

        self.input_multiplexer.mx_sink2 = Some(self.text_console.borrow().get_input_sink().clone());

        let mut previous_frame = delegate.init(self);
        let mut old_pos = None as Option<winit::dpi::LogicalPosition>;

        let (script_line_sink, script_lines_source) = channel();
        self.text_console
            .borrow_mut()
            .set_input_line_sink(script_line_sink);
        loop {
            previous_frame.cleanup_finished();

            if recreate_swapchain {
                // framebuffers = self.recreate_swapchain();
                delegate.framebuffer_changed(&self.vk_state);
                self.text_console
                    .borrow_mut()
                    .framebuffer_changed(&self.vk_state);

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
            self.text_console.borrow_mut().receive_input(self);
            // self.text_console.receive_input(self);
            self.text_console.borrow_mut().update(&self.vk_state);
            let update_fut = delegate.update(&self.vk_state, &mut self.script_env);

            loop {
                if let Ok(line) = script_lines_source.try_recv() {
                    script::parse(&line, &mut self.script_env);
                } else {
                    break;
                }
            }

            let builder = AutoCommandBufferBuilder::primary_one_time_submit(
                self.vk_state.device.clone(),
                self.vk_state.queue.family(),
            )
            .unwrap()
            // .update_buffer(colors_buffer_gpu.clone(), self.colors_cpu[..]).unwrap()
            .begin_render_pass(
                framebuffers[image_num].clone(),
                false,
                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
            )
            .unwrap();

            let builder = match delegate.render_frame(&self.vk_state, builder) {
                Ok(builder) => builder,
                Err(error) => panic!("error: {}", error),
            };

            let builder = match self.text_console.borrow_mut().render(builder) {
                Ok(builder) => builder,
                Err(error) => panic!("error: {}", error),
            };
            let command_buffer = builder.end_render_pass().unwrap().build();

            if let Ok(command_buffer) = command_buffer {
                let future = previous_frame
                    .join(update_fut)
                    .join(acquire_future)
                    .then_execute(self.vk_state.queue.clone(), command_buffer)
                    .unwrap()
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
                self.input_multiplexer.sink.send(event).unwrap();
            }
            self.input_multiplexer.update();

            if done {
                return;
            }
        }
    }
    // pub fn event(ev: winit::Event) {}
    pub fn set_player_input_sink(&mut self, sender: Sender<InputEvent>) {
        self.input_multiplexer.mx_sink1 = Some(sender);
    }
}

impl text_console::CompletionProvider for RenderTest {
    fn complete(&self, template: text_console::CompletionQuery) -> Vec<String> {
        let mut completions = Vec::new();
        match template {
            text_console::CompletionQuery::Variable(name) => {
                for (key, _) in &self.script_env.variables {
                    if key.len() < name.len() {
                        continue;
                    }

                    if key[..name.len()] == name {
                        completions.push(key.clone());
                    }
                }
            }
            _ => (),
        }

        completions
    }
}

pub trait RenderDelegate {
    fn init(&mut self, render_test: &mut RenderTest) -> Box<vulkano::sync::GpuFuture>;
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
    ) -> Result<AutoCommandBufferBuilder, Error>;
}

pub fn render_test(delegate: &mut RenderDelegate) {
    let mut render_test = RenderTest::new();
    //let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();

    render_test.main_loop(delegate);
}
