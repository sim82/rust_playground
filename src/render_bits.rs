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

use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;

use winit::Window;

use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Point3, Vector4};

use std::sync::Arc;

pub mod math;
pub mod text_console;

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

impl std::fmt::Debug for PlayerFlyModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "pos: [{} {} {}] rot: {:?} {:?}",
            self.pos.x, self.pos.y, self.pos.z, self.lon, self.lat
        )
    }
}

pub struct RenderTest {
    // instance: Arc<Instance>,
    events_loop: winit::EventsLoop,
    pub surface: Arc<vulkano::swapchain::Surface<Window>>,
    pub device: Arc<Device>,
    pub queue: Arc<vulkano::device::Queue>,

    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    pub render_pass: Arc<RenderPassAbstract + Send + Sync>,
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

        RenderTest {
            // instance: instance,
            events_loop: events_loop,
            surface: surface,
            device: device,
            queue: queue,
            swapchain: swapchain,
            images: images,
            render_pass: render_pass,
        }
    }

    fn window(&self) -> &Window {
        self.surface.window()
    }
    pub fn dimension(&self) -> [u32; 2] {
        let window = self.window();
        if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            panic!("panic")
        }
    }
    fn recreate_swapchain(&mut self) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        let dimensions = self.dimension();
        let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dimensions) {
            Ok(r) => r,
            // Err(SwapchainCreationError::UnsupportedDimensions) => panic!("{:?}", err),
            Err(err) => panic!("{:?}", err),
        };
        self.swapchain = new_swapchain;
        self.images = new_images;

        self.window_size_dependent_setup()
    }

    fn window_size_dependent_setup(&self) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        let dimensions = self.images[0].dimensions();

        let depth_buffer =
            AttachmentImage::transient(self.device.clone(), dimensions, Format::D32Sfloat).unwrap();

        let framebuffers = self
            .images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(self.render_pass.clone())
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
}

pub trait RenderDelegate {
    fn init(&mut self, render_test: &RenderTest) -> Box<vulkano::sync::GpuFuture>;
    fn shutdown(self);
    fn create_pipeline(
        &self,
        render_test: &RenderTest,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync>;

    fn update(&mut self, render_test: &RenderTest, input_state: &InputState) -> Box<GpuFuture>;

    fn frame(
        &mut self,
        render_test: &RenderTest,
        framebuffer: Arc<vulkano::framebuffer::FramebufferAbstract + Send + Sync>,
        pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> Option<
        Box<
            vulkano::command_buffer::CommandBuffer<
                PoolAlloc = vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc,
            >,
        >,
    >;
}

pub fn render_test(delegate: &mut RenderDelegate) {
    let mut render_test = RenderTest::new();
    //let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();

    let mut framebuffers = render_test.window_size_dependent_setup();
    let mut pipeline = delegate.create_pipeline(&render_test);
    let mut recreate_swapchain = false;

    let mut previous_frame = delegate.init(&render_test);
    let mut old_pos = None as Option<winit::dpi::LogicalPosition>;
    let mut input_state = InputState::new();

    loop {
        previous_frame.cleanup_finished();

        if recreate_swapchain {
            framebuffers = render_test.recreate_swapchain();
            pipeline = delegate.create_pipeline(&render_test);
            recreate_swapchain = false;
        }
        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(render_test.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let update_fut = delegate.update(&render_test, &input_state);
        let command_buffer = delegate.frame(
            &render_test,
            framebuffers[image_num].clone(),
            pipeline.clone(),
        );
        input_state.d_lon = Deg(0f32);
        input_state.d_lat = Deg(0f32);

        if let Some(command_buffer) = command_buffer {
            let future = previous_frame
                .join(update_fut)
                .join(acquire_future)
                .then_execute(render_test.queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    render_test.queue.clone(),
                    render_test.swapchain.clone(),
                    image_num,
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame = Box::new(future) as Box<_>;
                }
                Err(sync::FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame = Box::new(sync::now(render_test.device.clone())) as Box<_>;
                }
                Err(e) => {
                    println!("{:?}", e);
                    previous_frame = Box::new(sync::now(render_test.device.clone())) as Box<_>;
                }
            }
        };

        let mut done = false;
        render_test.events_loop.poll_events(|ev| match ev {
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
                    input_state.d_lon = Deg((pos.x - op.x) as f32);
                    input_state.d_lat = Deg((pos.y - op.y) as f32);
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
                let down = state == winit::ElementState::Pressed;
                match keycode {
                    winit::VirtualKeyCode::W => input_state.forward = down,
                    winit::VirtualKeyCode::S => input_state.backward = down,
                    winit::VirtualKeyCode::A => input_state.left = down,
                    winit::VirtualKeyCode::D => input_state.right = down,
                    winit::VirtualKeyCode::I => input_state.z_neg = down,
                    winit::VirtualKeyCode::K => input_state.z_pos = down,
                    winit::VirtualKeyCode::J => input_state.x_neg = down,
                    winit::VirtualKeyCode::L => input_state.x_pos = down,
                    winit::VirtualKeyCode::Q => input_state.action1 = down,
                    winit::VirtualKeyCode::E => input_state.action2 = down,
                    winit::VirtualKeyCode::LShift | winit::VirtualKeyCode::RShift => {
                        input_state.run = down
                    }
                    winit::VirtualKeyCode::F3 => done = true,
                    _ => {}
                }
            }
            _ => (),
        });
        if done {
            return;
        }
    }
}
