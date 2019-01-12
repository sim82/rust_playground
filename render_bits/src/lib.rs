// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate cgmath;
extern crate num_traits;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;

use winit::Window;

use cgmath::prelude::*;
use cgmath::{Deg, Matrix3, Matrix4, Point3, Rad, Vector3, Vector4};

use num_traits::clamp;
use std::iter;
use std::sync::Arc;
use std::time::Instant;

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

type Vec3 = Vector3<f32>;
type Vec4 = Vector4<f32>;

struct PlayerFlyModel {
    lon: Deg<f32>,
    lat: Deg<f32>,
    pos: Point3<f32>,
}
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
}

impl InputState {
    fn new() -> Self {
        InputState {
            forward: false,
            backward: false,
            left: false,
            right: false,
        }
    }
}

impl PlayerFlyModel {
    fn new() -> Self {
        PlayerFlyModel {
            lon: Deg(0.0),
            lat: Deg(0.0),
            pos: Point3::origin(),
        }
    }

    fn apply_delta_lon(&mut self, d: f32) {
        self.lon += Deg(d);
    }
    fn apply_delta_lat(&mut self, d: f32) {
        self.lat = num_traits::clamp(self.lat + Deg(d), Deg(-90.0), Deg(90.0));
    }
    fn get_rotation_lon(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(self.lon)
    }
    fn get_rotation_lat(&self) -> Matrix4<f32> {
        Matrix4::from_angle_x(self.lat)
    }
    fn apply_move_forward(&mut self, d: f32) {
        self.pos +=
            (self.get_rotation_lon() * self.get_rotation_lat() * Vec4::new(0.0, 0.0, d, 0.0))
                .truncate();
    }
    fn apply_move_right(&mut self, d: f32) {
        self.pos +=
            (self.get_rotation_lon() * self.get_rotation_lat() * Vec4::new(d, 0.0, 0.0, 0.0))
                .truncate();
    }
    fn get_view_matrix(&self) -> Matrix4<f32> {
        // (self.get_rotation_lat(true) * self.get_rotation_lon(true) * self.get_translation(true)).invert().unwrap()
        (self.get_translation() * self.get_rotation_lon() * self.get_rotation_lat())
            .invert()
            .unwrap()
    }
    fn get_translation(&self) -> Matrix4<f32> {
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
        );

        Ok(())
    }
}

struct RenderTest {
    instance: Arc<Instance>,
    events_loop: winit::EventsLoop,
    //window : Window,
    surface: Arc<vulkano::swapchain::Surface<Window>>,
    // physical : PhysicalDevice<'a>,
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,

    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,

    delegate: Arc<RenderDelegate>,
}

impl RenderTest {
    fn new(delegate: Arc<RenderDelegate>) -> RenderTest {
        let extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &extensions, None).unwrap();
        let mut events_loop = winit::EventsLoop::new();
        let surface = winit::WindowBuilder::new()
            .build_vk_surface(&events_loop, instance.clone())
            .unwrap();

        let mut dimensions = {
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
        let (mut swapchain, images) = {
            let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

            let caps = surface.capabilities(physical).unwrap();
            let usage = caps.supported_usage_flags;
            let format = caps.supported_formats[0].0;
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
                        format: Format::D16Unorm,
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
            instance: instance,
            events_loop: events_loop,
            surface: surface,
            device: device,
            queue: queue,
            swapchain: swapchain,
            images: images,
            render_pass: render_pass,
            delegate: delegate,
        }
    }

    fn window(&self) -> &Window {
        self.surface.window()
    }
    fn dimension(&self) -> [u32; 2] {
        let window = self.window();
        if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            panic!("panic")
        }
    }
    fn recreate_swapchain(
        &mut self,
    ) -> (
        Arc<GraphicsPipelineAbstract + Send + Sync>,
        Vec<Arc<FramebufferAbstract + Send + Sync>>,
    ) {
        let dimensions = self.dimension();
        let vs = vs::Shader::load(self.device.clone()).unwrap();
        let fs = fs::Shader::load(self.device.clone()).unwrap();
        let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dimensions) {
            Ok(r) => r,
            // Err(SwapchainCreationError::UnsupportedDimensions) => panic!("{:?}", err),
            Err(err) => panic!("{:?}", err),
        };
        self.swapchain = new_swapchain;
        self.images = new_images;

        self.window_size_dependent_setup()
    }

    fn window_size_dependent_setup(
        &self,
    ) -> (
        Arc<GraphicsPipelineAbstract + Send + Sync>,
        Vec<Arc<FramebufferAbstract + Send + Sync>>,
    ) {
        let dimensions = self.images[0].dimensions();

        let depth_buffer =
            AttachmentImage::transient(self.device.clone(), dimensions, Format::D16Unorm).unwrap();

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

        let pipeline = self.delegate.create_pipeline(self);
        (pipeline, framebuffers)
    }

    // fn mainloop(&mut self) {

    // }
}

trait RenderDelegate {
    fn create_pipeline(
        &self,
        render_test: &RenderTest,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync>;
}

struct CrystalRenderDelgate {}

impl CrystalRenderDelgate {
    fn new() -> Arc<CrystalRenderDelgate> {
        Arc::new(CrystalRenderDelgate {})
    }
}

impl RenderDelegate for CrystalRenderDelgate {
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
}

pub fn render_test(vertices: &[Vertex], normals: &[Normal], indices: &[u16]) {
    let delegate = CrystalRenderDelgate::new();

    let mut render_test = RenderTest::new(delegate.clone());
    //let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();
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
        ImmutableBuffer::from_iter(normals, BufferUsage::all(), render_test.queue.clone()).unwrap();

    let indices = indices.iter().cloned();
    let (index_buffer, ib_future) =
        ImmutableBuffer::from_iter(indices, BufferUsage::all(), render_test.queue.clone()).unwrap();

    let fence_signal = Arc::new(
        vb_future
            .join(nb_future.join(ib_future))
            .then_signal_fence_and_flush()
            .unwrap(),
    );

    let uniform_buffer =
        CpuBufferPool::<vs::ty::Data>::new(render_test.device.clone(), BufferUsage::all());

    let vs = vs::Shader::load(render_test.device.clone()).unwrap();
    let fs = fs::Shader::load(render_test.device.clone()).unwrap();

    let (mut pipeline, mut framebuffers) = render_test.window_size_dependent_setup();
    let mut recreate_swapchain = false;

    // let mut previous_frame = Box::new(vb_future.join(nb_future.join(ib_future))) as Box<GpuFuture>;//Box::new(sync::now(device.clone())) as Box<GpuFuture>;
    let mut previous_frame = Box::new(sync::now(render_test.device.clone())) as Box<GpuFuture>;
    let rotation_start = Instant::now();

    fence_signal.wait(None).unwrap();

    //let mut old_pos = winit::dpi::LogicalPosition::new();

    let mut old_pos = None as Option<winit::dpi::LogicalPosition>;

    let mut lon = 0.0f32;
    let mut lat = 0.0f32;
    let mut player_model = PlayerFlyModel::new();
    let mut input_state = InputState::new();

    loop {
        previous_frame.cleanup_finished();
        let dimensions = render_test.dimension();

        if recreate_swapchain {
            let tmp = render_test.recreate_swapchain();
            pipeline = tmp.0;
            framebuffers = tmp.1;
            recreate_swapchain = false;
        }

        let uniform_buffer_subbuffer = {
            let elapsed = rotation_start.elapsed();
            let rotation =
                elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

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

        let set = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 0)
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(render_test.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
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

        let future = previous_frame
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
                // println!("{} {}", pos.x, pos.y );

                if let Some(op) = old_pos {
                    let x = (pos.x - op.x) as f32;
                    let y = (pos.y - op.y) as f32;

                    player_model.apply_delta_lat(y);
                    player_model.apply_delta_lon(x);

                    lon += x;
                    lat += y;

                    lat = lat.max(-90.0f32).min(90.0f32);

                    lon = if lon < 0.0f32 {
                        lon + 360.0f32
                    } else if lon >= 360.0f32 {
                        lon - 360.0f32
                    } else {
                        lon
                    };
                    // println!("{} {}", lon, lat);
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

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/debug_vert.glsl"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/debug_frag.glsl"
    }
}
