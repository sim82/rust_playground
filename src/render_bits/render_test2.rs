use crate::render_bits;
use crate::render_bits::frame;
use crate::render_bits::text_console::{TextConsole, TextConsoleLog};
use crate::render_bits::RenderDelegate;
use crate::render_bits::Result;
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

pub fn main_loop(delegate: &mut RenderDelegate, timed: bool) -> Result<()> {
    let mut script_env = script::Environment::new();
    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical).unwrap();

        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return Err(render_bits::Error::UnhandledVulkanoError {
                error: "window get inner_size failed".into(),
            });
        };

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            initial_dimensions,
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
    let mut delegate_initialized = false;
    // Here is the basic initialization for the deferred system.
    let mut frame_system = frame::FrameSystem::new(queue.clone(), swapchain.format());
    let mut recreate_swapchain = false;
    // let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;

    let main_subpass = frame_system.main_subpass();

    let mut vk_state = render_bits::VulcanoState {
        device: device.clone(),
        queue: queue.clone(),
        images: images.clone(),
        surface: surface.clone(),
        swapchain: swapchain.clone(),
        render_pass: frame_system.render_pass.clone(),
    };

    let mut previous_frame_end = delegate.init2(&vk_state, &mut script_env);
    delegate.framebuffer_changed(&vk_state);
    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return Err(render_bits::Error::UnhandledVulkanoError {
                    error: "window get inner_size failed".into(),
                });
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(vulkano::swapchain::SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            vk_state.swapchain = swapchain.clone();
            images = new_images;
            recreate_swapchain = false;

            delegate.framebuffer_changed(&vk_state);
        }
        let update_future = delegate.update(&vk_state, &mut script_env);

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let future = previous_frame_end.join(acquire_future).join(update_future);

        let mut frame = frame_system.frame(future, images[image_num].clone());
        let mut after_future = None;

        while let Some(pass) = frame.next_pass() {
            match pass {
                frame::Pass::Main(mut draw_pass) => {
                    let builder = AutoCommandBufferBuilder::secondary_graphics_one_time_submit(
                        queue.device().clone(),
                        queue.family(),
                        main_subpass.clone(),
                    )
                    .unwrap();

                    let builder = delegate.render_frame(&vk_state, builder).unwrap();
                    draw_pass.execute(builder.build().unwrap());
                }
                frame::Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }

        let future = after_future
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => done = true,
            winit::Event::WindowEvent {
                event: winit::WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => (),
        });
        if done {
            return Ok(());
        }
    }
}
