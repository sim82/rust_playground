use rust_playground::render_bits;

use crate::render_bits::InputState;
use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;

use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::sync::GpuFuture;

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);

struct TestDelgate {
    text_console: Option<render_bits::text_console::TextConsole>,
}
impl TestDelgate {
    fn new() -> TestDelgate {
        TestDelgate { text_console: None }
    }
}

impl RenderDelegate for TestDelgate {
    fn init(&mut self, render_test: &mut RenderTest) -> Box<vulkano::sync::GpuFuture> {
        self.text_console = Some(render_bits::text_console::TextConsole::new(render_test));
        Box::new(vulkano::sync::now(render_test.device.clone()))
    }
    fn shutdown(self) {}

    fn framebuffer_changed(&mut self, render_test: &RenderTest) {
        if self.text_console.is_none() {
            return;
        }

        let text_console = self.text_console.as_mut().unwrap();
        text_console.framebuffer_changed(render_test);
    }

    fn update(&mut self, render_test: &RenderTest) -> Box<GpuFuture> {
        if let Some(text_console) = &mut self.text_console {
            text_console.add_line(&format!("time: {:?}\n", Instant::now()));
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
        let builder = AutoCommandBufferBuilder::primary_one_time_submit(
            render_test.device.clone(),
            render_test.queue.family(),
        )
        .unwrap();

        if self.text_console.is_none() {
            return Some(Box::new(builder.build().unwrap()));
        }

        let text_console = self.text_console.as_mut().unwrap();
        let builder = AutoCommandBufferBuilder::primary_one_time_submit(
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
        .unwrap();
        let builder = text_console.render(builder, framebuffer);

        Some(Box::new(
            builder.end_render_pass().unwrap().build().unwrap(),
        ))

        // Some(Box::new())
    }
}

fn main() {
    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    let mut delegate = TestDelgate::new();

    render_bits::render_test(&mut delegate);
    delegate.shutdown();
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/font/debug_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/font/debug_frag.glsl"
    }
}
