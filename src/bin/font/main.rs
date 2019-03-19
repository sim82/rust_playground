use rust_playground::{render_bits, script};

use crate::render_bits::RenderDelegate;
use crate::render_bits::RenderTest;
use crate::render_bits::{InputState, VulcanoState};

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
    text_console: render_bits::text_console::TextConsole,
}
impl TestDelgate {
    fn new(vk_state: &VulcanoState, _script_env: &mut script::Environment) -> TestDelgate {
        let (_, brx) = std::sync::mpsc::channel();

        TestDelgate {
            text_console: render_bits::text_console::TextConsole::new(
                &vk_state,
                crate::script::BindingDispatcher::new(brx),
            ),
        }
    }
}

impl RenderDelegate for TestDelgate {
    fn shutdown(self) {}
    fn get_input_sink(&self) -> std::sync::mpsc::Sender<rust_playground::render_bits::InputEvent> {
        self.text_console.get_input_sink()
    }
    fn framebuffer_changed(&mut self, _vk_state: &VulcanoState) {
        // if self.text_console.is_none() {
        //     return;
        // }

        // let text_console = self.text_console.as_mut().unwrap();
        // text_console.framebuffer_changed(vk_state);
    }

    fn update(&mut self, vk_state: &VulcanoState, env: &mut script::Environment) -> Box<GpuFuture> {
        // if let Some(text_console) = &mut self.text_console {
        //     text_console.add_line(&format!("time: {:?}\n", Instant::now()));
        //     text_console.update(render_test)
        // } else {
        Box::new(vulkano::sync::now(vk_state.device()))
        // }
    }

    fn render_frame(
        &mut self,
        _vk_state: &VulcanoState,
        builder: AutoCommandBufferBuilder,
    ) -> Result<AutoCommandBufferBuilder, render_bits::Error> {
        // let builder = AutoCommandBufferBuilder::primary_one_time_submit(
        //     render_test.device.clone(),
        //     render_test.queue.family(),
        // )
        // .unwrap();

        // if self.text_console.is_none() {
        //     return Some(Box::new(builder.build().unwrap()));
        // }

        // let text_console = self.text_console.as_mut().unwrap();
        // let builder = AutoCommandBufferBuilder::primary_one_time_submit(
        //     render_test.device.clone(),
        //     render_test.queue.family(),
        // )
        // .unwrap()
        // // .update_buffer(colors_buffer_gpu.clone(), self.colors_cpu[..]).unwrap()
        // .begin_render_pass(
        //     framebuffer.clone(),
        //     false,
        //     vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
        // )
        // .unwrap();
        // let builder = text_console.render(builder, framebuffer);

        // Some(Box::new(
        //     builder.end_render_pass().unwrap().build().unwrap(),
        // ))
        Ok(builder)
        // Some(Box::new())
    }
}

fn main() {
    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    let mut render_test = render_bits::RenderTest::new(false).unwrap();
    let mut delegate = TestDelgate::new(&render_test.vk_state(), render_test.script_env());
    render_test.main_loop(&mut delegate).unwrap();
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
