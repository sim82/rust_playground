use crate::render_bits;
use crate::render_bits::{InputEvent, TexCoord, Vertex, VulcanoState};
use log::Log;
use vulkano::buffer::{cpu_pool::CpuBufferPoolChunk, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::{FramebufferAbstract, Subpass};
use vulkano::image::ImmutableImage;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};

use cgmath::Matrix4;

use std::iter;
use std::sync::Arc;

use glyph_brush::{rusttype, BrushAction, BrushError, GlyphBrush, GlyphBrushBuilder, Section};
use image::ImageBuffer;

use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};

#[derive(Copy, Clone)]
pub struct Color {
    pub color: (f32, f32, f32),
}

vulkano::impl_vertex!(Color, color);
trait Canvas {
    fn blit(&mut self, rect: glyph_brush::rusttype::Rect<u32>, data: &[u8]);
}

impl Canvas for (&mut [u8], (u32, u32)) {
    fn blit(&mut self, rect: glyph_brush::rusttype::Rect<u32>, data: &[u8]) {
        let (mydata, (w, h)) = self;

        if rect.max.x > *w || rect.max.y > *h {
            return;
        }
        for y in 0..rect.height() {
            let ydest = y + rect.min.y;
            let dest = (ydest * *w + rect.min.x) as usize;
            let destx = dest + rect.width() as usize;
            let src = (y * rect.width()) as usize;
            let srcx = src + rect.width() as usize;

            mydata[dest..destx].copy_from_slice(&data[src..srcx]);
        }
    }
}

pub trait CompletionProvider {
    fn complete(&self, template: &str) -> Vec<String>;
}

type GlyphVertexData = (
    rusttype::Point<i32>,
    rusttype::Point<i32>,
    rusttype::Point<f32>,
    rusttype::Point<f32>,
);

struct TextConsoleBuffers {
    vb: Arc<CpuBufferPoolChunk<Vertex, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    tb: Arc<CpuBufferPoolChunk<TexCoord, Arc<vulkano::memory::pool::StdMemoryPool>>>,
    ib: Arc<CpuBufferPoolChunk<u32, Arc<vulkano::memory::pool::StdMemoryPool>>>,
}

pub struct TextConsole {
    vb_pool: CpuBufferPool<Vertex>,
    tb_pool: CpuBufferPool<TexCoord>,
    ib_pool: CpuBufferPool<u32>,
    ub_pool: CpuBufferPool<vs::ty::Data>,
    buffers: Option<TextConsoleBuffers>,
    glyph_image: Option<Arc<ImmutableImage<vulkano::format::Format>>>,
    glyph_data: Vec<u8>,
    brush: GlyphBrush<'static, GlyphVertexData>,
    sampler: Arc<vulkano::sampler::Sampler>,
    pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>>,
    text_lines: Vec<String>,

    tx: Sender<String>,
    rx: Receiver<String>,

    input_line: String,
    input_source: Receiver<render_bits::InputEvent>,
    input_sink: Sender<render_bits::InputEvent>,

    input_line_sink: Option<Sender<String>>,

    key_focus: bool,

    dimensions: [u32; 2],
    hidpi_factor: f64,

    history_line: String,
}

fn longest_common_prefix(strings: &Vec<String>) -> String {
    if strings.is_empty() {
        return "".to_string();
    }

    if strings.len() == 1 {
        return strings[0].to_string();
    }
    if let Some(min_len) = strings.iter().map(|x| x.len()).min() {
        for i in 0..min_len {
            if strings
                .iter()
                .map(|x| x[i..i + 1].to_string())
                .collect::<std::collections::HashSet<String>>()
                .len()
                != 1
            {
                return strings[0][..i].to_string();
            }
        }
    }

    "".to_string()
}

impl TextConsole {
    pub fn new(vk_state: &VulcanoState) -> Self {
        let (tx, rx) = channel();
        let (input_tx, input_rx) = channel();

        // render_test.add_input_sink(input_tx);
        TextConsole {
            vb_pool: CpuBufferPool::<Vertex>::new(vk_state.device.clone(), BufferUsage::all()),
            tb_pool: CpuBufferPool::<TexCoord>::new(vk_state.device.clone(), BufferUsage::all()),
            ib_pool: CpuBufferPool::<u32>::new(vk_state.device.clone(), BufferUsage::all()),
            ub_pool: CpuBufferPool::<vs::ty::Data>::new(
                vk_state.device.clone(),
                BufferUsage::all(),
            ),
            buffers: None,
            glyph_image: None,
            glyph_data: vec![0u8; 256 * 256],
            brush: GlyphBrushBuilder::using_font_bytes(&include_bytes!("DejaVuSansMono.ttf")[..])
                .build(),
            text_lines: Vec::new(),
            // glyph_image: gi,
            sampler: Sampler::new(
                vk_state.device.clone(),
                Filter::Linear,
                Filter::Linear,
                MipmapMode::Nearest,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .unwrap(),
            pipeline: None,
            tx: tx,
            rx: rx,
            input_line: "".into(),
            input_source: input_rx,
            input_sink: input_tx,
            key_focus: false,
            dimensions: [0u32; 2],
            hidpi_factor: 0f64,
            input_line_sink: None,
            history_line: "".into(),
        }
    }

    pub fn framebuffer_changed(&mut self, vk_state: &VulcanoState) {
        let vs = vs::Shader::load(vk_state.device.clone()).unwrap();
        let fs = fs::Shader::load(vk_state.device.clone()).unwrap();
        self.hidpi_factor = vk_state.surface.window().get_hidpi_factor();
        self.dimensions = vk_state.dimension();
        // In the triangle example we use a dynamic viewport, as its a simple example.
        // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
        // This allows the driver to optimize things, at the cost of slower window resizes.
        // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
        self.pipeline = Some(Arc::new(
            GraphicsPipeline::start()
                .vertex_input(TwoBuffersDefinition::<Vertex, TexCoord>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .front_face_clockwise()
                .cull_mode_back()
                .viewports_dynamic_scissors_irrelevant(1)
                .viewports(iter::once(Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [self.dimensions[0] as f32, self.dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }))
                .blend_alpha_blending()
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(Subpass::from(vk_state.render_pass.clone(), 0).unwrap())
                .build(vk_state.device.clone())
                .unwrap(),
        ))
    }

    pub fn add_line(&mut self, line: &str) {
        self.text_lines.push(line.into());

        while self.text_lines.len() > 20 {
            self.text_lines.remove(0);
        }
    }

    pub fn receive_input(&mut self, completion_provider: &CompletionProvider) {
        loop {
            match self.rx.try_recv() {
                Ok(string) => {
                    for line in string.lines() {
                        self.add_line(line)
                    }
                }
                _ => break,
            }
        }

        loop {
            let mut exec_lines = Vec::new();
            match self.input_source.try_recv() {
                Ok(InputEvent::Key(keycode, state)) => {
                    if state == winit::ElementState::Pressed {
                        match keycode {
                            winit::VirtualKeyCode::Back => {
                                if !self.input_line.is_empty() {
                                    self.input_line.pop();
                                }
                            }
                            winit::VirtualKeyCode::Return => {
                                exec_lines.push(self.input_line.clone());
                                self.history_line = self.input_line.clone();
                                // self.add_line(&self.input_line);
                                self.input_line.clear();
                            }
                            winit::VirtualKeyCode::Up => {
                                self.input_line = self.history_line.clone();
                            }
                            winit::VirtualKeyCode::Down => {
                                self.input_line.clear();
                            }
                            winit::VirtualKeyCode::Tab => {
                                let completions = completion_provider.complete(&self.input_line);

                                if completions.len() == 1 {
                                    self.input_line = completions[0].clone() + " ";
                                } else if completions.len() > 1 {
                                    for comp in &completions {
                                        self.add_line(comp);
                                    }
                                    self.input_line = longest_common_prefix(&completions);
                                }

                                // if completions.len() == 1 {
                                //     self.input_line = completions[0].clone();
                                // }

                                // let completion_query = self.get_completion_query();
                                // let completions = completion_provider.complete(completion_query);

                                // if completions.len() == 1 {}
                            }
                            _ => (),
                        }
                        // }
                    }
                }
                Ok(InputEvent::KeyFocus(b)) => self.key_focus = b,
                Ok(InputEvent::Character(c)) if !c.is_control() => self.input_line.push(c),
                _ => break,
            }

            for line in exec_lines {
                self.add_line(&line);
                if let Some(input_line_sink) = &self.input_line_sink {
                    input_line_sink.send(line).unwrap();
                }
            }
        }
    }

    pub fn update(&mut self, vk_state: &VulcanoState) -> Box<vulkano::sync::GpuFuture> {
        let font_size = (16f64 * self.hidpi_factor) as f32;
        let mut il: String = ">".into();

        il += &self.input_line;
        il += "_";
        let il2 = [il];
        let lines_iter: Box<std::iter::Iterator<Item = &String>> = if self.key_focus {
            Box::new(self.text_lines.iter().chain(il2.iter()))
        } else {
            let num = 5;
            if self.text_lines.len() <= num {
                Box::new(self.text_lines.iter())
            } else {
                Box::new(self.text_lines[self.text_lines.len() - num..].iter())
            }
        };
        for (i, line) in lines_iter.enumerate() {
            self.brush.queue(Section {
                //text: "MMMHello qwertyuiopasdfghjklzxcvbnmQWERTYUIOASDFGHJKLZXCVBNM",
                text: line,
                scale: glyph_brush::rusttype::Scale::uniform(font_size),
                screen_position: (0f32, font_size * i as f32),
                ..Section::default()
            });

            // println!("queue: {}", line);
        }
        // println!("done");
        let mut verts = Vec::new();
        let mut tex = Vec::new();
        let mut indices = Vec::new();
        let mut glyph_updated = false;

        loop {
            let size = self.brush.texture_dimensions();
            let glyph_data = &mut self.glyph_data;

            match self.brush.process_queued(
                (self.dimensions[0], self.dimensions[1]),
                |rect, tex_data| {
                    (&mut glyph_data[..], size).blit(rect, tex_data);
                    glyph_updated = true;
                    // println!("blit");
                },
                |vertex_data| {
                    (
                        vertex_data.pixel_coords.min,
                        vertex_data.pixel_coords.max,
                        vertex_data.tex_coords.min,
                        vertex_data.tex_coords.max,
                    )
                },
            ) {
                Ok(BrushAction::Draw(vertices)) => {
                    for (vmin, vmax, tmin, tmax) in vertices {
                        let v = verts.len() as u32;
                        indices.append(&mut vec![v, v + 1, v + 2, v, v + 2, v + 3]);

                        verts.push(Vertex {
                            position: (vmin.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmax.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmin.x as f32, vmax.y as f32, 0f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmax.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmax.y as f32),
                        });
                    }

                    if false {
                        let (width, height) = self.brush.texture_dimensions();

                        // let hf = render_test.surface.window().get_hidpi_factor(); // what's with this hidpi crap?
                        let hf = 1f64;
                        let screen_width = (self.dimensions[0] as f64 * hf) as i32;
                        let screen_height = (self.dimensions[1] as f64 * hf) as i32;

                        let vmin = rusttype::Point::<i32> {
                            x: screen_width - width as i32,
                            y: screen_height - height as i32,
                        };
                        let vmax = rusttype::Point::<i32> {
                            x: screen_width,
                            y: screen_height,
                        };

                        let tmin = rusttype::Point::<f32> { x: 0f32, y: 0f32 };
                        let tmax = rusttype::Point::<f32> { x: 1f32, y: 1f32 };

                        let v = verts.len() as u32;
                        indices.append(&mut vec![v, v + 1, v + 2, v, v + 2, v + 3]);

                        verts.push(Vertex {
                            position: (vmin.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmin.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmax.x as f32, vmax.y as f32, 0f32),
                        });
                        verts.push(Vertex {
                            position: (vmin.x as f32, vmax.y as f32, 0f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmin.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmax.x as f32, tmax.y as f32),
                        });
                        tex.push(TexCoord {
                            tex: (tmin.x as f32, tmax.y as f32),
                        });
                    }
                    self.buffers = Some(TextConsoleBuffers {
                        vb: Arc::new(self.vb_pool.chunk(verts.iter().cloned()).unwrap()),
                        tb: Arc::new(self.tb_pool.chunk(tex.iter().cloned()).unwrap()),
                        ib: Arc::new(self.ib_pool.chunk(indices.iter().cloned()).unwrap()),
                    });
                    // println!("draw");

                    break;
                }
                Ok(BrushAction::ReDraw) => {
                    // Re-draw last frame's vertices unmodified.
                    // println!("redraw");
                    break;
                }
                Err(BrushError::TextureTooSmall { suggested }) => {
                    println!("too small {:?}", suggested);
                    // Enlarge texture + glyph_brush texture cache and retry.

                    self.glyph_data = vec![0u8; (suggested.0 * suggested.1) as usize];
                    self.brush.resize_texture(suggested.0, suggested.1);
                }
            }
        }

        if glyph_updated {
            let (width, height) = self.brush.texture_dimensions();

            let (gi, gi_fut) = ImmutableImage::from_iter(
                self.glyph_data.iter().cloned(),
                vulkano::image::Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                vulkano::format::Format::R8Srgb,
                vk_state.queue.clone(),
            )
            .unwrap();
            if false {
                let image = ImageBuffer::<image::Luma<u8>, _>::from_raw(
                    width,
                    height,
                    &self.glyph_data[..],
                )
                .unwrap();
                image.save("font.png").unwrap();
            }
            self.glyph_image = Some(gi);
            Box::new(gi_fut)
        } else {
            Box::new(vulkano::sync::now(vk_state.device.clone()))
        }
    }

    pub fn render<PoolBuilder>(
        &mut self,
        builder: AutoCommandBufferBuilder<PoolBuilder>,
    ) -> Result<AutoCommandBufferBuilder<PoolBuilder>, render_bits::Error> {
        if let (Some(buffers), Some(glyph_image), Some(pipeline)) =
            (&self.buffers, &self.glyph_image, &self.pipeline)
        {
            let uniform_buffer_subbuffer = {
                let width = self.dimensions[0];
                let height = self.dimensions[1];

                let unity = Matrix4::from_scale(1f32);
                let uniform_data = vs::ty::Data {
                    view: unity.into(),
                    world: unity.into(),
                    proj: render_bits::math::canvas_projection(width, height).into(),
                };

                self.ub_pool.next(uniform_data).unwrap()
            };

            let set = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 0)
                    .add_buffer(uniform_buffer_subbuffer)
                    .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                        error: "failed to add buffer to descriptor set".into(),
                    })?
                    .add_sampled_image(glyph_image.clone(), self.sampler.clone())
                    .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                        error: "failed to add sampled image to descriptor set".into(),
                    })?
                    .build()
                    .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                        error: "failed to build descriptor set".into(),
                    })?,
            );

            Ok(builder
                .draw_indexed(
                    pipeline.clone(),
                    &DynamicState::none(),
                    vec![buffers.vb.clone(), buffers.tb.clone()],
                    buffers.ib.clone(),
                    set.clone(),
                    (),
                )
                .map_err(|_| render_bits::Error::UnhandledVulkanoError {
                    error: "draw indexed failed in TextConsole".into(),
                })?)
        } else {
            Ok(builder)
        }
    }

    pub fn get_sender(&self) -> Sender<String> {
        self.tx.clone()
    }

    pub fn get_input_sink(&self) -> Sender<InputEvent> {
        self.input_sink.clone()
    }

    pub fn set_input_line_sink(&mut self, sink: Sender<String>) {
        self.input_line_sink = Some(sink);
    }
}

pub struct TextConsoleLog {
    // the Mutex/RefCell/Option construct is only necessary to support setting sink on a TextConsoleLog with 'static lifetime...
    // otherwise a plain SyncSender would be sufficient
    pub sink: std::sync::Mutex<std::cell::RefCell<Option<Sender<String>>>>,
}
impl TextConsoleLog {
    pub fn empty() -> TextConsoleLog {
        TextConsoleLog {
            sink: std::sync::Mutex::new(std::cell::RefCell::new(None)),
        }
    }

    pub fn set_sink(&self, new_sink: Sender<String>) {
        if let Ok(sink_lock) = self.sink.lock() {
            *sink_lock.borrow_mut() = Some(new_sink);
        }
    }
}
impl Log for TextConsoleLog {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let mut fallback = true;
        if let Ok(sink_lock) = self.sink.lock() {
            if let Some(ref sink) = *sink_lock.borrow() {
                let msg = match record.level() {
                    log::Level::Info | log::Level::Trace | log::Level::Debug => {
                        format!("{}", record.args())
                    }
                    log::Level::Warn => format!("{} -- {}", record.level(), record.args()),
                    log::Level::Error => match (record.file(), record.line()) {
                        (Some(file), Some(line)) => {
                            format!("{} {}:{} -- {}", record.level(), file, line, record.args())
                        }
                        _ => format!("{} -- {}", record.level(), record.args()),
                    },
                };

                match sink.send(msg) {
                    Err(_) => fallback = true,
                    _ => fallback = false,
                };
            }
            // discard sink on send error
            if fallback {
                *sink_lock.borrow_mut() = None;
            }
        }
        if fallback {
            println!("{}", record.args())
        }
    }
    fn flush(&self) {}
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_bits/text_vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_bits/text_frag.glsl"
    }
}
