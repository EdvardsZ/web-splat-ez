use std::{
    io::{Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};
use std::sync::Mutex;

use image::Pixel;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::{util::DeviceExt, Backends, Extent3d};

use cgmath::{Deg, EuclideanSpace, Point3, Quaternion, UlpsEq, Vector2, Vector3};
use egui::FullOutput;
use num_traits::One;
use flume;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::prelude::*,
    wasm_bindgen_futures,
    gloo_timers,
};
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;
#[cfg(target_arch = "wasm32")]
use rfd::AsyncFileDialog;

mod animation;
mod ui;
pub use animation::{Animation, Sampler, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

pub mod io;

mod renderer;
pub use renderer::{GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

use cfg_if::cfg_if;

pub struct RenderConfig {
    pub no_vsync: bool,
    pub skybox: Option<PathBuf>,
    pub hdr: bool,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using {}", adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        max_storage_buffers_per_shader_stage: 12,
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..adapter_limits
                    },

                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..adapter_limits
                    },
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    scene_file_path: Option<PathBuf>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
    
    // Shared state for async operations like file loading
    shared_state: Arc<SharedState>,

    // File loading state
    loading_state: Option<LoadingState>,
}

// Structure to track the loading state of a new PLY file
pub struct LoadingState {
    pub file_path: PathBuf,
    pub progress: f32, // 0.0 to 1.0
    pub start_time: Instant,
    pub pc_raw: Option<io::GenericGaussianPointCloud>,
    pub pending_pc: Option<PointCloud>, // Hold parsed PC while renderer is created
}

// Shared state for communicating async results back to the main thread
struct SharedState {
    file_load_request: Mutex<Option<(Vec<u8>, String)>>, // (file_data, file_name)
    new_renderer_result: Mutex<Option<(GaussianRenderer, PointCloud)>>, // Holds both renderer and point cloud
    skybox_texture: Mutex<Option<wgpu::Texture>>, // For async skybox loading
    
    // Flume channels for async file loading
    pc_raw_sender: flume::Sender<(io::GenericGaussianPointCloud, PathBuf)>,
    pc_raw_receiver: flume::Receiver<(io::GenericGaussianPointCloud, PathBuf)>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();

        let render_format = if render_config.hdr {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc_raw = io::GenericGaussianPointCloud::load(pc_file)?;
        let pc = PointCloud::new(&device, pc_raw)?;
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed())
                .await;

        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            //aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
            Point3::new(0., 0., 0.),
            Quaternion::one(),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(90.), Deg(90. / aspect)),
                0.01,
                10.,
            ),
        );

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = pc.center();
        // controller.up = pc.up;
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );

        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                show_env_map: false,
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
                resolution: Vector2::new(size.width, size.height),
            },
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            display,
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            pointcloud_file_path: None,
            scene_file_path: None,

            stopwatch,
            
            // Shared state for async operations like file loading
            shared_state: Arc::new({
                let (pc_raw_sender, pc_raw_receiver) = flume::unbounded();
                SharedState {
                    file_load_request: Mutex::new(None),
                    new_renderer_result: Mutex::new(None),
                    skybox_texture: Mutex::new(None),
                    pc_raw_sender,
                    pc_raw_receiver,
                }
            }),

            // File loading state
            loading_state: None,
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        } else {
            return Err(anyhow::anyhow!("no pointcloud file path present"));
        }
        if let Some(scene_path) = &self.scene_file_path {
            log::info!("reloading scene from {:?}", scene_path);
            let file = std::fs::File::open(scene_path)?;

            self.set_scene(Scene::from_json(file)?);
        }
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration) -> anyhow::Result<()> {
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args
                    .camera
                    .projection
                    .resize(self.config.width, self.config.height);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.splatting_args.camera);
                }
            }
        } else {
            self.controller
                .update_camera(&mut self.splatting_args.camera, dt);

            // check if camera moved out of selected view
            if let Some(idx) = self.current_view {
                if let Some(scene) = &self.scene {
                    if let Some(camera) = scene.camera(idx) {
                        let scene_camera: PerspectiveCamera = camera.into();
                        if !self.splatting_args.camera.position.ulps_eq(
                            &scene_camera.position,
                            1e-4,
                            f32::default_max_ulps(),
                        ) || !self.splatting_args.camera.rotation.ulps_eq(
                            &scene_camera.rotation,
                            1e-4,
                            f32::default_max_ulps(),
                        ) {
                            self.current_view.take();
                        }
                    }
                }
            }
        }

        let aabb = self.pc.bbox();
        self.splatting_args.camera.fit_near_far(aabb);
        
        // Check for async skybox textures
        self.check_skybox_texture();
        
        // Check if async renderer creation has finished (WASM)
        if let Some((new_renderer, new_pc)) = self.shared_state.new_renderer_result.lock().unwrap().take() {
            if let Some(loading_state) = &mut self.loading_state {
                log::info!("New point cloud fully loaded and ready to apply");
                
                // Important: First request a redraw of the current point cloud
                // This ensures we have a fully rendered frame with the old point cloud
                self.window.request_redraw();
                
                // Make a backup of old camera parameters
                let old_camera_pos = self.splatting_args.camera.position;
                let old_camera_rot = self.splatting_args.camera.rotation;
                
                // Then in a single atomic operation, replace everything
                self.renderer = new_renderer;
                self.pc = new_pc;
                self.pointcloud_file_path = Some(loading_state.file_path.clone());
                
                // Adjust camera to view the new point cloud
                let aabb = self.pc.bbox();
                self.splatting_args.camera.fit_near_far(aabb);
                
                // Keep camera oriented in the same way
                self.controller.center = self.pc.center();
                
                // Keep looking at approximately the same view direction
                self.splatting_args.camera.position = old_camera_pos;
                self.splatting_args.camera.rotation = old_camera_rot;
                
                // Now clear the loading state to indicate completion
                self.loading_state = None;
                
                // Force an immediate redraw with the new point cloud
                // This will happen in the next frame, ensuring no lag is visible
                self.window.request_redraw();
                log::info!("New point cloud applied successfully");
            } else {
                log::warn!("Renderer and PointCloud results found, but loading state was already cleared.");
            }
        }

        // Check for pending file load requests (primarily from wasm async dialog)
        let file_request = self.shared_state.file_load_request.lock().unwrap().take();
        if let Some((data, name)) = file_request {
            if self.loading_state.is_none() {
                // Start loading from data
                log::info!("Starting to load new point cloud from uploaded file: {}", name);
                self.loading_state = Some(LoadingState {
                    file_path: PathBuf::from(&name), // Store filename as path for consistency (it's PathBuf)
                    progress: 0.0, // Start progress simulation
                    start_time: Instant::now(),
                    pc_raw: None,
                    pending_pc: None,
                });
                
                // Process the data in a background thread if on native platform
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let sender = self.shared_state.pc_raw_sender.clone();
                    let data_clone = data.clone();
                    let path = PathBuf::from(&name);
                    
                    std::thread::spawn(move || {
                        match io::GenericGaussianPointCloud::load(std::io::Cursor::new(data_clone)) {
                            Ok(pc_raw) => {
                                log::info!("Background thread: Parsed point cloud data with {} points", pc_raw.num_points);
                                if let Err(e) = sender.send((pc_raw, path)) {
                                    log::error!("Failed to send parsed point cloud through channel: {:?}", e);
                                }
                            },
                            Err(e) => {
                                log::error!("Background thread: Failed to parse point cloud: {:?}", e);
                            }
                        }
                    });
                    
                    // Set initial progress
                    if let Some(ls) = &mut self.loading_state {
                        ls.progress = 0.3; // Show that parsing has started
                    }
                }
                
                // For WASM, parse directly here since there's no true multithreading
                #[cfg(target_arch = "wasm32")]
                {
                    match io::GenericGaussianPointCloud::load(std::io::Cursor::new(data)) {
                        Ok(pc_raw) => {
                            if let Some(ls) = &mut self.loading_state {
                                log::info!("Successfully parsed point cloud data for {}", name);
                                ls.pc_raw = Some(pc_raw);
                                ls.progress = 1.0; // Mark as ready to be applied immediately
                            }
                        },
                        Err(e) => {
                            log::error!("Failed to parse point cloud from uploaded data: {:?}", e);
                            self.loading_state = None; // Abort loading
                        }
                    }
                }
            } else {
                log::warn!("Ignoring file load request while another load is in progress.");
            }
        }

        // Update file loading if in progress
        let loading_needs_redraw = if self.loading_state.is_some() {
            if let Err(e) = self.update_loading() {
                log::error!("Error during update_loading: {:?}", e);
                // Optionally clear loading state on error
                self.loading_state = None;
                false // Don't signal redraw if update_loading failed catastrophically
            } else {
                true // Loading is in progress (or just finished this cycle)
            }
        } else {
            false
        };

        // Always request redraw while loading is in progress
        // This keeps the UI responsive and continues to display the current scene
        if self.loading_state.is_some() || loading_needs_redraw {
            self.window.request_redraw();
            
            // Add a small yield to allow the UI to render between loading steps
            #[cfg(target_arch = "wasm32")]
            {
                // Use a synchronous timeout to yield to the browser without awaiting
                gloo_timers::callback::Timeout::new(5, || {
                    // Empty callback, just letting browser do some work
                }).forget();
            }
        }
        
        Ok(())
    }

    fn render(
        &mut self,
        redraw_scene: bool,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::SurfaceError> {
        // Skip performance measurement during loading to avoid overhead
        let is_loading = self.loading_state.is_some();
        if !is_loading {
            self.stopwatch.as_mut().map(|s| s.reset());
        }

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        // Decide whether to render the scene
        let should_render_scene = redraw_scene || is_loading;
        
        if should_render_scene {
            // Prepare rendering with simpler path during loading for better performance
            if is_loading {
                // Use a simplified rendering path for loading state (fewer features)
                self.renderer.prepare(
                    &mut encoder,
                    &self.wgpu_context.device,
                    &self.wgpu_context.queue,
                    &self.pc,
                    self.splatting_args.clone(), // Clone to avoid borrow issues
                    &mut None, // Skip stopwatch during loading
                );
            } else {
                // Normal rendering path with all features
                self.renderer.prepare(
                    &mut encoder,
                    &self.wgpu_context.device,
                    &self.wgpu_context.queue,
                    &self.pc,
                    self.splatting_args.clone(), // Clone to avoid borrow issues
                    (&mut self.stopwatch).into(),
                );
            }
        }

        // Always prepare UI to maintain responsiveness
        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        // Skip performance measurement during loading
        if !is_loading && self.stopwatch.is_some() {
            self.stopwatch.as_mut().unwrap().start(&mut encoder, "rasterization").unwrap();
        }
        
        // Render the scene if needed
        if should_render_scene {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.display.texture(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.splatting_args.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }
        
        // Skip performance measurement during loading
        if !is_loading && self.stopwatch.is_some() {
            self.stopwatch.as_mut().unwrap().stop(&mut encoder, "rasterization").unwrap();
        }

        // Always render the display
        self.display.render(
            &mut encoder,
            &view_rgb,
            self.splatting_args.background_color,
            self.renderer.camera(),
            &self.renderer.render_settings(),
        );
        
        // Skip unnecessary measurement during loading
        if !is_loading {
            self.stopwatch.as_mut().map(|s| s.end(&mut encoder));
        }

        // Always render the UI to keep it responsive
        if let Some(state) = &ui_state {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_srgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }

        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        
        // Optimize command submission for web
        #[cfg(target_arch = "wasm32")]
        {
            if is_loading {
                // Use async submission during loading to keep UI responsive
                let queue = self.wgpu_context.queue.clone();
                let encoder_finished = encoder.finish();
                
                wasm_bindgen_futures::spawn_local(async move {
                    // Short yield to browser before GPU submission
                    gloo_timers::future::TimeoutFuture::new(1).await;
                    queue.submit(Some(encoder_finished));
                });
            } else {
                // Normal submission path
                self.wgpu_context.queue.submit(Some(encoder.finish()));
            }
        }
        
        // Normal submission for native platforms
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.wgpu_context.queue.submit(Some(encoder.finish()));
        }

        output.present();
        self.splatting_args.resolution = Vector2::new(self.config.width, self.config.height);
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.splatting_args.scene_extend = Some(scene.extend());
        let mut center = Point3::origin();
        for c in scene.cameras(None) {
            let z_axis: Vector3<f32> = c.rotation[2].into();
            center += Vector3::from(c.position) + z_axis * 2.;
        }
        center /= scene.num_cameras() as f32;

        self.controller.center = center;
        self.scene.replace(scene);
        if self.saved_cameras.is_empty() {
            self.saved_cameras = self
                .scene
                .as_ref()
                .unwrap()
                .cameras(Some(Split::Test))
                .clone();
        }
    }

    fn set_env_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let env_map_exr = image::open(path)?;
        let env_map_data: Vec<[f32; 4]> = env_map_exr
            .as_rgb32f()
            .ok_or(anyhow::anyhow!("env map must be rgb"))?
            .pixels()
            .map(|p| p.to_rgba().0)
            .collect();

        let env_texture = self.wgpu_context.device.create_texture_with_data(
            &self.wgpu_context.queue,
            &wgpu::TextureDescriptor {
                label: Some("env map texture"),
                size: Extent3d {
                    width: env_map_exr.width(),
                    height: env_map_exr.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&env_map_data.as_slice()),
        );
        self.display.set_env_map(
            &self.wgpu_context.device,
            Some(&env_texture.create_view(&Default::default())),
        );
        self.splatting_args.show_env_map = true;
        Ok(())
    }

    fn start_tracking_shot(&mut self) {
        if self.saved_cameras.len() > 1 {
            let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
            let a = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 2.),
                true,
                Box::new(shot),
            );
            self.animation = Some((a, true));
        }
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn set_scene_camera(&mut self, i: usize) {
        if let Some(scene) = &self.scene {
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            if let Some(camera) = scene.camera(i) {
                self.set_camera(camera, Duration::from_millis(200));
            } else {
                log::error!("camera {i} not found");
            }
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        let camera: PerspectiveCamera = camera.into();
        if animation_duration.is_zero() {
            self.update_camera(camera.into())
        } else {
            let target_camera = camera.into();
            let a = Animation::new(
                animation_duration,
                false,
                Box::new(Transition::new(
                    self.splatting_args.camera.clone(),
                    target_camera,
                    smoothstep,
                )),
            );
            self.animation = Some((a, true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.splatting_args.camera = camera;
        self.splatting_args
            .camera
            .projection
            .resize(self.config.width, self.config.height);
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) + 1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.splatting_args.camera,
            id.to_string(),
            id,
            Vector2::new(self.config.width, self.config.height),
            Split::Test,
        ));
    }

    // Start loading a new PLY file while continuing to display the current one
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_new_file<P: AsRef<Path>>(&mut self, file_path: P) -> anyhow::Result<()> {
        let path = file_path.as_ref().to_path_buf();
        
        // Don't start loading if already loading or if it's the same file
        if let Some(state) = &self.loading_state {
            if state.file_path == path {
                log::warn!("Attempted to load the same file again: {:?}", path);
                return Ok(());
            }
        }
        
        log::info!("Starting to load new point cloud from {:?}", path);
        self.loading_state = Some(LoadingState {
            file_path: path.clone(),
            progress: 0.0,
            start_time: Instant::now(),
            pc_raw: None,
            pending_pc: None,
        });
        
        // Launch background thread using flume channel
        let sender = self.shared_state.pc_raw_sender.clone();
        std::thread::spawn(move || {
            match std::fs::File::open(&path) {
                Ok(file) => {
                    match io::GenericGaussianPointCloud::load(file) {
                        Ok(pc_raw) => {
                            log::info!("Background thread: Successfully loaded {:?} with {} points", 
                                      path, pc_raw.num_points);
                            if let Err(e) = sender.send((pc_raw, path)) {
                                log::error!("Failed to send point cloud through channel: {:?}", e);
                            }
                        },
                        Err(e) => {
                            log::error!("Background thread: Failed to load point cloud: {:?}", e);
                        }
                    }
                },
                Err(e) => {
                    log::error!("Background thread: Failed to open file: {:?}", e);
                }
            }
        });
        
        Ok(())
    }
    
    // Updates the loading progress and completes the loading if necessary
    #[cfg(not(target_arch = "wasm32"))]
    fn update_loading(&mut self) -> anyhow::Result<()> {
        if let Some(loading_state) = &mut self.loading_state {
            // We have a pending PointCloud that needs a renderer
            if let Some(new_pc) = loading_state.pending_pc.take() {
                log::info!("Creating new renderer for point cloud with {} points", new_pc.num_points());
                
                // Request redraw of current scene while we work
                self.window.request_redraw();
                
                // Make a backup of the current camera parameters
                let old_camera_pos = self.splatting_args.camera.position;
                let old_camera_rot = self.splatting_args.camera.rotation;
                
                // Create new renderer asynchronously on a separate thread
                let device = self.wgpu_context.device.clone();
                let queue = self.wgpu_context.queue.clone();
                let color_format = self.renderer.color_format();
                let sh_deg = new_pc.sh_deg();
                let compressed = new_pc.compressed();
                let shared_state = self.shared_state.clone();
                
                // Move the point cloud to the thread
                std::thread::spawn(move || {
                    // Create the renderer in the background
                    let renderer = pollster::block_on(
                        GaussianRenderer::new(
                            &device,
                            &queue,
                            color_format,
                            sh_deg,
                            compressed
                        )
                    );
                    
                    // Store results in shared state
                    let mut renderer_guard = shared_state.new_renderer_result.lock().unwrap();
                    *renderer_guard = Some((renderer, new_pc));
                });
                
                // Update progress to indicate renderer creation in progress
                loading_state.progress = 0.9;
                
                // Force redraw to show loading indicator
                self.window.request_redraw();
                
            // We have PC data, need to create a PointCloud
            } else if let Some(pc_raw) = loading_state.pc_raw.take() {
                log::info!("Creating new point cloud with {} points", pc_raw.num_points);
                
                // Create new point cloud
                match PointCloud::new(&self.wgpu_context.device, pc_raw) {
                    Ok(new_pc) => {
                        // Store in pending_pc for safe keeping until we're ready to swap
                        loading_state.pending_pc = Some(new_pc);
                        loading_state.progress = 0.8; // Almost done
                        
                        // Force redraw to update loading progress
                        self.window.request_redraw();
                    },
                    Err(e) => {
                        log::error!("Failed to create point cloud: {:?}", e);
                        self.loading_state = None;
                        return Err(e);
                    }
                }
            } else if loading_state.progress < 0.7 {
                // Read file data in background thread using proper channel communication
                let file_path = loading_state.file_path.clone();
                let sender = self.shared_state.pc_raw_sender.clone();
                
                log::info!("Starting background file load for: {:?}", file_path);
                
                // Spawn thread for actual file loading
                std::thread::spawn(move || {
                    match std::fs::File::open(&file_path) {
                        Ok(file) => {
                            match io::GenericGaussianPointCloud::load(file) {
                                Ok(pc_raw) => {
                                    log::info!("Background thread: Loaded point cloud with {} points", pc_raw.num_points);
                                    // Send directly through the channel
                                    if let Err(e) = sender.send((pc_raw, file_path)) {
                                        log::error!("Failed to send point cloud through channel: {:?}", e);
                                    }
                                },
                                Err(e) => {
                                    log::error!("Background thread: Failed to load point cloud: {:?}", e);
                                }
                            }
                        },
                        Err(e) => {
                            log::error!("Background thread: Failed to open file: {:?}", e);
                        }
                    }
                });
                
                // Set progress to indicate file loading in progress
                loading_state.progress = 0.7;
                
                // Force redraw to show updated progress
                self.window.request_redraw();
            }
            
            // Check if any point cloud data has been received through the channel
            if let Ok((pc_raw, path)) = self.shared_state.pc_raw_receiver.try_recv() {
                if let Some(loading_state) = &mut self.loading_state {
                    if loading_state.file_path == path {
                        log::info!("Received point cloud data through channel for: {:?}", path);
                        loading_state.pc_raw = Some(pc_raw);
                        loading_state.progress = 0.75;
                        self.window.request_redraw();
                    } else {
                        log::warn!("Received point cloud data for wrong path: expected {:?}, got {:?}", 
                            loading_state.file_path, path);
                    }
                }
            }
        }
        
        // Check for new renderer + pointcloud from background thread
        if let Some((new_renderer, new_pc)) = self.shared_state.new_renderer_result.lock().unwrap().take() {
            if let Some(loading_state) = &mut self.loading_state {
                log::info!("New point cloud fully loaded and ready to apply");
                
                // Grab camera parameters before swapping
                let old_camera_pos = self.splatting_args.camera.position;
                let old_camera_rot = self.splatting_args.camera.rotation;
                
                // Swap everything in one atomic operation
                self.renderer = new_renderer;
                self.pc = new_pc;
                self.pointcloud_file_path = Some(loading_state.file_path.clone());
                
                // Adjust camera to view the new point cloud
                let aabb = self.pc.bbox();
                self.splatting_args.camera.fit_near_far(aabb);
                self.controller.center = self.pc.center();
                
                // Preserve camera orientation
                self.splatting_args.camera.position = old_camera_pos;
                self.splatting_args.camera.rotation = old_camera_rot;
                
                // Clear loading state
                self.loading_state = None;
                
                // Force redraw with new point cloud
                self.window.request_redraw();
                log::info!("New point cloud applied successfully");
            }
        }
        
        Ok(())
    }

    // WASM version of update_loading: Spawns async renderer creation
    #[cfg(target_arch = "wasm32")]
    fn update_loading(&mut self) -> anyhow::Result<()> {
        use wasm_bindgen_futures::spawn_local;

        if let Some(loading_state) = &mut self.loading_state {
            // Check if pc_raw is ready from parsing in update()
            if let Some(pc_raw) = loading_state.pc_raw.take() {
                log::info!("WASM: Parsed data ready, creating PointCloud asynchronously...");
                
                // Clone necessary data for the async task
                let device = self.wgpu_context.device.clone();
                let queue = self.wgpu_context.queue.clone();
                let color_format = self.renderer.color_format();
                let shared_state_clone = self.shared_state.clone();
                let splatting_args = self.splatting_args.clone();
                
                // Move pc_raw into async task and perform full loading in background
                spawn_local(async move {
                    log::info!("WASM: Async task started: Creating PointCloud...");
                    
                    // Allow UI to update before starting intensive work
                    gloo_timers::future::TimeoutFuture::new(10).await;
                    
                    // Create PointCloud in a separate task to avoid blocking the UI
                    let new_pc = match PointCloud::new(&device, pc_raw) {
                        Ok(pc) => {
                            log::info!("WASM: PointCloud created successfully ({} points)", pc.num_points());
                            pc
                        },
                        Err(e) => {
                            log::error!("WASM: Failed to create PointCloud: {:?}", e);
                            return; // Exit the async task
                        }
                    };
                    
                    // Yield back to the browser to ensure UI responsiveness
                    gloo_timers::future::TimeoutFuture::new(20).await;
                    
                    let sh_deg = new_pc.sh_deg();
                    let compressed = new_pc.compressed();
                    
                    log::info!("WASM: Creating GaussianRenderer...");
                    
                    // Create the renderer asynchronously - make mutable
                    let mut renderer = GaussianRenderer::new(
                        &device,
                        &queue,
                        color_format,
                        sh_deg,
                        compressed
                    ).await;
                    
                    // Let browser process events
                    gloo_timers::future::TimeoutFuture::new(20).await;
                    
                    // Critical step: Pre-initialize the renderer with the new point cloud
                    // This ensures the first frame with the new point cloud is fully prepared
                    log::info!("WASM: Pre-initializing renderer for first frame...");
                    
                    // Create a command encoder for initialization
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Pre-initialization command encoder"),
                    });
                    
                    // Prepare the renderer with the new point cloud (this avoids the lag on first use)
                    renderer.prepare(
                        &mut encoder,
                        &device,
                        &queue,
                        &new_pc,
                        splatting_args.clone(),
                        &mut None
                    );
                    
                    // Submit the initialization commands to the GPU
                    queue.submit(Some(encoder.finish()));
                    
                    // Let browser process events one more time
                    gloo_timers::future::TimeoutFuture::new(10).await;
                    
                    log::info!("WASM: Async task complete: PointCloud and Renderer fully initialized and ready.");
                    
                    // Put both results in shared state, only when completely ready
                    let mut renderer_guard = shared_state_clone.new_renderer_result.lock().unwrap();
                    *renderer_guard = Some((renderer, new_pc));
                });
                
                // Set progress to indicate an active background operation
                loading_state.progress = 0.5;
                loading_state.start_time = Instant::now(); // Reset timer for progress estimation
            } else if loading_state.progress < 1.0 {
                // Calculate elapsed time for better progress estimation
                let elapsed = Instant::now().duration_since(loading_state.start_time);
                let elapsed_secs = elapsed.as_secs_f32();
                
                // Create a more realistic progress indicator based on time and phase
                if loading_state.progress < 0.1 {
                    // Initial phase: Just started
                    loading_state.progress = (elapsed_secs / 1.0).min(0.1);
                } else if loading_state.progress < 0.5 {
                    // Either waiting for parsing or in PointCloud creation
                    loading_state.progress = (0.1 + (elapsed_secs / 3.0) * 0.4).min(0.5);
                } else if loading_state.progress < 0.95 {
                    // PointCloud created, renderer being made
                    if elapsed_secs < 2.0 {
                        // Creating renderer (up to 70%)
                        loading_state.progress = (0.5 + (elapsed_secs / 2.0) * 0.2).min(0.7); 
                    } else if elapsed_secs < 4.0 {
                        // Renderer created, preparing (up to 90%)
                        loading_state.progress = (0.7 + ((elapsed_secs - 2.0) / 2.0) * 0.2).min(0.9);
                    } else {
                        // Final initialization (up to 95%)
                        loading_state.progress = (0.9 + ((elapsed_secs - 4.0) / 2.0) * 0.05).min(0.95);
                    }
                }
                
                // Keep rendering while loading
                self.window.request_redraw();
            }
        }
        Ok(())
    }

    // Method to check for and apply pending skybox texture
    fn check_skybox_texture(&mut self) {
        if let Some(texture) = self.shared_state.skybox_texture.lock().unwrap().take() {
            log::info!("Applying async loaded skybox texture");
            self.display.set_env_map(
                &self.wgpu_context.device,
                Some(&texture.create_view(&Default::default())),
            );
            self.splatting_args.show_env_map = true;
        }
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    // Start loading scene in background
    let scene_future = async {
        if let Some(scene_file) = scene_file {
            match Scene::from_json(scene_file) {
                Ok(s) => Some(s),
                Err(err) => {
                    log::error!("cannot load scene: {:?}", err);
                    None
                }
            }
        } else {
            None
        }
    };

    let window_size = LogicalSize::new(800, 600);
    let window_attributes = Window::default_attributes()
        .with_inner_size(window_size)
        .with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));
        
    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                doc.body()
            })
            .and_then(|body| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                canvas.set_width(body.client_width() as u32);
                canvas.set_height(body.client_height() as u32);
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%;")
                    .unwrap();
                body.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to document body");
    }

    // limit the redraw rate to the monitor refresh rate
    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    // Create window context first
    let mut state = WindowContext::new(window, file, &config).await.unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    // Load scene (this was done in parallel, just use result)
    let scene = scene_future.await;
    if let Some(scene) = scene {
        state.set_scene(scene);
        state.set_scene_camera(0);
        state.scene_file_path = scene_file_path;
    }

    // Load skybox if provided
    if let Some(skybox) = &config.skybox {
        #[cfg(target_arch = "wasm32")]
        {
            // For web, load asynchronously
            let skybox_path = skybox.clone();
            let device = state.wgpu_context.device.clone();
            let queue = state.wgpu_context.queue.clone();
            let shared_state = state.shared_state.clone();
            
            wasm_bindgen_futures::spawn_local(async move {
                match load_skybox_async(&skybox_path, &device, &queue).await {
                    Ok(texture) => {
                        log::info!("Skybox loaded asynchronously");
                        let mut skybox_guard = shared_state.skybox_texture.lock().unwrap();
                        *skybox_guard = Some(texture);
                    },
                    Err(e) => {
                        log::error!("Failed to load skybox: {:?}", e);
                    }
                }
            });
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = state.set_env_map(skybox.as_path()) {
            log::error!("Failed to set skybox: {e}");
        }
    }

    #[cfg(target_arch = "wasm32")]
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.get_element_by_id("spinner")
                .unwrap()
                .set_attribute("style", "display:none;")
                .unwrap();
            doc.body()
        });

    let mut last = Instant::now();

    #[allow(deprecated)]
    event_loop.run(move |event, target| 
        match event {
            Event::NewEvents(e) => match e {
                winit::event::StartCause::ResumeTimeReached { .. } => {
                    state.window.request_redraw();
                }
                _ => {}
            },
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window, event) => match event {
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size, None);
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    ..
                } => {
                    state.scale_factor = *scale_factor as f32;
                }
                WindowEvent::CloseRequested => {log::info!("close!"); target.exit()},
                WindowEvent::ModifiersChanged(m) => {
                    state.controller.alt_pressed = m.state().alt_key();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if event.state == ElementState::Released {
                            if key == KeyCode::KeyT {
                                if state.animation.is_none() {
                                    state.start_tracking_shot();
                                } else {
                                    state.stop_animation()
                                }
                            } else if key == KeyCode::KeyU {
                                state.ui_visible = !state.ui_visible;
                                
                            } else if key == KeyCode::KeyC {
                                state.save_view();
                            } else if key == KeyCode::KeyO {
                                // Add a keyboard shortcut for opening a PLY file
                                cfg_if! {
                                    if #[cfg(target_arch = "wasm32")] {
                                        use wasm_bindgen_futures::spawn_local;
                                        if state.loading_state.is_none() {
                                            log::info!("Opening file dialog...");
                                            let shared_state_clone = state.shared_state.clone();
                                            spawn_local(async move {
                                                if let Some(file_handle) = AsyncFileDialog::new()
                                                    .add_filter("PLY Files", &["ply"])
                                                    .pick_file()
                                                    .await
                                                {
                                                    log::info!("File selected, reading data...");
                                                    let data = file_handle.read().await;
                                                    let name = file_handle.file_name();
                                                    log::info!("Read {} bytes from {}", data.len(), name);
                                                    let mut request = shared_state_clone.file_load_request.lock().unwrap();
                                                    *request = Some((data, name));
                                                    // Request redraw implicitly handled by checking state in update loop
                                                } else {
                                                    log::info!("File selection cancelled.");
                                                }
                                            });
                                        } else {
                                            log::warn!("Already loading a file, cannot open another.");
                                        }
                                    } else {
                                        // Native platform: use synchronous dialog
                                        if state.loading_state.is_none() {
                                            if let Some(path) = FileDialog::new()
                                                .add_filter("PLY Files", &["ply"])
                                                .pick_file()
                                            {
                                                if let Err(err) = state.load_new_file(&path) {
                                                    log::error!("failed to load file: {:?}", err);
                                                }
                                            }
                                        } else {
                                            log::warn!("Already loading a file, cannot open another.");
                                        }
                                    }
                                }
                            } else if key == KeyCode::KeyR && state.controller.alt_pressed {
                                if let Err(err) = state.reload() {
                                    log::error!("failed to reload volume: {:?}", err);
                                }   
                            } else if let Some(scene) = &state.scene {
                                let new_camera = 
                                    if let Some(num) = key_to_num(key) {
                                        Some(num as usize)
                                    }
                                    else if key == KeyCode::KeyR {
                                        Some((rand::random::<u32>() as usize) % scene.num_cameras())
                                    } else if key == KeyCode::KeyN {
                                        scene.nearest_camera(state.splatting_args.camera.position,None)
                                    } else if key == KeyCode::PageUp {
                                        Some(state.current_view.map_or(0, |v| v+1) % scene.num_cameras())
                                    } else if key == KeyCode::KeyT {
                                        Some(state.current_view.map_or(0, |v| v+1) % scene.num_cameras())
                                    }
                                    else if key == KeyCode::PageDown {
                                        Some(state.current_view.map_or(0, |v| v-1) % scene.num_cameras())
                                    } else {
                                        None
                                    };

                                if let Some(new_camera) = new_camera {
                                    state.set_scene_camera(new_camera);
                                }
                            }
                        }
                        state.controller.process_keyboard(key, event.state == ElementState::Pressed);
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                        state.controller.process_scroll(*dy)
                    }
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        state.controller.process_scroll(p.y as f32 / 100.)
                    }
                },
                WindowEvent::MouseInput { state: button_state, button, .. } => {
                    match button {
                        winit::event::MouseButton::Left => state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                        winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !config.no_vsync {
                        // make sure the next redraw is called with a small delay
                        target.set_control_flow(ControlFlow::wait_duration(min_wait));
                    }
                    let now = Instant::now();
                    let dt = now-last;
                    last = now;

                    let old_settings = state.splatting_args.clone();
                    let old_loading_state = state.loading_state.is_some();
                    state.update(dt).unwrap();
                    let new_loading_state = state.loading_state.is_some();

                    let (redraw_ui, shapes) = state.ui();

                    let resolution_change = state.splatting_args.resolution != Vector2::new(state.config.width, state.config.height);

                    // Check if any state has changed that requires a redraw
                    let request_redraw = old_settings != state.splatting_args 
                        || resolution_change 
                        || old_loading_state != new_loading_state
                        || state.loading_state.is_some(); // Always redraw while loading
        
                    if request_redraw || redraw_ui {
                        // During loading, prioritize smooth UI updates by not calculating FPS
                        // which can cause stuttering due to smoothing calculations
                        if !state.loading_state.is_some() {
                            state.fps = (1. / dt.as_secs_f32()) * 0.05 + state.fps * 0.95;
                        }
                        
                        match state.render(request_redraw, state.ui_visible.then_some(shapes)) {
                            Ok(_) => {}
                            // Reconfigure the surface if lost
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => println!("error: {:?}", e),
                        }
                    }
                    
                    // Always request a redraw when loading a file to keep progress bar updating and scene rendering
                    if state.loading_state.is_some() || config.no_vsync {
                        state.window.request_redraw();
                        
                        // Set polling mode during loading for maximum responsiveness
                        if state.loading_state.is_some() {
                            target.set_control_flow(ControlFlow::Poll);
                        }
                    }
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => {
                state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
            }
            _ => {},
        }
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(
    pc: Vec<u8>,
    scene: Option<Vec<u8>>,
    pc_file: Option<String>,
    scene_file: Option<String>,
) {
    use std::{io::Cursor, str::FromStr};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig {
            no_vsync: false,
            skybox: None,
            hdr: false,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
    ));
}

#[cfg(target_arch = "wasm32")]
async fn load_skybox_async<P: AsRef<Path>>(
    path: P,
    device: &wgpu::Device, 
    queue: &wgpu::Queue
) -> Result<wgpu::Texture, anyhow::Error> {
    // On web, we would need to fetch the image using web APIs
    // For now, let's simulate this with a delay
    gloo_timers::future::TimeoutFuture::new(100).await;
    
    // In reality, you'd use browser APIs through web_sys to fetch and load the image
    // Here's a stub for the real implementation
    let env_map_exr = image::open(path)?;
    let env_map_data: Vec<[f32; 4]> = env_map_exr
        .as_rgb32f()
        .ok_or(anyhow::anyhow!("env map must be rgb"))?
        .pixels()
        .map(|p| p.to_rgba().0)
        .collect();

    let env_texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("env map texture"),
            size: Extent3d {
                width: env_map_exr.width(),
                height: env_map_exr.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&env_map_data.as_slice()),
    );
    
    Ok(env_texture)
}
