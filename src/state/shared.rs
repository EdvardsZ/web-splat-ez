use std::sync::Mutex;
use wgpu::Texture;
use flume;

use crate::io;
use std::path::PathBuf;

/// Shared state for communicating async results back to the main thread
pub struct SharedState {
    /// Storage for a pending file load request containing file data and name
    pub file_load_request: Mutex<Option<(Vec<u8>, String)>>,
    
    /// Storage for a new renderer and point cloud when async creation completes
    pub new_renderer_result: Mutex<Option<(crate::GaussianRenderer, crate::PointCloud)>>,
    
    /// Storage for an asynchronously loaded skybox texture
    pub skybox_texture: Mutex<Option<Texture>>,
    
    /// Flume channel sender for async file loading
    pub pc_raw_sender: flume::Sender<(io::GenericGaussianPointCloud, PathBuf)>,
    
    /// Flume channel receiver for async file loading
    pub pc_raw_receiver: flume::Receiver<(io::GenericGaussianPointCloud, PathBuf)>,
}

impl SharedState {
    /// Create a new SharedState instance with initialized channels
    pub fn new() -> Self {
        let (pc_raw_sender, pc_raw_receiver) = flume::unbounded();
        
        Self {
            file_load_request: Mutex::new(None),
            new_renderer_result: Mutex::new(None),
            skybox_texture: Mutex::new(None),
            pc_raw_sender,
            pc_raw_receiver,
        }
    }
} 