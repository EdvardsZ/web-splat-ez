use std::sync::Mutex;
use wgpu::Texture;
use flume;

use crate::io;
use std::path::PathBuf;
use crate::state::WebSocketState;

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
    
    /// Storage for WebSocket connection state
    pub websocket_state: Mutex<WebSocketState>,
    
    /// Channel for receiving WebSocket-fetched Gaussian data
    pub websocket_data_sender: flume::Sender<Vec<u8>>,
    
    /// Channel for receiving WebSocket-fetched Gaussian data
    pub websocket_data_receiver: flume::Receiver<Vec<u8>>,
    
    /// Flag to trigger a WebSocket refresh
    pub websocket_refresh_requested: Mutex<bool>,
}

impl SharedState {
    /// Create a new SharedState instance with initialized channels
    pub fn new() -> Self {
        let (pc_raw_sender, pc_raw_receiver) = flume::unbounded();
        let (websocket_data_sender, websocket_data_receiver) = flume::unbounded();
        
        Self {
            file_load_request: Mutex::new(None),
            new_renderer_result: Mutex::new(None),
            skybox_texture: Mutex::new(None),
            pc_raw_sender,
            pc_raw_receiver,
            websocket_state: Mutex::new(WebSocketState::new("ws://localhost:8765".to_string())),
            websocket_data_sender,
            websocket_data_receiver,
            websocket_refresh_requested: Mutex::new(false),
        }
    }
} 