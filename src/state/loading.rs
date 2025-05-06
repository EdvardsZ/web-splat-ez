use std::path::PathBuf;
use std::time::Instant;

use crate::io;
use crate::PointCloud;

/// Structure to track the loading state of a new PLY file
pub struct LoadingState {
    /// Path to the file being loaded
    pub file_path: PathBuf,
    
    /// Loading progress from 0.0 to 1.0
    pub progress: f32,
    
    /// When the loading operation started
    pub start_time: Instant,
    
    /// The parsed raw point cloud data, if available
    pub pc_raw: Option<io::GenericGaussianPointCloud>,
    
    /// The created PointCloud, waiting for renderer initialization
    pub pending_pc: Option<PointCloud>,
}

impl LoadingState {
    /// Create a new loading state for a file
    pub fn new(file_path: PathBuf) -> Self {
        Self {
            file_path,
            progress: 0.0,
            start_time: Instant::now(),
            pc_raw: None,
            pending_pc: None,
        }
    }
    
    /// Check if the loading has a parsed point cloud
    pub fn has_raw_data(&self) -> bool {
        self.pc_raw.is_some()
    }
    
    /// Check if the loading has a pending point cloud
    pub fn has_pending_pc(&self) -> bool {
        self.pending_pc.is_some()
    }
    
    /// Update the progress value
    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }
} 