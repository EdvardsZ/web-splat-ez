use std::time::Instant;

/// WebSocket connection state
#[derive(Clone, Debug, PartialEq)]
pub enum ConnectionState {
    /// Not connected to a WebSocket server
    Disconnected,
    /// Currently attempting to connect
    Connecting,
    /// Connected and waiting for data
    Connected,
    /// Actively receiving data
    Receiving,
    /// Connection error occurred
    Error(String),
}

/// Structure to track the state of WebSocket connections for live updates
pub struct WebSocketState {
    /// Current connection state
    pub connection_state: ConnectionState,
    /// Last connection attempt time
    pub last_connection_attempt: Option<Instant>,
    /// URL of the WebSocket server
    pub server_url: String,
    /// Progress of current data transfer (0.0-1.0)
    pub transfer_progress: f32,
    /// Timestamp of last successful update
    pub last_update_time: Option<Instant>,
    /// Whether auto-refresh is enabled
    pub auto_refresh: bool,
    /// Auto-refresh interval in seconds
    pub refresh_interval: u64,
    /// Number of gaussians received in last update
    pub points_received: Option<usize>,
}

impl WebSocketState {
    /// Create a new WebSocket state
    pub fn new(server_url: String) -> Self {
        Self {
            connection_state: ConnectionState::Disconnected,
            last_connection_attempt: None,
            server_url,
            transfer_progress: 0.0,
            last_update_time: None,
            auto_refresh: false,
            refresh_interval: 5,
            points_received: None,
        }
    }
    
    /// Check if a connection is active
    pub fn is_connected(&self) -> bool {
        matches!(self.connection_state, ConnectionState::Connected | ConnectionState::Receiving)
    }
    
    /// Check if we should try auto-refresh
    pub fn should_auto_refresh(&self) -> bool {
        if !self.auto_refresh || !matches!(self.connection_state, ConnectionState::Connected) {
            return false;
        }
        
        if let Some(last_update) = self.last_update_time {
            let elapsed = Instant::now().duration_since(last_update);
            return elapsed.as_secs() >= self.refresh_interval;
        }
        
        false
    }
    
    /// Update connection state
    pub fn set_state(&mut self, state: ConnectionState) {
        // Set the timestamp if connecting
        if let ConnectionState::Connecting = state {
            self.last_connection_attempt = Some(Instant::now());
        } else if let ConnectionState::Connected = state {
            self.transfer_progress = 0.0;
        }
        
        // Update the state last, after checking the old value
        self.connection_state = state;
    }
    
    /// Update transfer progress
    pub fn update_progress(&mut self, progress: f32) {
        self.transfer_progress = progress;
        
        if progress >= 1.0 {
            self.last_update_time = Some(Instant::now());
        }
    }
} 