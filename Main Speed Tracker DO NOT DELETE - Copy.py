import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from collections import deque
import time
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import platform
from abc import ABC, abstractmethod
import json
import csv

# Try to import PIL
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL (Pillow) not installed. Install with: pip install pillow")

# Try to import Matplotlib for graphing
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Graphing features will be disabled.")
    print("Install with: pip install matplotlib")

# Check for GPU support
GPU_AVAILABLE = False
CUDA_AVAILABLE = False
try:
    # Check if CUDA is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        GPU_AVAILABLE = True
        print(f"CUDA GPU detected: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
            cv2.cuda.printCudaDeviceInfo(i)
except:
    CUDA_AVAILABLE = False

# Check for OpenCL support (fallback for AMD GPUs)
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        GPU_AVAILABLE = True
        print("OpenCL GPU acceleration available")
except:
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get CPU core count
CPU_COUNT = mp.cpu_count()
print(f"CPU cores detected: {CPU_COUNT}")
print(f"Platform: {platform.system()} {platform.machine()}")

@dataclass
class SpeedMeasurement:
    """Data class for speed measurements"""
    mps: float = 0.0
    kph: float = 0.0
    mph: float = 0.0
    knots: float = 0.0
    mach: float = 0.0
    
    @classmethod
    def from_mps(cls, mps: float):
        return cls(
            mps=mps,
            kph=mps * 3.6,
            mph=mps * 2.237,
            knots=mps * 1.944,
            mach=mps / 343.0  # Speed of sound at sea level
        )

class ThreadSafeConfig:
    """Thread-safe configuration dictionary"""
    def __init__(self, initial_config):
        self._config = initial_config
        self._lock = threading.RLock()
    
    def __getitem__(self, key):
        with self._lock:
            return self._config[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._config[key] = value
            
    def get(self, key, default=None):
        with self._lock:
            return self._config.get(key, default)
            
    def update(self, updates):
        with self._lock:
            self._config.update(updates)

class ThreadSafeQueue:
    """Thread-safe queue for GUI updates"""
    def __init__(self, maxsize=10):
        self._queue = queue.Queue(maxsize=maxsize)
        
    def put(self, item, block=False):
        try:
            self._queue.put(item, block=block)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put(item, block=False)
            except queue.Empty:
                pass
                
    def get(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

class MotionModel(ABC):
    """Abstract base class for motion models"""
    @abstractmethod
    def predict(self, state, dt):
        pass
    
    @abstractmethod
    def get_process_noise(self, state, dt):
        pass

class LinearMotionModel(MotionModel):
    """Simple constant velocity model"""
    def predict(self, state, dt):
        # State: [x, y, vx, vy, w, h]
        return np.array([
            state[0] + state[2] * dt,
            state[1] + state[3] * dt,
            state[2],
            state[3],
            state[4],
            state[5]
        ])
    
    def get_process_noise(self, state, dt):
        return np.eye(6) * 0.03

class AircraftMotionModel(MotionModel):
    """Motion model for aircraft with turn constraints"""
    def __init__(self, max_turn_rate=0.1):  # rad/s
        self.max_turn_rate = max_turn_rate
        
    def predict(self, state, dt):
        # Include turn rate constraints
        speed = np.sqrt(state[2]**2 + state[3]**2)
        if speed > 0:
            # Limit turn rate
            direction = np.arctan2(state[3], state[2])
            # Simple prediction for now
        return LinearMotionModel().predict(state, dt)
    
    def get_process_noise(self, state, dt):
        speed = np.sqrt(state[2]**2 + state[3]**2)
        # Higher noise for higher speeds
        noise_scale = 0.03 + (speed / 1000.0) * 0.02
        return np.eye(6) * noise_scale

class BallisticMotionModel(MotionModel):
    """Motion model for ballistic objects with gravity"""
    def __init__(self, gravity=9.81):
        self.gravity = gravity
        
    def predict(self, state, dt):
        # Add gravity effect (assuming y is vertical)
        new_state = LinearMotionModel().predict(state, dt)
        new_state[3] += self.gravity * dt  # Add gravity to vy
        return new_state
    
    def get_process_noise(self, state, dt):
        return np.eye(6) * 0.05  # Higher noise for ballistic objects

class KalmanTracker:
    """Enhanced Kalman Filter with adaptive noise and motion models"""
    def __init__(self, fps=30.0, frame_skip=1, motion_model=None):
        # State vector: [x, y, vx, vy, w, h]
        self.kalman = cv2.KalmanFilter(6, 4)
        self.dt = frame_skip / fps
        self.motion_model = motion_model or LinearMotionModel()
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Update transition matrix
        self.update_dt(self.dt)
        
        # Initial noise values
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        
        self.initialized = False
        self.innovation_history = deque(maxlen=10)
        
    def update_dt(self, new_dt):
        """Update time step for variable frame rates"""
        self.dt = new_dt
        self.kalman.transitionMatrix = np.array([
            [1, 0, self.dt, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
    def initialize(self, bbox):
        """Initialize the filter with the first measurement"""
        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2
        self.kalman.statePost = np.array([center_x, center_y, 0, 0, w, h], np.float32)
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        self.initialized = False  # Reset to force re-initialization
        self.innovation_history.clear()
        self.initialized = True
        
    def predict(self):
        """Predict the next state"""
        if not self.initialized:
            return None
        return self.kalman.predict()
        
    def update(self, bbox):
        """Update with measurement and adapt noise"""
        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2
        measurement = np.array([center_x, center_y, w, h], np.float32)
        
        # Calculate innovation (measurement residual)
        if self.initialized:
            predicted_measurement = self.kalman.measurementMatrix @ self.kalman.statePre
            innovation = measurement - predicted_measurement[:4]
            self.innovation_history.append(np.linalg.norm(innovation))
            
            # Adapt measurement noise based on innovation
            self.adapt_measurement_noise()
            
        self.kalman.correct(measurement)
        
        # Adapt process noise based on motion
        self.adapt_process_noise()
        
    def adapt_measurement_noise(self):
        """Adapt measurement noise based on innovation history"""
        if len(self.innovation_history) >= 5:
            avg_innovation = np.mean(self.innovation_history)
            if avg_innovation > 10:  # High innovation = high noise
                scale = min(2.0, avg_innovation / 10.0)
                self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * (0.5 * scale)
            else:
                self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
                
    def adapt_process_noise(self):
        """Adapt process noise based on velocity and acceleration"""
        state = self.kalman.statePost
        velocity = np.sqrt(state[2]**2 + state[3]**2)
        
        # Get noise from motion model
        process_noise = self.motion_model.get_process_noise(state, self.dt)
        self.kalman.processNoiseCov = process_noise.astype(np.float32)
        
    def get_estimated_bbox(self):
        """Return the smoothed bounding box"""
        state = self.kalman.statePost
        x = state[0] - state[4]/2
        y = state[1] - state[5]/2
        return (int(x), int(y), int(state[4]), int(state[5]))
        
    def get_velocity(self):
        """Return velocity in pixels/frame"""
        return (self.kalman.statePost[2], self.kalman.statePost[3])
        
    def get_uncertainty(self):
        """Get position uncertainty from covariance"""
        return self.kalman.errorCovPost[:2, :2]
        
    def get_confidence(self):
        """Calculate tracking confidence based on innovation"""
        if not self.innovation_history:
            return 1.0
        recent_innovation = np.mean(list(self.innovation_history)[-5:])
        # Convert to confidence score (0-1)
        confidence = np.exp(-recent_innovation / 50.0)
        return np.clip(confidence, 0.0, 1.0)

class MultiHypothesisTracker:
    """Multiple Hypothesis Tracking for uncertain detections"""
    def __init__(self, n_hypotheses=3, fps=30.0, frame_skip=1):
        self.n_hypotheses = n_hypotheses
        self.trackers = [KalmanTracker(fps, frame_skip) for _ in range(n_hypotheses)]
        self.weights = np.ones(n_hypotheses) / n_hypotheses
        self.active_hypothesis = 0
        
    def initialize(self, bbox, detections=None):
        """Initialize with multiple hypotheses"""
        if detections and len(detections) > 1:
            # Initialize each hypothesis with different detection
            for i, (tracker, det) in enumerate(zip(self.trackers, detections[:self.n_hypotheses])):
                tracker.initialize(det)
        else:
            # Initialize all with same bbox but add noise
            for i, tracker in enumerate(self.trackers):
                noisy_bbox = self._add_noise_to_bbox(bbox, i * 5)
                tracker.initialize(noisy_bbox)
                
    def _add_noise_to_bbox(self, bbox, noise_level):
        """Add noise to bbox for hypothesis diversity"""
        x, y, w, h = bbox
        noise = np.random.randn(4) * noise_level
        return (x + noise[0], y + noise[1], max(10, w + noise[2]), max(10, h + noise[3]))
        
    def predict(self):
        """Predict all hypotheses"""
        predictions = []
        for tracker in self.trackers:
            pred = tracker.predict()
            if pred is not None:
                predictions.append(pred)
        return predictions
        
    def update(self, detections):
        """Update with multiple detections using probabilistic data association"""
        if not detections:
            return
            
        # Calculate association probabilities
        association_probs = np.zeros((self.n_hypotheses, len(detections)))
        
        for i, tracker in enumerate(self.trackers):
            if not tracker.initialized:
                continue
                
            pred_state = tracker.kalman.statePre
            pred_center = pred_state[:2]
            
            for j, det in enumerate(detections):
                det_center = np.array([det[0] + det[2]/2, det[1] + det[3]/2])
                distance = np.linalg.norm(pred_center - det_center)
                association_probs[i, j] = np.exp(-distance / 50.0)  # Gaussian-like
                
        # Normalize
        association_probs /= (association_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        # Update each hypothesis
        for i, tracker in enumerate(self.trackers):
            if not tracker.initialized:
                continue
                
            # Weighted update with all detections
            if len(detections) > 0:
                best_det_idx = np.argmax(association_probs[i])
                tracker.update(detections[best_det_idx])
                
                # Update weight based on innovation
                innovation = tracker.innovation_history[-1] if tracker.innovation_history else 100
                self.weights[i] *= np.exp(-innovation / 100.0)
                
        # Normalize weights
        self.weights /= self.weights.sum()
        
        # Select best hypothesis
        self.active_hypothesis = np.argmax(self.weights)
        
    def get_best_estimate(self):
        """Get estimate from best hypothesis"""
        return self.trackers[self.active_hypothesis].get_estimated_bbox()
        
    def get_confidence(self):
        """Get confidence from best hypothesis"""
        return self.trackers[self.active_hypothesis].get_confidence()

class GPUVideoProcessor:
    """GPU-accelerated video processing using CUDA or OpenCL"""
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.cuda_available = CUDA_AVAILABLE
        
        if self.use_gpu:
            if self.cuda_available:
                logger.info("Using CUDA GPU acceleration")
                self.stream1 = cv2.cuda_Stream()
                self.stream2 = cv2.cuda_Stream()
            else:
                logger.info("Using OpenCL GPU acceleration")
                
    def upload_frame(self, frame):
        """Upload frame to GPU memory"""
        if self.use_gpu and self.cuda_available:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            return gpu_frame
        elif self.use_gpu:
            return cv2.UMat(frame)
        return frame
        
    def download_frame(self, gpu_frame):
        """Download frame from GPU memory"""
        if self.use_gpu and self.cuda_available:
            return gpu_frame.download()
        elif self.use_gpu and isinstance(gpu_frame, cv2.UMat):
            return gpu_frame.get()
        return gpu_frame

class SpeedEstimationEngine:
    """Core processing engine with Kalman filtering and perspective correction"""
    def __init__(self, app_config, gpu_processor=None, app_instance=None):
        self.config = ThreadSafeConfig(app_config)
        self.gpu_processor = gpu_processor or GPUVideoProcessor()
        self.app = app_instance  # Reference to main app for accessing methods</        
        # Initialize trackers
        self.single_tracker = None
        self.multi_tracker = None
        self.use_mht = False
        
        # Perspective transform
        self.homography = None
        self.reference_points = None
        
        # Distance tracking
        self.distance_history = deque(maxlen=100)
        self.initial_distance = None
        self.initial_bbox_size = None
        
        # Trajectory estimation
        self.trajectory_points = deque(maxlen=1000)
        self.distance_model = None
        self.speed_history_3d = deque(maxlen=20)
        
        # Initialize motion model based on object type
        self._update_motion_model()
        
    def _update_motion_model(self):
        """Update motion model based on selected object"""
        obj_data = self.config['object_database'].get(self.config['selected_object'], {})
        category = obj_data.get('category', 'Unknown')
        
        if category in ['Aircraft', 'Military']:
            motion_model = AircraftMotionModel()
        elif category == 'Missile':
            motion_model = BallisticMotionModel()
        else:
            motion_model = LinearMotionModel()
            
        # Update tracker with new motion model
        fps = self.config['fps']
        frame_skip = self.config.get('frame_skip', 1)
        
        self.single_tracker = KalmanTracker(fps, frame_skip, motion_model)
        
        if self.use_mht:
            self.multi_tracker = MultiHypothesisTracker(n_hypotheses=3, fps=fps, frame_skip=frame_skip)
            
    def initialize_distance_tracking(self, initial_bbox, initial_distance=None):
        """Initialize distance tracking with first measurement"""
        self.initial_bbox_size = initial_bbox[2] if self.config['use_wingspan'] else initial_bbox[3]
        
        if initial_distance:
            self.initial_distance = initial_distance
        else:
            self.initial_distance = self.calculate_initial_distance(initial_bbox)
            
        self.distance_history.clear()
        self.distance_history.append({
            'frame': 0,
            'distance': self.initial_distance,
            'bbox_size': self.initial_bbox_size
        })
        
    def calculate_initial_distance(self, bbox):
        """Calculate initial distance from bbox size"""
        pixel_size = bbox[2] if self.config['use_wingspan'] else bbox[3]
        
        # Get object real size
        if self.app and hasattr(self.app, 'get_object_real_size'):
            real_size_m = self.app.get_object_real_size()
        else:
            # Fallback to database values
            obj_data = self.config['object_database'].get(self.config['selected_object'], {})
            if self.config['use_wingspan']:
                real_size_m = obj_data.get('wingspan', obj_data.get('width', obj_data.get('rotor_diameter', 10.0)))
            else:
                real_size_m = obj_data.get('length', obj_data.get('height', obj_data.get('diameter', 10.0)))
            
        # Calculate distance
        image_width = self.config.get('frame_width', 1920)
        focal_length_px = (self.config['focal_length_mm'] * image_width) / self.config['sensor_width_mm']
        estimated_distance = (real_size_m * focal_length_px) / pixel_size
        
        return estimated_distance
        
    def calculate_dynamic_distance(self, current_bbox, frame_num):
        """Calculate distance based on change in apparent size"""
        current_size = current_bbox[2] if self.config['use_wingspan'] else current_bbox[3]
        
        if self.initial_bbox_size and self.initial_distance:
            # Distance is inversely proportional to apparent size
            # d2/d1 = s1/s2 (where d=distance, s=size)
            size_ratio = self.initial_bbox_size / current_size
            current_distance = self.initial_distance * size_ratio
            
            # Smooth the distance estimate using running average
            if len(self.distance_history) > 5:
                recent_distances = [d['distance'] for d in list(self.distance_history)[-5:]]
                smoothed_distance = 0.7 * current_distance + 0.3 * np.mean(recent_distances)
            else:
                smoothed_distance = current_distance
                
            # Store in history
            self.distance_history.append({
                'frame': frame_num,
                'distance': smoothed_distance,
                'bbox_size': current_size
            })
            
            return smoothed_distance
        else:
            return self.config.get('estimated_distance', 1000)
            
    def estimate_3d_trajectory(self):
        """Estimate 3D trajectory from 2D tracks and distance changes"""
        if len(self.trajectory_points) < 10:
            return None
            
        points = list(self.trajectory_points)
        
        # Extract data
        times = np.array([p['time'] for p in points])
        x_pixels = np.array([p['x'] for p in points])
        y_pixels = np.array([p['y'] for p in points])
        distances = np.array([p['distance'] for p in points])
        
        # Convert pixel coordinates to angles
        focal_px = (self.config['focal_length_mm'] * self.config['frame_width']) / self.config['sensor_width_mm']
        
        # Angular position relative to camera center
        theta_x = np.arctan((x_pixels - self.config['frame_width']/2) / focal_px)
        theta_y = np.arctan((y_pixels - self.config['frame_height']/2) / focal_px)
        
        # Convert to 3D coordinates
        x_meters = distances * np.tan(theta_x)
        y_meters = distances * np.tan(theta_y)
        z_meters = distances
        
        return {
            'times': times,
            'x': x_meters,
            'y': y_meters,
            'z': z_meters,
            'distances': distances
        }
        
    def calculate_3d_speed(self):
        """Calculate speed from 3D trajectory"""
        trajectory = self.estimate_3d_trajectory()
        if trajectory is None:
            return None
            
        # Use recent points for speed calculation
        n_points = min(10, len(trajectory['times']))
        
        if n_points < 2:
            return None
            
        # Get recent positions
        recent_times = trajectory['times'][-n_points:]
        recent_x = trajectory['x'][-n_points:]
        recent_y = trajectory['y'][-n_points:]
        recent_z = trajectory['z'][-n_points:]
        
        # Calculate velocities using finite differences
        dt = recent_times[-1] - recent_times[-2]
        if dt <= 0:
            return None
            
        vx = (recent_x[-1] - recent_x[-2]) / dt
        vy = (recent_y[-1] - recent_y[-2]) / dt
        vz = (recent_z[-1] - recent_z[-2]) / dt
        
        speed_3d = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Smooth with previous estimates
        self.speed_history_3d.append(speed_3d)
        if len(self.speed_history_3d) > 5:
            return np.median(list(self.speed_history_3d)[-5:])
            
        return speed_3d
        
    def apply_high_speed_corrections(self, velocity_mps, distance):
        """Apply corrections for high-speed objects"""
        obj_data = self.config['object_database'].get(self.config['selected_object'], {})
        category = obj_data.get('category', 'Unknown')
        
        # Distance-based corrections
        if distance > 10000:  # Beyond 10km
            # Account for atmospheric distortion
            velocity_mps *= 1.1
            
        if distance > 20000:  # Beyond 20km
            # Account for extreme distance effects
            velocity_mps *= 1.15
            
        # Category-specific corrections
        if category == 'Missile':
            # Missiles often tracked at extreme distances
            if distance > 5000:
                velocity_mps *= 1.3
            # Boost phase detection
            if len(self.distance_history) > 10:
                recent_distances = [d['distance'] for d in list(self.distance_history)[-10:]]
                if recent_distances[-1] > recent_distances[0] * 1.5:  # Rapidly increasing distance
                    velocity_mps *= 1.5  # Likely in boost phase
                    
        elif category == 'Military':
            # Fighter jets
            if distance > 5000:
                velocity_mps *= 1.2
                
        return velocity_mps
        
    def enhance_frame_for_tracking(self, frame):
        """Enhance frame for better tracking of fast objects"""
        if self.config.get('high_speed_mode', False):
            # Apply motion deblur if available
            enhanced = frame.copy()
            
            # Increase contrast
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        return frame
            
    def process_frame(self, frame, original_frame, frame_count, frame_num):
        """Main processing function for a single frame"""
        # Apply enhancement for high-speed mode
        if self.config.get('high_speed_mode', False):
            frame = self.enhance_frame_for_tracking(frame)
            
        # Apply stabilization if enabled
        if self.config.get('stabilization_enabled', True) and frame_count % self.config.get('stabilization_interval', 3) == 0:
            processing_frame = self.stabilize_frame(frame.copy(), frame_count)
        else:
            processing_frame = frame.copy()
            
        # Track object if tracking is active
        if self.config['tracking'] and self.config['tracker']:
            if self.use_mht:
                success, results = self._track_with_mht(processing_frame)
            else:
                success, results = self._track_with_kalman(processing_frame)
                
            if success:
                bbox, speed_mps, uncertainty, confidence, distance = results
                self.config['bbox'] = bbox
                
                # Draw tracking information
                self.draw_tracking_info(original_frame, bbox, speed_mps, uncertainty, confidence, distance, is_original=True)
                self.draw_tracking_info(processing_frame, bbox, speed_mps, uncertainty, confidence, distance, is_original=False)
                
                # Update GUI with enhanced telemetry
                self._update_gui_info(speed_mps, bbox, frame_num, confidence, distance)
            else:
                self._handle_tracking_failure(processing_frame)
                
        return original_frame, processing_frame
        
    def _track_with_kalman(self, frame):
        """Track using single Kalman filter"""
        # Update tracker time step if needed
        if hasattr(self, 'single_tracker'):
            new_dt = self.config.get('frame_skip', 1) / self.config['fps']
            if abs(self.single_tracker.dt - new_dt) > 1e-6:
                self.single_tracker.update_dt(new_dt)
                
        # Get detection from OpenCV tracker
        success, bbox = self.config['tracker'].update(frame)
        
        if success:
            if not self.single_tracker.initialized:
                self.single_tracker.initialize(bbox)
            else:
                # Predict first
                self.single_tracker.predict()
                # Then update with measurement
                self.single_tracker.update(bbox)
                
            # Get smoothed estimates
            estimated_bbox = self.single_tracker.get_estimated_bbox()
            
            # Calculate speed with dynamic distance
            speed_mps, distance = self.calculate_speed_with_dynamic_distance()
            
            uncertainty = self.single_tracker.get_uncertainty()
            confidence = self.single_tracker.get_confidence()
            
            return True, (estimated_bbox, speed_mps, uncertainty, confidence, distance)
        else:
            return False, None
            
    def _track_with_mht(self, frame):
        """Track using Multiple Hypothesis Tracking"""
        # Get multiple detections (could be from different detection methods)
        detections = self._get_multiple_detections(frame)
        
        if not self.multi_tracker.trackers[0].initialized:
            if detections:
                self.multi_tracker.initialize(detections[0], detections)
                return True, (detections[0], 0.0, np.eye(2), 1.0, self.config.get('estimated_distance', 1000))
            return False, None
            
        # Predict all hypotheses
        self.multi_tracker.predict()
        
        # Update with detections
        self.multi_tracker.update(detections)
        
        # Get best estimate
        estimated_bbox = self.multi_tracker.get_best_estimate()
        
        # Calculate speed with dynamic distance
        speed_mps, distance = self.calculate_speed_with_dynamic_distance(use_mht=True)
        
        # Get uncertainty from best hypothesis
        best_tracker = self.multi_tracker.trackers[self.multi_tracker.active_hypothesis]
        uncertainty = best_tracker.get_uncertainty()
        confidence = self.multi_tracker.get_confidence()
        
        return True, (estimated_bbox, speed_mps, uncertainty, confidence, distance)
        
    def _get_multiple_detections(self, frame):
        """Get multiple possible detections for MHT"""
        detections = []
        
        # Primary detection from OpenCV tracker
        success, bbox = self.config['tracker'].update(frame)
        if success:
            detections.append(bbox)
            
        # Could add more detection methods here
        # e.g., template matching, deep learning detector, etc.
        
        return detections
        
    def calculate_speed_with_dynamic_distance(self, use_mht=False):
        """Enhanced speed calculation using dynamic distance"""
        # Get the active tracker
        if use_mht:
            tracker = self.multi_tracker.trackers[self.multi_tracker.active_hypothesis]
        else:
            tracker = self.single_tracker
            
        if not tracker.initialized:
            return 0.0, 0.0
            
        # Get current state
        bbox = tracker.get_estimated_bbox()
        frame_num = self.config.get('current_frame_num', 0)
        
        # Calculate current distance
        if self.config.get('manual_distance', False):
            try:
                current_distance = float(self.config.get('manual_distance_value', '1000'))
            except:
                current_distance = 1000
        else:
            current_distance = self.calculate_dynamic_distance(bbox, frame_num)
        
        # Get velocity from Kalman filter
        vx_px, vy_px = tracker.get_velocity()
        velocity_pixels_per_frame = np.sqrt(vx_px**2 + vy_px**2)
        
        # Store trajectory point
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        self.trajectory_points.append({
            'frame': frame_num,
            'time': frame_num / self.config['fps'],
            'x': center_x,
            'y': center_y,
            'distance': current_distance,
            'vx_px': vx_px,
            'vy_px': vy_px
        })
        
        # Calculate 3D speed if we have enough trajectory points
        if len(self.trajectory_points) > 10:
            speed_3d = self.calculate_3d_speed()
            if speed_3d is not None:
                # Apply corrections for high-speed mode
                if self.config.get('high_speed_mode', False):
                    speed_3d = self.apply_high_speed_corrections(speed_3d, current_distance)
                    
                # Update speed history
                self.config['speed_history'].append(speed_3d)
                
                return speed_3d, current_distance
                
        # Fallback to 2D calculation with current distance
        # Angular velocity method
        focal_px = (self.config['focal_length_mm'] * self.config['frame_width']) / self.config['sensor_width_mm']
        
        # Angular velocity in radians per frame
        angular_velocity_x = vx_px / focal_px
        angular_velocity_y = vy_px / focal_px
        angular_velocity = np.sqrt(angular_velocity_x**2 + angular_velocity_y**2)
        
        # Linear velocity = distance × angular velocity
        velocity_mps = current_distance * angular_velocity * self.config['fps']
        
        # Apply corrections for high-speed mode
        if self.config.get('high_speed_mode', False):
            velocity_mps = self.apply_high_speed_corrections(velocity_mps, current_distance)
            
        # Update speed history
        self.config['speed_history'].append(velocity_mps)
        
        # Return filtered speed
        if len(self.config['speed_history']) > 3:
            filtered_speed = self._robust_speed_estimate(list(self.config['speed_history']))
            return filtered_speed, current_distance
            
        return velocity_mps, current_distance
        
    def calculate_speed_from_kalman(self, use_mht=False):
        """Calculate speed using Kalman-filtered velocity (legacy method)"""
        speed_mps, distance = self.calculate_speed_with_dynamic_distance(use_mht)
        return speed_mps
        
    def _robust_speed_estimate(self, speeds):
        """Get robust speed estimate from history"""
        if self.config.get('high_speed_mode', False) and np.mean(speeds[-10:]) > 340:
            # For supersonic, use median
            return np.median(speeds[-15:])
        else:
            # For subsonic, remove outliers then average
            q1, q3 = np.percentile(speeds[-10:], [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [s for s in speeds[-10:] if lower <= s <= upper]
            return np.mean(filtered) if filtered else np.median(speeds[-10:])
            
    def stabilize_frame(self, frame, frame_count):
        """Stabilize frame using feature matching"""
        # Simple implementation - expand as needed
        return frame
        
    def draw_tracking_info(self, frame, bbox, speed_mps, uncertainty, confidence, distance, is_original):
        """Draw tracking information with uncertainty visualization"""
        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw center point
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Draw uncertainty ellipse
        if uncertainty is not None and not is_original:
            self.draw_uncertainty_ellipse(frame, (center_x, center_y), uncertainty)
            
        # Draw speed information
        if not is_original:
            # Speed info
            y_offset = 30
            speed_text = f"Speed: {speed_mps:.1f} m/s"
            cv2.putText(frame, speed_text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Additional info
            y_offset += 30
            speed_measurement = SpeedMeasurement.from_mps(speed_mps)
            extra_text = f"{speed_measurement.kph:.1f} km/h | {speed_measurement.mph:.1f} mph"
            cv2.putText(frame, extra_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mach number for high speeds
            if speed_mps > 340:
                y_offset += 30
                mach_text = f"Mach {speed_measurement.mach:.2f}"
                color = (0, 0, 255) if speed_measurement.mach > 5 else (0, 165, 255)
                cv2.putText(frame, mach_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                           
            # Distance and confidence info
            y_offset += 30
            cv2.putText(frame, f"Distance: {distance:.1f} m", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset += 30
            conf_text = f"Confidence: {confidence:.2%}"
            conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, conf_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
                       
    def draw_uncertainty_ellipse(self, frame, center, covariance):
        """Draw uncertainty ellipse around tracked object"""
        if covariance.shape != (2, 2):
            return
            
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        
        # Sort by eigenvalue
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Calculate ellipse parameters (95% confidence)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = int(2 * np.sqrt(5.991 * eigenvalues[0]))
        height = int(2 * np.sqrt(5.991 * eigenvalues[1]))
        
        # Limit size for display
        width = min(width, 200)
        height = min(height, 200)
        
        # Draw ellipse
        cv2.ellipse(frame, center, (width, height), angle, 0, 360, (255, 255, 0), 2)
        
    def _update_gui_info(self, speed_mps, bbox, frame_num, confidence, distance):
        """Update GUI with tracking information"""
        speed_measurement = SpeedMeasurement.from_mps(speed_mps)
        
        # Get center position for trajectory
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        # Create telemetry point with all data
        telemetry_point = {
            'frame': frame_num,
            'time': frame_num / self.config['fps'],
            'speed': speed_mps,
            'speed_kph': speed_measurement.kph,
            'speed_mph': speed_measurement.mph,
            'speed_mach': speed_measurement.mach,
            'distance': distance,
            'distance_change': 0,  # Will be calculated
            'confidence': confidence,
            'bbox_center_x': center_x,
            'bbox_center_y': center_y,
            'bbox_width': bbox[2],
            'bbox_height': bbox[3],
            'apparent_size': bbox[2] if self.config['use_wingspan'] else bbox[3]
        }
        
        # Send updates to GUI
        self.config['gui_update_queue'].put({
            'type': 'speed',
            'mps': speed_measurement.mps,
            'kph': speed_measurement.kph,
            'mph': speed_measurement.mph,
            'knots': speed_measurement.knots,
            'mach': speed_measurement.mach if self.config.get('high_speed_mode') else None
        })
        
        self.config['gui_update_queue'].put({
            'type': 'distance',
            'distance': distance
        })
        
        self.config['gui_update_queue'].put({
            'type': 'telemetry',
            'data': telemetry_point
        })
        
        self.config['gui_update_queue'].put({
            'type': 'confidence',
            'confidence': f"{confidence:.0%}"
        })
        
    def _handle_tracking_failure(self, frame):
        """Handle tracking failure"""
        cv2.putText(frame, "TRACKING LOST", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        self.config['gui_update_queue'].put({
            'type': 'confidence',
            'confidence': 'Lost'
        })

class AdvancedSpeedEstimator:
    """Main application class with GUI"""
    def __init__(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "PIL (Pillow) is required. Please install it with: pip install pillow")
            sys.exit(1)
            
        self.root = tk.Tk()
        self.root.title("Advanced Object Speed Estimator (Hypersonic Edition)")
        self.root.geometry("1600x900")
        
        # Telemetry data with size limit to prevent unbounded growth
        self.telemetry_data = deque(maxlen=10000)
        
        # Store initial bounding box for replay
        self.initial_bbox = None
        
        # Initialize configuration
        self.config = {
            'tracking': False,
            'tracker': None,
            'bbox': None,
            'use_wingspan': True,
            'selected_object': "Boeing 737-800",
            'object_database': self._initialize_object_database(),
            'sensor_database': self._initialize_sensor_database(),
            'focal_length_mm': 50.0,
            'sensor_width_mm': 36.0,
            'sensor_height_mm': 24.0,
            'estimated_distance': 0.0,
            'manual_distance': False,
            'use_manual_size': False,
            'manual_width_value': '10.0',
            'manual_length_value': '10.0',
            'speed_history': deque(maxlen=60),
            'fps': 30.0,
            'frame_width': 1920,
            'frame_height': 1080,
            'current_frame_num': 0,
            'frame_skip': 1,
            'use_gpu': GPU_AVAILABLE,
            'use_multicore': True,
            'high_speed_mode': False,
            'stabilization_enabled': True,
            'stabilization_interval': 3,
            'max_expected_speed': 5000.0,
            'gui_update_queue': ThreadSafeQueue(maxsize=20),
            'video_loaded': False,
            'paused': True,
            'video_finished': False
        }
        
        # Initialize components
        self.gpu_processor = GPUVideoProcessor(use_gpu=self.config['use_gpu'])
        self.engine = SpeedEstimationEngine(self.config, self.gpu_processor, self)
        
        # Video properties
        self.cap = None
        self.current_frame = None
        self.total_frames = 0
        
        # Threading
        self.processing_thread = None
        self.stop_thread = threading.Event()
        self.frame_queue = queue.Queue(maxsize=5)
        self.preprocessor_pool = ProcessPoolExecutor(max_workers=min(CPU_COUNT, 4))
        
        # Create GUI
        self.create_gui()
        
        # Show system info in title
        gpu_status = "GPU: CUDA" if CUDA_AVAILABLE else ("GPU: OpenCL" if GPU_AVAILABLE else "CPU Only")
        self.root.title(f"Advanced Object Speed Estimator ({gpu_status}, {CPU_COUNT} cores)")
        
        # Start GUI update timer
        self.update_gui_timer()
        
        # Bind window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def _initialize_object_database(self) -> Dict:
        """Initialize comprehensive object database"""
        return {
            # Aircraft
            "Cessna 172": {"wingspan": 11.0, "length": 8.28, "category": "Aircraft"},
            "Boeing 737-800": {"wingspan": 35.8, "length": 39.5, "category": "Aircraft"},
            "Boeing 747-400": {"wingspan": 68.4, "length": 76.3, "category": "Aircraft"},
            "Airbus A320": {"wingspan": 35.8, "length": 37.6, "category": "Aircraft"},
            "F-16 Fighter": {"wingspan": 10.0, "length": 15.1, "category": "Military"},
            "F-22 Raptor": {"wingspan": 13.6, "length": 18.9, "category": "Military"},
            "SR-71 Blackbird": {"wingspan": 16.9, "length": 32.7, "category": "Military"},
            
            # Missiles/Projectiles
            "Cruise Missile": {"length": 6.25, "wingspan": 2.67, "category": "Missile"},
            "Ballistic Missile": {"length": 12.0, "diameter": 1.5, "category": "Missile"},
            "Hypersonic Missile": {"length": 8.0, "diameter": 0.8, "category": "Missile"},
            
            # Drones
            "DJI Mavic 3": {"wingspan": 0.347, "length": 0.283, "category": "Drone"},
            "DJI Phantom 4": {"wingspan": 0.350, "length": 0.350, "category": "Drone"},
            "Military Drone (Large)": {"wingspan": 20.0, "length": 11.0, "category": "Drone"},
            
            # Vehicles
            "Car (Sedan)": {"length": 4.5, "width": 1.8, "category": "Vehicle"},
            "SUV": {"length": 4.8, "width": 1.9, "category": "Vehicle"},
            "Truck": {"length": 6.5, "width": 2.5, "category": "Vehicle"},
            
            # People
            "Walking Adult": {"height": 1.7, "width": 0.45, "category": "Person"},
            "Running Adult": {"height": 1.7, "width": 0.45, "category": "Person"},
            
            # Custom
            "Custom Object": {"wingspan": 10.0, "length": 10.0, "category": "Custom"}
        }
        
    def _initialize_sensor_database(self) -> Dict:
        """Initialize camera sensor database with accurate specifications"""
        return {
            "DSLR/Mirrorless": {
                "Full Frame (36x24mm)": {"width": 36.0, "height": 24.0, "focal_length": None},
                "APS-C Canon (22.3x14.9mm)": {"width": 22.3, "height": 14.9, "focal_length": None},
                "APS-C Nikon/Sony (23.5x15.6mm)": {"width": 23.5, "height": 15.6, "focal_length": None},
                "Micro 4/3 (17.3x13mm)": {"width": 17.3, "height": 13.0, "focal_length": None},
                "1-inch (13.2x8.8mm)": {"width": 13.2, "height": 8.8, "focal_length": None}
            },
            "Smartphone": {
                "iPhone 15 Pro Max": {
                    "width": 9.8, "height": 7.3, 
                    "focal_length": 6.86,  # Main camera 24mm equivalent
                    "type": "1/1.28-inch"
                },
                "iPhone 14 Pro": {
                    "width": 9.8, "height": 7.3, 
                    "focal_length": 6.86,  # Main camera 24mm equivalent
                    "type": "1/1.28-inch"
                },
                "iPhone 13 Pro": {
                    "width": 7.6, "height": 5.7,
                    "focal_length": 5.7,   # Main camera 26mm equivalent
                    "type": "1/1.65-inch"
                },
                "Samsung S24 Ultra": {
                    "width": 9.8, "height": 7.3,
                    "focal_length": 6.3,   # Main camera 23mm equivalent
                    "type": "1/1.3-inch"
                },
                "Samsung S23 Ultra": {
                    "width": 9.0, "height": 6.75,
                    "focal_length": 6.3,   # Main camera 23mm equivalent
                    "type": "1/1.33-inch"
                },
                "Google Pixel 8 Pro": {
                    "width": 9.8, "height": 7.3,
                    "focal_length": 6.9,   # Main camera 25mm equivalent
                    "type": "1/1.31-inch"
                },
                "Google Pixel 7 Pro": {
                    "width": 7.9, "height": 5.9,
                    "focal_length": 6.81,  # Main camera 25mm equivalent
                    "type": "1/1.31-inch"
                },
                "OnePlus 11": {
                    "width": 8.8, "height": 6.6,
                    "focal_length": 6.06,  # Main camera 23mm equivalent
                    "type": "1/1.56-inch"
                },
                "Xiaomi 13 Pro": {
                    "width": 13.2, "height": 8.8,
                    "focal_length": 8.7,   # Main camera 23mm equivalent (1-inch sensor)
                    "type": "1-inch"
                },
                "Generic 1/2.3-inch": {"width": 6.17, "height": 4.56, "focal_length": 4.3},
                "Generic 1/2.55-inch": {"width": 5.6, "height": 4.2, "focal_length": 4.0}
            }
        }
        
    def create_gui(self):
        """Create the GUI interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display tab
        self.create_video_tab(notebook)
        
        # Settings tab
        self.create_settings_tab(notebook)
        
        # Data & Plots tab
        self.create_plot_tab(notebook)
        
        # Distance Analysis tab
        self.create_distance_plot_tab(notebook)
        
        # Analysis tab
        self.create_analysis_tab(notebook)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def create_control_panel(self, parent):
        """Create main control panel with tracker selection"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Basic controls
        ttk.Button(control_frame, text="Open Video", command=self.open_video).grid(row=0, column=0, padx=5)
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.grid(row=0, column=1, padx=5)
        self.select_button = ttk.Button(control_frame, text="Select Object", command=self.select_object, state=tk.DISABLED)
        self.select_button.grid(row=0, column=2, padx=5)
        
        # Tracker selection
        tracker_select_frame = ttk.LabelFrame(control_frame, text="Tracker", padding="5")
        tracker_select_frame.grid(row=0, column=3, padx=10)
        
        self.tracker_type_var = tk.StringVar(value="CSRT")
        tracker_types = ["CSRT", "KCF", "MOSSE", "MIL"]
        ttk.Label(tracker_select_frame, text="Type:").grid(row=0, column=0)
        self.tracker_combo = ttk.Combobox(tracker_select_frame, textvariable=self.tracker_type_var,
                                         values=tracker_types, width=10, state="readonly")
        self.tracker_combo.grid(row=0, column=1, padx=5)
        
        # Tracking options
        tracking_frame = ttk.LabelFrame(control_frame, text="Tracking", padding="5")
        tracking_frame.grid(row=0, column=4, padx=10)
        
        self.mht_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tracking_frame, text="Multi-Hypothesis", variable=self.mht_var,
                       command=self.toggle_mht).grid(row=0, column=0)
        
        self.stab_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracking_frame, text="Stabilization", variable=self.stab_var,
                       command=self.toggle_stabilization).grid(row=1, column=0)
        
        # Add manual distance override
        distance_frame = ttk.LabelFrame(control_frame, text="Distance", padding="5")
        distance_frame.grid(row=0, column=5, padx=10)
        
        self.manual_distance_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(distance_frame, text="Manual", variable=self.manual_distance_var,
                       command=self.toggle_manual_distance).grid(row=0, column=0)
        
        self.distance_var = tk.StringVar(value="1000")
        self.distance_entry = ttk.Entry(distance_frame, textvariable=self.distance_var, width=8, state=tk.DISABLED)
        self.distance_entry.grid(row=1, column=0)
        ttk.Label(distance_frame, text="meters").grid(row=1, column=1)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(control_frame, text="Performance", padding="5")
        perf_frame.grid(row=0, column=6, padx=10)
        
        self.gpu_var = tk.BooleanVar(value=self.config['use_gpu'])
        ttk.Checkbutton(perf_frame, text="GPU", variable=self.gpu_var,
                       command=self.toggle_gpu,
                       state=tk.NORMAL if GPU_AVAILABLE else tk.DISABLED).grid(row=0, column=0)
        
        self.multicore_var = tk.BooleanVar(value=self.config['use_multicore'])
        ttk.Checkbutton(perf_frame, text=f"Multicore ({CPU_COUNT})",
                       variable=self.multicore_var,
                       command=self.toggle_multicore).grid(row=1, column=0)
        
        self.highspeed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(perf_frame, text="High-Speed",
                       variable=self.highspeed_var,
                       command=self.toggle_highspeed).grid(row=2, column=0)
        
        # Speed display
        speed_frame = ttk.LabelFrame(control_frame, text="Speed", padding="10")
        speed_frame.grid(row=0, column=7, padx=20)
        
        self.speed_display_mps = ttk.Label(speed_frame, text="0.0 m/s", font=("Arial", 14, "bold"))
        self.speed_display_mps.grid(row=0, column=0, padx=5)
        
        self.speed_display_kph = ttk.Label(speed_frame, text="0.0 km/h", font=("Arial", 12))
        self.speed_display_kph.grid(row=0, column=1, padx=5)
        
        self.speed_display_mph = ttk.Label(speed_frame, text="0.0 mph", font=("Arial", 12))
        self.speed_display_mph.grid(row=0, column=2, padx=5)
        
        self.speed_display_mach = ttk.Label(speed_frame, text="", font=("Arial", 12, "bold"))
        self.speed_display_mach.grid(row=1, column=0, columnspan=3, pady=5)
        
    def create_video_tab(self, notebook):
        """Create video display tab"""
        video_tab = ttk.Frame(notebook)
        notebook.add(video_tab, text="Video")
        
        video_container = ttk.Frame(video_tab)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        # Original video
        orig_frame = ttk.LabelFrame(video_container, text="Original", padding="5")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_original = tk.Canvas(orig_frame, bg="black")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Processed video
        proc_frame = ttk.LabelFrame(video_container, text="Processed", padding="5")
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_processed = tk.Canvas(proc_frame, bg="black")
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)
        
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Settings")
        
        settings_container = ttk.Frame(settings_tab, padding="20")
        settings_container.pack(fill=tk.BOTH, expand=True)
        
        # Object selection
        self.create_object_selection(settings_container)
        
        # Camera settings
        self.create_camera_settings(settings_container)
        
        # Diagnostics section
        self.create_diagnostics_section(settings_container)
        
    def create_object_selection(self, parent):
        """Create object selection interface with manual size entry"""
        obj_frame = ttk.LabelFrame(parent, text="Object Specifications", padding="15")
        obj_frame.pack(fill=tk.X, pady=10)
        
        # Category filter
        ttk.Label(obj_frame, text="Category:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.category_var = tk.StringVar(value="All")
        categories = ["All"] + sorted(set(obj["category"] for obj in self.config['object_database'].values()))
        self.category_combo = ttk.Combobox(obj_frame, textvariable=self.category_var,
                                          values=categories, width=15, state="readonly")
        self.category_combo.grid(row=0, column=1, padx=5, pady=5)
        self.category_combo.bind("<<ComboboxSelected>>", self.on_category_changed)
        
        # Object selection
        ttk.Label(obj_frame, text="Object:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.object_var = tk.StringVar(value=self.config['selected_object'])
        self.object_combo = ttk.Combobox(obj_frame, textvariable=self.object_var,
                                        values=list(self.config['object_database'].keys()), width=25)
        self.object_combo.grid(row=1, column=1, padx=5, pady=5)
        self.object_combo.bind("<<ComboboxSelected>>", self.on_object_changed)
        
        # Measurement type
        ttk.Label(obj_frame, text="Measure:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        measure_frame = ttk.Frame(obj_frame)
        measure_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.measure_var = tk.StringVar(value="wingspan")
        ttk.Radiobutton(measure_frame, text="Width/Wingspan",
                       variable=self.measure_var, value="wingspan",
                       command=self.update_measurement_type).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(measure_frame, text="Length/Height",
                       variable=self.measure_var, value="length",
                       command=self.update_measurement_type).pack(side=tk.LEFT, padx=5)
        
        # Manual size entry
        size_frame = ttk.LabelFrame(obj_frame, text="Object Size", padding="10")
        size_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Size mode selection
        self.size_mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(size_frame, text="Use Database", 
                       variable=self.size_mode_var, value="auto",
                       command=self.toggle_size_mode).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(size_frame, text="Manual Entry", 
                       variable=self.size_mode_var, value="manual",
                       command=self.toggle_size_mode).grid(row=0, column=1, padx=5)
        
        # Manual entry fields
        ttk.Label(size_frame, text="Width/Wingspan (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.manual_width_var = tk.StringVar(value="10.0")
        self.manual_width_entry = ttk.Entry(size_frame, textvariable=self.manual_width_var, width=10, state=tk.DISABLED)
        self.manual_width_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(size_frame, text="Length/Height (m):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.manual_length_var = tk.StringVar(value="10.0")
        self.manual_length_entry = ttk.Entry(size_frame, textvariable=self.manual_length_var, width=10, state=tk.DISABLED)
        self.manual_length_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Common size presets
        preset_frame = ttk.LabelFrame(size_frame, text="Quick Presets", padding="5")
        preset_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        preset_buttons = [
            ("Human (1.7m)", 0.45, 1.7),
            ("Car (4.5m)", 1.8, 4.5),
            ("Bus (12m)", 2.5, 12.0),
            ("Fighter Jet (15m)", 10.0, 15.0),
            ("Airliner (40m)", 36.0, 40.0),
            ("Small Drone (0.35m)", 0.35, 0.35)
        ]
        
        for i, (name, width, length) in enumerate(preset_buttons):
            ttk.Button(preset_frame, text=name, 
                      command=lambda w=width, l=length: self.apply_size_preset(w, l),
                      width=15).grid(row=i//2, column=i%2, padx=2, pady=2)
        
        # Display current size
        self.size_display_label = ttk.Label(obj_frame, text="", font=("Arial", 10, "italic"))
        self.size_display_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Update initial display
        self.update_size_display()
        
    def create_camera_settings(self, parent):
        """Create camera settings interface"""
        cam_frame = ttk.LabelFrame(parent, text="Camera Parameters", padding="15")
        cam_frame.pack(fill=tk.X, pady=10)
        
        # Camera type
        ttk.Label(cam_frame, text="Camera Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_type_var = tk.StringVar(value="Custom")
        self.camera_type_combo = ttk.Combobox(cam_frame, textvariable=self.camera_type_var,
                                             values=["Custom", "DSLR/Mirrorless", "Smartphone"],
                                             width=20, state="readonly")
        self.camera_type_combo.grid(row=0, column=1, padx=5, pady=5)
        self.camera_type_combo.bind("<<ComboboxSelected>>", self.on_camera_type_changed)
        
        # Sensor model
        ttk.Label(cam_frame, text="Sensor:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sensor_model_var = tk.StringVar(value="Custom")
        self.sensor_model_combo = ttk.Combobox(cam_frame, textvariable=self.sensor_model_var,
                                              values=[], width=30, state="disabled")
        self.sensor_model_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        self.sensor_model_combo.bind("<<ComboboxSelected>>", self.on_sensor_model_changed)
        
        # Focal length
        ttk.Label(cam_frame, text="Focal Length (mm):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.focal_var = tk.StringVar(value="50.0")
        ttk.Entry(cam_frame, textvariable=self.focal_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Sensor width
        ttk.Label(cam_frame, text="Sensor Width (mm):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.sensor_var = tk.StringVar(value="36.0")
        self.sensor_entry = ttk.Entry(cam_frame, textvariable=self.sensor_var, width=10)
        self.sensor_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(cam_frame, text="Apply", command=self.apply_camera_settings).grid(row=4, column=0, columnspan=2, pady=10)
        
    def create_diagnostics_section(self, parent):
        """Create diagnostics section in settings"""
        diag_frame = ttk.LabelFrame(parent, text="Diagnostics", padding="15")
        diag_frame.pack(fill=tk.X, pady=10)
        
        # Display calculated values
        ttk.Label(diag_frame, text="Focal Length (pixels):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.focal_px_label = ttk.Label(diag_frame, text="N/A")
        self.focal_px_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(diag_frame, text="Pixel Angular Size (mrad):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.pixel_angle_label = ttk.Label(diag_frame, text="N/A")
        self.pixel_angle_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(diag_frame, text="Distance (m):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.distance_label = ttk.Label(diag_frame, text="N/A")
        self.distance_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(diag_frame, text="Pixel-to-Meter:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.pixel_meter_label = ttk.Label(diag_frame, text="N/A")
        self.pixel_meter_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(diag_frame, text="Update Diagnostics", command=self.update_diagnostics).grid(row=4, column=0, columnspan=2, pady=10)
        
    def create_plot_tab(self, notebook):
        """Creates the tab for displaying telemetry graphs"""
        if not MATPLOTLIB_AVAILABLE:
            return  # Don't create tab if matplotlib isn't available
            
        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="Data & Plots")
        
        # Create frame for plot controls
        plot_controls_frame = ttk.Frame(plot_tab)
        plot_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(plot_controls_frame, text="Update Plot", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls_frame, text="Clear Data", command=self.clear_plot_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls_frame, text="Export Data", command=self.export_telemetry).pack(side=tk.LEFT, padx=5)
        
        # Plot type selection
        ttk.Label(plot_controls_frame, text="Plot Type:").pack(side=tk.LEFT, padx=(20, 5))
        self.plot_type_var = tk.StringVar(value="speed_time")
        plot_types = ttk.Combobox(plot_controls_frame, textvariable=self.plot_type_var,
                                 values=["speed_time", "multi_plot", "trajectory", "confidence"],
                                 width=15, state="readonly")
        plot_types.pack(side=tk.LEFT, padx=5)
        plot_types.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        
        # Create matplotlib figure and canvas
        self.plot_figure = Figure(figsize=(12, 8), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=plot_tab)
        self.plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.plot_canvas.draw()
        
    def create_distance_plot_tab(self, notebook):
        """Create tab for distance visualization"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        distance_tab = ttk.Frame(notebook)
        notebook.add(distance_tab, text="Distance Analysis")
        
        # Controls
        controls_frame = ttk.Frame(distance_tab)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Update Distance Plot", 
                  command=self.update_distance_plot).pack(side=tk.LEFT, padx=5)
        
        self.distance_method_var = tk.StringVar(value="dynamic")
        ttk.Radiobutton(controls_frame, text="Dynamic Distance", 
                       variable=self.distance_method_var, value="dynamic").pack(side=tk.LEFT)
        ttk.Radiobutton(controls_frame, text="Fixed Distance", 
                       variable=self.distance_method_var, value="fixed").pack(side=tk.LEFT)
        
        # Create plot
        self.distance_figure = Figure(figsize=(12, 8), dpi=100)
        self.distance_canvas = FigureCanvasTkAgg(self.distance_figure, master=distance_tab)
        self.distance_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def create_analysis_tab(self, notebook):
        """Create analysis tab with detailed tracker information"""
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis")
        
        # Create scrollable frame
        canvas = tk.Canvas(analysis_tab)
        scrollbar = ttk.Scrollbar(analysis_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Tracker Information
        tracker_info = """
        TRACKER TYPES EXPLAINED:
        
        • CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability):
          - Best For: General purpose tracking with good accuracy
          - Pros: Handles scale changes well, robust to occlusions
          - Cons: Slower than other trackers (30-40 fps)
          - Use When: Tracking objects with moderate speed and size changes
        
        • KCF (Kernelized Correlation Filter):
          - Best For: Objects with consistent appearance
          - Pros: Fast performance (100+ fps), good for real-time
          - Cons: Doesn't handle scale changes well, can drift
          - Use When: Object maintains similar size, need fast processing
        
        • MOSSE (Minimum Output Sum of Squared Error):
          - Best For: Very fast tracking of high-speed objects
          - Pros: Extremely fast (450+ fps), good for rapid motion
          - Cons: Less accurate, prone to drift, basic features only
          - Use When: Tracking very fast objects where speed > accuracy
        
        • MIL (Multiple Instance Learning):
          - Best For: Objects with partial occlusions
          - Pros: Handles occlusions well, learns from multiple samples
          - Cons: Moderate speed, can accumulate errors over time
          - Use When: Object may be partially hidden during tracking
        
        TRACKING FEATURES:
        
        • Kalman Filter: Provides smoothed position and velocity estimates
        • Uncertainty Ellipse: Shows 95% confidence region for object position
        • Motion Models: Automatically selected based on object type
        • Multi-Hypothesis Tracking: Handles uncertain detections
        
        DYNAMIC DISTANCE TRACKING:
        • Tracks object distance based on apparent size changes
        • Provides 3D trajectory estimation
        • More accurate for objects that don't maintain constant distance
        
        HIGH-SPEED MODE OPTIMIZATIONS:
        • Adaptive process noise for rapid acceleration
        • Motion prediction for fast objects
        • Robust speed estimation using median filtering
        • Perspective correction for angled views
        • Frame enhancement for better tracking
        
        DATA ANALYSIS:
        • Real-time telemetry plotting
        • Distance tracking visualization
        • Export data to CSV for external analysis
        • Multiple plot types for comprehensive analysis
        • Confidence tracking over time
        
        TIPS FOR BEST RESULTS:
        • For Hypersonic Objects: Use MOSSE tracker + High-Speed Mode
        • For Aircraft: Use CSRT tracker for accuracy
        • For Small/Distant Objects: Enable frame enhancement
        • For Known Distances: Use manual distance entry
        • For Varying Distances: Enable dynamic distance tracking
        • Always verify camera parameters and object dimensions
        """
        
        ttk.Label(scrollable_frame, text=tracker_info, font=("Arial", 11),
                 justify=tk.LEFT).pack(pady=20, padx=20)
                 
    def clear_plot_data(self):
        """Clears the collected telemetry data and the plot"""
        self.telemetry_data.clear()
        if hasattr(self, 'plot_figure'):
            self.plot_figure.clear()
            ax = self.plot_figure.add_subplot(111)
            ax.grid(True)
            ax.set_title("Object Telemetry (Cleared)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (m/s)")
            self.plot_canvas.draw()
        self.status_bar.config(text="Telemetry data cleared.")
        
    def update_plot(self):
        """Updates the plot with current telemetry data"""
        if not MATPLOTLIB_AVAILABLE or not self.telemetry_data:
            self.status_bar.config(text="No telemetry data to plot.")
            return
            
        plot_type = self.plot_type_var.get()
        self.plot_figure.clear()
        
        # Sample data if too many points for performance
        data = list(self.telemetry_data)
        if len(data) > 1000:
            step = len(data) // 1000
            data = data[::step]
            
        if plot_type == "speed_time":
            self._plot_speed_time(data)
        elif plot_type == "multi_plot":
            self._plot_multi(data)
        elif plot_type == "trajectory":
            self._plot_trajectory(data)
        elif plot_type == "confidence":
            self._plot_confidence(data)
            
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
        self.status_bar.config(text=f"Plot updated with {len(data)} data points.")
        
    def _plot_speed_time(self, data):
        """Plot speed vs time"""
        ax = self.plot_figure.add_subplot(111)
        
        times = [d['time'] for d in data]
        speeds = [d['speed'] for d in data]
        
        ax.plot(times, speeds, 'b-', linewidth=2, label='Speed')
        
        # Add confidence bands if available
        if 'confidence' in data[0]:
            confidences = [d['confidence'] * 10 for d in data]  # Scale for visibility
            speeds_upper = [s + c for s, c in zip(speeds, confidences)]
            speeds_lower = [s - c for s, c in zip(speeds, confidences)]
            ax.fill_between(times, speeds_lower, speeds_upper, alpha=0.3, label='Confidence Band')
            
        ax.set_title("Object Speed Over Time", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Speed (m/s)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_multi(self, data):
        """Plot multiple metrics with different units"""
        # Speed plot with multiple units
        ax1 = self.plot_figure.add_subplot(311)
        times = [d['time'] for d in data]
        speeds_mps = [d['speed'] for d in data]
        speeds_kph = [d['speed_kph'] for d in data]
        speeds_mph = [d['speed_mph'] for d in data]
        
        # Create twin axes for different units
        ax1_kph = ax1.twinx()
        ax1_mph = ax1.twinx()
        
        # Offset the right spine of ax1_mph
        ax1_mph.spines['right'].set_position(('outward', 60))
        
        # Plot all three units
        line1 = ax1.plot(times, speeds_mps, 'b-', label='m/s', linewidth=2)
        line2 = ax1_kph.plot(times, speeds_kph, 'g--', label='km/h', alpha=0.7)
        line3 = ax1_mph.plot(times, speeds_mph, 'r:', label='mph', alpha=0.7)
        
        ax1.set_ylabel('Speed (m/s)', color='b')
        ax1_kph.set_ylabel('Speed (km/h)', color='g')
        ax1_mph.set_ylabel('Speed (mph)', color='r')
        
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_kph.tick_params(axis='y', labelcolor='g')
        ax1_mph.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Speed in Multiple Units")
        
        # Distance plot
        ax2 = self.plot_figure.add_subplot(312, sharex=ax1)
        distances = [d['distance'] for d in data]
        ax2.plot(times, distances, 'purple', label='Distance', linewidth=2)
        ax2.set_ylabel('Distance (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Confidence plot
        ax3 = self.plot_figure.add_subplot(313, sharex=ax1)
        confidences = [d['confidence'] for d in data]
        ax3.plot(times, confidences, 'orange', label='Confidence', linewidth=2)
        ax3.fill_between(times, 0, confidences, alpha=0.3, color='orange')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add threshold lines
        ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.4, color='red', linestyle='--', alpha=0.5)
        
        self.plot_figure.suptitle("Multi-Metric Analysis", fontsize=14)
        
    def _plot_trajectory(self, data):
        """Plot object trajectory"""
        ax = self.plot_figure.add_subplot(111)
        
        x_positions = [d['bbox_center_x'] for d in data]
        y_positions = [d['bbox_center_y'] for d in data]
        speeds = [d['speed'] for d in data]
        
        # Create color map based on speed
        scatter = ax.scatter(x_positions, y_positions, c=speeds, cmap='jet', s=10)
        
        # Add colorbar
        cbar = self.plot_figure.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (m/s)', fontsize=12)
        
        ax.set_title("Object Trajectory (Colored by Speed)", fontsize=14)
        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
        ax.grid(True, alpha=0.3)
        
    def _plot_confidence(self, data):
        """Plot tracking confidence over time"""
        ax = self.plot_figure.add_subplot(111)
        
        times = [d['time'] for d in data]
        confidences = [d['confidence'] for d in data]
        
        ax.plot(times, confidences, 'g-', linewidth=2)
        ax.fill_between(times, 0, confidences, alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=0.7, color='orange', linestyle='--', label='Good Tracking')
        ax.axhline(y=0.4, color='red', linestyle='--', label='Poor Tracking')
        
        ax.set_title("Tracking Confidence Over Time", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def update_distance_plot(self):
        """Update distance tracking plots"""
        if not hasattr(self.engine, 'distance_history') or not self.engine.distance_history:
            self.status_bar.config(text="No distance data to plot")
            return
            
        self.distance_figure.clear()
        
        # Create subplots
        ax1 = self.distance_figure.add_subplot(311)
        ax2 = self.distance_figure.add_subplot(312)
        ax3 = self.distance_figure.add_subplot(313)
        
        # Extract data
        data = list(self.engine.distance_history)
        frames = [d['frame'] for d in data]
        distances = [d['distance'] for d in data]
        sizes = [d['bbox_size'] for d in data]
        
        # Plot distance over time
        ax1.plot(frames, distances, 'b-', linewidth=2)
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Object Distance Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot apparent size
        ax2.plot(frames, sizes, 'r-', linewidth=2)
        ax2.set_ylabel('Apparent Size (pixels)')
        ax2.set_title('Object Apparent Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot speed with distance
        if self.telemetry_data:
            telemetry = list(self.telemetry_data)
            t_frames = [t['frame'] for t in telemetry]
            speeds = [t['speed'] for t in telemetry]
            
            # Interpolate distances to match telemetry frames
            interp_distances = np.interp(t_frames, frames, distances)
            
            # Create twin axis for distance
            ax3_twin = ax3.twinx()
            
            ax3.plot(t_frames, speeds, 'g-', linewidth=2, label='Speed')
            ax3_twin.plot(t_frames, interp_distances, 'b--', alpha=0.5, label='Distance')
            
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Speed (m/s)', color='g')
            ax3_twin.set_ylabel('Distance (m)', color='b')
            ax3.set_title('Speed vs Distance')
            ax3.grid(True, alpha=0.3)
            
        self.distance_figure.tight_layout()
        self.distance_canvas.draw()
        
        self.status_bar.config(text=f"Distance plot updated: {len(data)} measurements")
        
    def export_telemetry(self):
        """Export telemetry data to CSV"""
        if not self.telemetry_data:
            messagebox.showwarning("No Data", "No telemetry data to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as csvfile:
                    if self.telemetry_data:
                        fieldnames = list(self.telemetry_data[0].keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.telemetry_data)
                        
                messagebox.showinfo("Success", f"Telemetry data exported to {filename}")
                self.status_bar.config(text=f"Exported {len(self.telemetry_data)} data points")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
                
    def toggle_manual_distance(self):
        """Toggle manual distance entry"""
        if self.manual_distance_var.get():
            self.distance_entry.config(state=tk.NORMAL)
            self.config['manual_distance'] = True
        else:
            self.distance_entry.config(state=tk.DISABLED)
            self.config['manual_distance'] = False
            
    def toggle_mht(self):
        """Toggle Multi-Hypothesis Tracking"""
        self.engine.use_mht = self.mht_var.get()
        if self.engine.use_mht:
            self.engine.multi_tracker = MultiHypothesisTracker(
                n_hypotheses=3,
                fps=self.config['fps'],
                frame_skip=self.config.get('frame_skip', 1)
            )
            self.status_bar.config(text="Multi-Hypothesis Tracking enabled")
        else:
            self.status_bar.config(text="Single Kalman Filter tracking")
            
    def toggle_stabilization(self):
        """Toggle video stabilization"""
        self.config['stabilization_enabled'] = self.stab_var.get()
        self.status_bar.config(text=f"Stabilization {'enabled' if self.stab_var.get() else 'disabled'}")
        
    def toggle_gpu(self):
        """Toggle GPU acceleration"""
        self.config['use_gpu'] = self.gpu_var.get()
        self.gpu_processor = GPUVideoProcessor(use_gpu=self.config['use_gpu'])
        self.engine.gpu_processor = self.gpu_processor
        self.status_bar.config(text=f"GPU acceleration {'enabled' if self.gpu_var.get() else 'disabled'}")
        
    def toggle_multicore(self):
        """Toggle multicore processing"""
        self.config['use_multicore'] = self.multicore_var.get()
        self.status_bar.config(text=f"Multicore processing {'enabled' if self.multicore_var.get() else 'disabled'}")
        
    def toggle_highspeed(self):
        """Toggle high-speed mode"""
        self.config['high_speed_mode'] = self.highspeed_var.get()
        if self.highspeed_var.get():
            self.config['max_expected_speed'] = 10000.0  # 10 km/s for hypersonic
            self.status_bar.config(text="High-speed mode enabled (hypersonic tracking)")
        else:
            self.config['max_expected_speed'] = 5000.0
            self.status_bar.config(text="Standard speed mode")
    
    def toggle_size_mode(self):
        """Toggle between database and manual size entry"""
        if self.size_mode_var.get() == "manual":
            self.manual_width_entry.config(state=tk.NORMAL)
            self.manual_length_entry.config(state=tk.NORMAL)
            self.config['use_manual_size'] = True
            self.config['manual_width_value'] = self.manual_width_var.get()
            self.config['manual_length_value'] = self.manual_length_var.get()
        else:
            self.manual_width_entry.config(state=tk.DISABLED)
            self.manual_length_entry.config(state=tk.DISABLED)
            self.config['use_manual_size'] = False
        self.update_size_display()

    def apply_size_preset(self, width, length):
        """Apply a preset size"""
        self.size_mode_var.set("manual")
        self.toggle_size_mode()
        self.manual_width_var.set(str(width))
        self.manual_length_var.set(str(length))
        self.config['manual_width_value'] = str(width)
        self.config['manual_length_value'] = str(length)
        self.update_size_display()

    def update_size_display(self):
        """Update the display of current object size"""
        if self.config.get('use_manual_size', False):
            try:
                width = float(self.manual_width_var.get())
                length = float(self.manual_length_var.get())
                self.size_display_label.config(text=f"Current size: {width:.2f}m × {length:.2f}m (Manual)")
            except:
                self.size_display_label.config(text="Invalid manual size values")
        else:
            obj_data = self.config['object_database'].get(self.config['selected_object'], {})
            width = obj_data.get('wingspan', obj_data.get('width', 'N/A'))
            length = obj_data.get('length', obj_data.get('height', 'N/A'))
            self.size_display_label.config(text=f"Current size: {width}m × {length}m (Database)")
    
    def get_object_real_size(self):
        """Get object real size from database or manual entry"""
        if self.config.get('use_manual_size', False):
            try:
                if self.config['use_wingspan']:
                    return float(self.config.get('manual_width_value', '10.0'))
                else:
                    return float(self.config.get('manual_length_value', '10.0'))
            except:
                # Fallback to database
                pass
        
        # Use database values
        obj_data = self.config['object_database'].get(self.config['selected_object'], {})
        if self.config['use_wingspan']:
            return obj_data.get('wingspan', obj_data.get('width', obj_data.get('rotor_diameter', 10.0)))
        else:
            return obj_data.get('length', obj_data.get('height', obj_data.get('diameter', 10.0)))
            
    def on_category_changed(self, event=None):
        """Handle category filter change"""
        category = self.category_var.get()
        if category == "All":
            objects = list(self.config['object_database'].keys())
        else:
            objects = [name for name, data in self.config['object_database'].items()
                      if data['category'] == category]
        
        self.object_combo['values'] = sorted(objects)
        if objects and self.object_var.get() not in objects:
            self.object_var.set(objects[0])
            self.on_object_changed()
            
    def on_object_changed(self, event=None):
        """Handle object selection change"""
        self.config['selected_object'] = self.object_var.get()
        self.engine._update_motion_model()
        
        # Update display info
        obj_data = self.config['object_database'].get(self.config['selected_object'], {})
        info_text = f"Selected: {self.config['selected_object']}"
        if 'wingspan' in obj_data:
            info_text += f" (Wingspan: {obj_data['wingspan']}m)"
        elif 'width' in obj_data:
            info_text += f" (Width: {obj_data['width']}m)"
        if 'length' in obj_data:
            info_text += f" (Length: {obj_data['length']}m)"
        elif 'height' in obj_data:
            info_text += f" (Height: {obj_data['height']}m)"
        
        self.status_bar.config(text=info_text)
        
    def update_measurement_type(self):
        """Update measurement type preference"""
        self.config['use_wingspan'] = (self.measure_var.get() == "wingspan")
        
    def on_camera_type_changed(self, event=None):
        """Handle camera type change"""
        camera_type = self.camera_type_var.get()
        
        if camera_type == "Custom":
            self.sensor_model_combo.config(state="disabled")
            self.sensor_entry.config(state=tk.NORMAL)
            self.sensor_model_var.set("Custom")
        else:
            self.sensor_model_combo.config(state="readonly")
            if camera_type in self.config['sensor_database']:
                models = list(self.config['sensor_database'][camera_type].keys())
                self.sensor_model_combo['values'] = models
                if models:
                    self.sensor_model_var.set(models[0])
                    self.on_sensor_model_changed()
                    
    def on_sensor_model_changed(self, event=None):
        """Handle sensor model change with focal length update"""
        camera_type = self.camera_type_var.get()
        sensor_model = self.sensor_model_var.get()
        
        if camera_type in self.config['sensor_database'] and sensor_model in self.config['sensor_database'][camera_type]:
            sensor_data = self.config['sensor_database'][camera_type][sensor_model]
            
            # Update sensor width
            self.sensor_var.set(str(sensor_data['width']))
            
            # Update focal length if available
            if sensor_data.get('focal_length'):
                self.focal_var.set(str(sensor_data['focal_length']))
                self.status_bar.config(text=f"Sensor: {sensor_model} ({sensor_data['width']}mm × {sensor_data.get('height', 'N/A')}mm, f={sensor_data['focal_length']}mm)")
            else:
                self.status_bar.config(text=f"Sensor: {sensor_model} ({sensor_data['width']}mm × {sensor_data.get('height', 'N/A')}mm)")
            
            # Update sensor height in config
            self.config['sensor_height_mm'] = sensor_data.get('height', sensor_data['width'] * 0.667)
            
            self.sensor_entry.config(state=tk.DISABLED)
            
    def update_diagnostics(self):
        """Update diagnostic display"""
        try:
            # Calculate focal length in pixels
            image_width = self.config.get('frame_width', 1920)
            focal_px = (self.config['focal_length_mm'] * image_width) / self.config['sensor_width_mm']
            self.focal_px_label.config(text=f"{focal_px:.1f}")
            
            # Calculate pixel angular size
            sensor_height_mm = self.config.get('sensor_height_mm', self.config['sensor_width_mm'] * 0.667)
            image_height = self.config.get('frame_height', 1080)
            pixel_angle_mrad = (sensor_height_mm / self.config['focal_length_mm']) / image_height * 1000
            self.pixel_angle_label.config(text=f"{pixel_angle_mrad:.3f}")
            
            # Display current distance
            distance = self.config.get('estimated_distance', 0)
            self.distance_label.config(text=f"{distance:.1f}")
            
            # Calculate pixel to meter at current distance
            if distance > 0:
                pixel_to_meter = distance * (pixel_angle_mrad / 1000)
                self.pixel_meter_label.config(text=f"{pixel_to_meter:.3f}")
            else:
                self.pixel_meter_label.config(text="N/A")
                
        except Exception as e:
            self.status_bar.config(text=f"Diagnostic update error: {str(e)}")
            
    def apply_camera_settings(self):
        """Apply camera parameter changes"""
        try:
            self.config['focal_length_mm'] = float(self.focal_var.get())
            self.config['sensor_width_mm'] = float(self.sensor_var.get())
            self.status_bar.config(text="Camera settings applied successfully")
            self.update_diagnostics()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values")
            
    def open_video(self):
        """Open video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if filename:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(filename)
            ret, frame = self.cap.read()
            
            if ret:
                # Reset flags and clear old data
                self.config['video_finished'] = False
                self.initial_bbox = None
                self.clear_plot_data()
                
                self.config['video_loaded'] = True
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.config['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
                self.config['frame_width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.config['frame_height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Enable controls
                self.play_button.config(state=tk.NORMAL)
                self.select_button.config(state=tk.NORMAL)
                
                # Display first frame
                self.display_frame(frame, frame)
                
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                self.status_bar.config(text=f"Video loaded: {os.path.basename(filename)} ({self.total_frames} frames @ {self.config['fps']:.1f} fps)")
            else:
                messagebox.showerror("Error", "Failed to read video file")
                
    def reset_and_replay(self):
        """Resets the video to the beginning and re-initializes tracking if possible"""
        if not self.cap:
            return
            
        self.status_bar.config(text="Replaying video...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.config['video_finished'] = False
        
        # Clear old data for the new run
        self.config['speed_history'].clear()
        self.telemetry_data.clear()
        
        # Clear engine history
        if hasattr(self.engine, 'distance_history'):
            self.engine.distance_history.clear()
        if hasattr(self.engine, 'trajectory_points'):
            self.engine.trajectory_points.clear()

        # Re-initialize tracking with the stored bounding box
        if self.initial_bbox:
            ret, frame = self.cap.read()
            if ret:
                # Initialize OpenCV tracker
                tracker_type = self.tracker_type_var.get()
                
                if tracker_type == 'CSRT':
                    self.config['tracker'] = cv2.TrackerCSRT_create()
                elif tracker_type == 'KCF':
                    self.config['tracker'] = cv2.TrackerKCF_create()
                elif tracker_type == 'MOSSE':
                    self.config['tracker'] = cv2.TrackerMOSSE_create()
                elif tracker_type == 'MIL':
                    self.config['tracker'] = cv2.TrackerMIL_create()
                
                self.config['tracker'].init(frame, self.initial_bbox)
                self.config['tracking'] = True
                
                # Re-initialize the Kalman filter
                if hasattr(self.engine, 'single_tracker'):
                    self.engine.single_tracker.initialize(self.initial_bbox)
                    
                # Re-initialize distance tracking
                initial_distance = None
                if self.config.get('manual_distance', False):
                    try:
                        initial_distance = float(self.distance_var.get())
                    except:
                        initial_distance = None
                        
                self.engine.initialize_distance_tracking(self.initial_bbox, initial_distance)
                
                self.status_bar.config(text="Replaying video and re-acquiring track.")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind again after reading one frame
            else:
                self.config['tracking'] = False
        else:
            self.config['tracking'] = False  # Cannot track if no initial object was selected
            
    def select_object(self):
        """Select object to track with better tracker initialization"""
        if not self.config['video_loaded']:
            return
            
        self.config['paused'] = True
        self.play_button.config(text="Play")
        
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Let user select ROI
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object")
        
        if bbox[2] > 0 and bbox[3] > 0:
            # Store the first manually selected bbox
            self.initial_bbox = bbox
            
            # Initialize tracker based on selection
            tracker_type = self.tracker_type_var.get()
            
            if tracker_type == 'CSRT':
                self.config['tracker'] = cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                self.config['tracker'] = cv2.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                self.config['tracker'] = cv2.TrackerMOSSE_create()
            elif tracker_type == 'MIL':
                self.config['tracker'] = cv2.TrackerMIL_create()
            
            self.config['tracker'].init(frame, bbox)
            self.config['tracking'] = True
            self.config['bbox'] = bbox
            
            # Initialize Kalman filter
            self.engine.single_tracker.initialize(bbox)
            
            # Initialize distance tracking
            initial_distance = None
            if self.config.get('manual_distance', False):
                try:
                    initial_distance = float(self.distance_var.get())
                    self.config['manual_distance_value'] = self.distance_var.get()
                except:
                    initial_distance = None
                    
            self.engine.initialize_distance_tracking(bbox, initial_distance)
            
            # Estimate initial distance if not manual
            if not self.config.get('manual_distance', False):
                self._estimate_initial_distance(bbox)
            
            self.status_bar.config(text=f"Object selected: {bbox[2]}x{bbox[3]} pixels")
            
        # Reset to frame where selection was made
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        
    def _estimate_initial_distance(self, bbox):
        """Estimate initial distance based on object size and camera parameters"""
        # Get object real size
        real_size_m = self.get_object_real_size()
        
        if self.config['use_wingspan']:
            pixel_size = bbox[2]  # width
        else:
            pixel_size = bbox[3]  # height
        
        # Calculate distance using pinhole camera model
        # Distance = (Real Size × Focal Length in pixels) / Pixel Size
        image_width = self.config.get('frame_width', 1920)
        focal_length_px = (self.config['focal_length_mm'] * image_width) / self.config['sensor_width_mm']
        
        estimated_distance = (real_size_m * focal_length_px) / pixel_size
        
        # Sanity check the distance
        if estimated_distance < 10:  # Less than 10 meters seems unlikely
            estimated_distance = 100  # Default to 100m
            messagebox.showwarning("Distance Warning", 
                                 f"Calculated distance ({estimated_distance:.1f}m) seems too small. "
                                 f"Please check object size and camera settings, or use manual distance.")
        elif estimated_distance > 50000:  # More than 50km seems unlikely
            estimated_distance = 5000  # Default to 5km
            messagebox.showwarning("Distance Warning", 
                                 f"Calculated distance ({estimated_distance:.1f}m) seems too large. "
                                 f"Please check object size and camera settings, or use manual distance.")
        
        self.config['estimated_distance'] = estimated_distance
        self.distance_var.set(f"{estimated_distance:.0f}")
        
    def toggle_play(self):
        """Toggle play/pause, with added logic for replaying a finished video"""
        if not self.config['video_loaded']:
            return

        # If video is finished and user hits play, reset and replay
        if self.config['video_finished'] and self.config['paused']:
            self.reset_and_replay()
            self.config['paused'] = False  # Set to play
        else:
            self.config['paused'] = not self.config['paused']
        
        self.play_button.config(text="Pause" if not self.config['paused'] else "Play")
        
        if not self.config['paused']:
            self.stop_thread.clear()
            self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
            self.processing_thread.start()
        else:
            self.stop_processing()
            
    def stop_processing(self):
        """Stop video processing"""
        self.stop_thread.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def process_video(self):
        """Process video frames in a separate thread"""
        frame_count = 0
        
        while not self.config['paused'] and not self.stop_thread.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # When video ends, set the finished flag and pause
                self.config['paused'] = True
                self.config['video_finished'] = True
                self.root.after(0, lambda: self.play_button.config(text="Play Again"))
                self.status_bar.config(text="Video finished. Press 'Play Again' to replay.")
                break
                
            # Get frame number for telemetry
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.config['current_frame_num'] = frame_num
            
            # Process frame with engine
            original_frame = frame.copy()
            original_processed, processed_frame = self.engine.process_frame(
                frame, original_frame, frame_count, frame_num
            )
            
            # Queue frames for display
            try:
                self.frame_queue.put((original_processed, processed_frame), block=False)
            except queue.Full:
                pass
                
            # Update progress
            progress = (frame_num / self.total_frames) * 100
            self.progress_var.set(progress)
            
            frame_count += 1
            
            # Frame skip for performance
            if self.config.get('frame_skip', 1) > 1:
                for _ in range(self.config['frame_skip'] - 1):
                    self.cap.read()
                    
    def update_gui_timer(self):
        """Timer to update GUI elements"""
        # Update video display
        try:
            original, processed = self.frame_queue.get_nowait()
            self.display_frame(original, processed)
        except queue.Empty:
            pass
            
        # Process GUI update queue
        while True:
            update = self.config['gui_update_queue'].get()
            if not update:
                break
                
            if update['type'] == 'speed':
                self.speed_display_mps.config(text=f"{update['mps']:.1f} m/s")
                self.speed_display_kph.config(text=f"{update['kph']:.1f} km/h")
                self.speed_display_mph.config(text=f"{update['mph']:.1f} mph")
                
                if update.get('mach') is not None:
                    self.speed_display_mach.config(text=f"Mach {update['mach']:.2f}")
                else:
                    self.speed_display_mach.config(text="")
                    
            elif update['type'] == 'telemetry':
                self.telemetry_data.append(update['data'])
                
            elif update['type'] == 'confidence':
                # Could add confidence display to GUI
                pass
                
            elif update['type'] == 'distance':
                # Update diagnostics if visible
                if hasattr(self, 'distance_label'):
                    self.distance_label.config(text=f"{update['distance']:.1f}")
                
        # Schedule next update
        self.root.after(30, self.update_gui_timer)
        
    def display_frame(self, original_frame, processed_frame):
        """Display frames on canvases"""
        # Convert frames to PhotoImage
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas_original.winfo_width()
        canvas_height = self.canvas_original.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scale to fit
            scale = min(canvas_width / original_frame.shape[1],
                       canvas_height / original_frame.shape[0])
            
            new_width = int(original_frame.shape[1] * scale)
            new_height = int(original_frame.shape[0] * scale)
            
            original_resized = cv2.resize(original_rgb, (new_width, new_height))
            processed_resized = cv2.resize(processed_rgb, (new_width, new_height))
            
            # Convert to PIL Image then to PhotoImage
            original_pil = Image.fromarray(original_resized)
            processed_pil = Image.fromarray(processed_resized)
            
            self.photo_original = ImageTk.PhotoImage(original_pil)
            self.photo_processed = ImageTk.PhotoImage(processed_pil)
            
            # Display on canvases
            self.canvas_original.delete("all")
            self.canvas_processed.delete("all")
            
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.canvas_original.create_image(x, y, anchor=tk.NW, image=self.photo_original)
            self.canvas_processed.create_image(x, y, anchor=tk.NW, image=self.photo_processed)
            
    def on_closing(self):
        """Handle window closing"""
        self.stop_processing()
        
        if self.cap:
            self.cap.release()
            
        if self.preprocessor_pool:
            self.preprocessor_pool.shutdown(wait=False)
            
        cv2.destroyAllWindows()
        self.root.destroy()
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Matplotlib not found. Graphing features will be disabled.")
        print("Install with: pip install matplotlib")
    
    app = AdvancedSpeedEstimator()
    app.run()