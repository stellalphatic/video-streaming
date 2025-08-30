# app/utils/performance_monitor.py - Performance monitoring for video service
import time
import threading
import psutil
import torch
from typing import Dict, List
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance metrics
        self.frame_times = deque(maxlen=100)
        self.audio_processing_times = deque(maxlen=100)
        self.session_metrics = {}
        
        # System metrics
        self.cpu_usage_history = deque(maxlen=60)  # 1 minute history
        self.memory_usage_history = deque(maxlen=60)
        self.gpu_memory_history = deque(maxlen=60)
        
        # Performance thresholds
        self.thresholds = {
            'max_frame_time_ms': 50,  # 20 FPS minimum
            'max_cpu_usage': 80,
            'max_memory_usage': 85,
            'max_gpu_memory_usage': 90
        }
        
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def record_frame_time(self, session_id: str, processing_time: float):
        """Record frame processing time"""
        self.frame_times.append(processing_time)
        
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = {
                'frame_times': deque(maxlen=50),
                'audio_times': deque(maxlen=50),
                'start_time': time.time(),
                'frame_count': 0,
                'audio_count': 0
            }
        
        self.session_metrics[session_id]['frame_times'].append(processing_time)
        self.session_metrics[session_id]['frame_count'] += 1
    
    def record_audio_processing_time(self, session_id: str, processing_time: float):
        """Record audio processing time"""
        self.audio_processing_times.append(processing_time)
        
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = {
                'frame_times': deque(maxlen=50),
                'audio_times': deque(maxlen=50),
                'start_time': time.time(),
                'frame_count': 0,
                'audio_count': 0
            }
        
        self.session_metrics[session_id]['audio_times'].append(processing_time)
        self.session_metrics[session_id]['audio_count'] += 1
    
    def report_fps(self, session_id: str, fps: float):
        """Report FPS for a session"""
        if session_id in self.session_metrics:
            self.session_metrics[session_id]['current_fps'] = fps
    
    def get_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
        
        # Frame processing metrics
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        avg_audio_time = sum(self.audio_processing_times) / len(self.audio_processing_times) if self.audio_processing_times else 0
        
        # Session metrics
        active_sessions = len(self.session_metrics)
        total_frames = sum(metrics['frame_count'] for metrics in self.session_metrics.values())
        total_audio_chunks = sum(metrics['audio_count'] for metrics in self.session_metrics.values())
        
        return {
            'timestamp': current_time,
            'system': {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'gpu_memory_used_gb': gpu_memory_used / (1024**3),
                'gpu_memory_total_gb': gpu_memory_total / (1024**3),
                'gpu_memory_percent': (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
            },
            'processing': {
                'avg_frame_time_ms': avg_frame_time * 1000,
                'avg_audio_time_ms': avg_audio_time * 1000,
                'estimated_fps': 1.0 / avg_frame_time if avg_frame_time > 0 else 0,
                'frames_processed': len(self.frame_times),
                'audio_chunks_processed': len(self.audio_processing_times)
            },
            'sessions': {
                'active_count': active_sessions,
                'total_frames_generated': total_frames,
                'total_audio_chunks_processed': total_audio_chunks
            },
            'health': self._get_health_status()
        }
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        if session_id not in self.session_metrics:
            return {}
        
        metrics = self.session_metrics[session_id]
        current_time = time.time()
        session_duration = current_time - metrics['start_time']
        
        avg_frame_time = sum(metrics['frame_times']) / len(metrics['frame_times']) if metrics['frame_times'] else 0
        avg_audio_time = sum(metrics['audio_times']) / len(metrics['audio_times']) if metrics['audio_times'] else 0
        
        return {
            'session_id': session_id,
            'duration_seconds': session_duration,
            'frame_count': metrics['frame_count'],
            'audio_count': metrics['audio_count'],
            'avg_frame_time_ms': avg_frame_time * 1000,
            'avg_audio_time_ms': avg_audio_time * 1000,
            'estimated_fps': 1.0 / avg_frame_time if avg_frame_time > 0 else 0,
            'current_fps': metrics.get('current_fps', 0),
            'frames_per_second': metrics['frame_count'] / session_duration if session_duration > 0 else 0
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up metrics for a session"""
        if session_id in self.session_metrics:
            del self.session_metrics[session_id]
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory.percent)
                
                # Collect GPU metrics
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_percent = (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
                    self.gpu_memory_history.append(gpu_percent)
                
                # Check for performance issues
                self._check_performance_alerts()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(5.0)
    
    def _check_performance_alerts(self):
        """Check for performance issues and log alerts"""
        current_stats = self.get_stats()
        
        # Check CPU usage
        if current_stats['system']['cpu_usage_percent'] > self.thresholds['max_cpu_usage']:
            logger.warning(f"High CPU usage: {current_stats['system']['cpu_usage_percent']:.1f}%")
        
        # Check memory usage
        if current_stats['system']['memory_usage_percent'] > self.thresholds['max_memory_usage']:
            logger.warning(f"High memory usage: {current_stats['system']['memory_usage_percent']:.1f}%")
        
        # Check GPU memory usage
        if current_stats['system']['gpu_memory_percent'] > self.thresholds['max_gpu_memory_usage']:
            logger.warning(f"High GPU memory usage: {current_stats['system']['gpu_memory_percent']:.1f}%")
        
        # Check frame processing time
        if current_stats['processing']['avg_frame_time_ms'] > self.thresholds['max_frame_time_ms']:
            logger.warning(f"Slow frame processing: {current_stats['processing']['avg_frame_time_ms']:.1f}ms")
    
    def _get_health_status(self) -> str:
        """Get overall system health status"""
        current_stats = self.get_stats()
        
        # Check various metrics
        issues = []
        
        if current_stats['system']['cpu_usage_percent'] > self.thresholds['max_cpu_usage']:
            issues.append("high_cpu")
        
        if current_stats['system']['memory_usage_percent'] > self.thresholds['max_memory_usage']:
            issues.append("high_memory")
        
        if current_stats['system']['gpu_memory_percent'] > self.thresholds['max_gpu_memory_usage']:
            issues.append("high_gpu_memory")
        
        if current_stats['processing']['avg_frame_time_ms'] > self.thresholds['max_frame_time_ms']:
            issues.append("slow_processing")
        
        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return f"warning_{issues[0]}"
        else:
            return "critical_multiple_issues"
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current performance"""
        suggestions = []
        current_stats = self.get_stats()
        
        if current_stats['system']['cpu_usage_percent'] > 70:
            suggestions.append("Consider reducing the number of concurrent sessions or upgrading CPU")
        
        if current_stats['system']['memory_usage_percent'] > 80:
            suggestions.append("Consider reducing memory usage by clearing caches or upgrading RAM")
        
        if current_stats['system']['gpu_memory_percent'] > 85:
            suggestions.append("Consider reducing video quality or batch size to save GPU memory")
        
        if current_stats['processing']['avg_frame_time_ms'] > 40:
            suggestions.append("Consider optimizing video processing pipeline or upgrading GPU")
        
        if current_stats['processing']['estimated_fps'] < 15:
            suggestions.append("Frame rate too low - check processing pipeline bottlenecks")
        
        if not suggestions:
            suggestions.append("System performance is optimal")
        
        return suggestions