"""
Background Event Processing Queue System
Handles asynchronous processing of operational events and impact score updates
"""

import threading
import queue
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist

from ..models import OperationalEvent, ImpactScore, Stope
from ..impact.impact_service import ImpactCalculationService

logger = logging.getLogger(__name__)


class EventProcessingQueue:
    """
    Thread-safe queue for processing operational events in the background
    """
    
    def __init__(self, max_workers: int = 3, max_queue_size: int = 1000):
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.impact_service = ImpactCalculationService()
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Thread lock for statistics
        self._stats_lock = threading.Lock()
    
    def start(self):
        """Start the background processing workers"""
        if self.running:
            logger.warning("Event processing queue is already running")
            return
        
        self.running = True
        self.start_time = timezone.now()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"EventProcessor-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} event processing workers")
    
    def stop(self, timeout: float = 30.0):
        """Stop the background processing workers"""
        if not self.running:
            return
        
        self.running = False
        
        # Add stop signals to queue for each worker
        for _ in range(self.max_workers):
            try:
                self.event_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout/self.max_workers)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop gracefully")
        
        self.workers.clear()
        logger.info("Event processing queue stopped")
    
    def queue_event(self, event: OperationalEvent, priority: str = 'normal') -> bool:
        """
        Queue an event for background processing
        
        Args:
            event: OperationalEvent instance to process
            priority: Processing priority ('high', 'normal', 'low')
            
        Returns:
            True if queued successfully, False if queue is full
        """
        try:
            task = {
                'type': 'process_event',
                'event_id': event.id,
                'priority': priority,
                'queued_at': timezone.now(),
                'retry_count': 0
            }
            
            self.event_queue.put(task, timeout=1.0)
            logger.debug(f"Queued event {event.id} for processing")
            return True
            
        except queue.Full:
            logger.error(f"Event queue is full, dropping event {event.id}")
            return False
    
    def queue_batch_update(self, stope_ids: List[int], priority: str = 'low') -> bool:
        """
        Queue a batch update task
        """
        try:
            task = {
                'type': 'batch_update',
                'stope_ids': stope_ids,
                'priority': priority,
                'queued_at': timezone.now(),
                'retry_count': 0
            }
            
            self.event_queue.put(task, timeout=1.0)
            logger.debug(f"Queued batch update for {len(stope_ids)} stopes")
            return True
            
        except queue.Full:
            logger.error("Event queue is full, dropping batch update task")
            return False
    
    def _worker_loop(self):
        """Main worker loop for processing events"""
        worker_name = threading.current_thread().name
        logger.info(f"Event processing worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = self.event_queue.get(timeout=1.0)
                
                # Check for stop signal
                if task is None:
                    break
                
                # Process the task
                self._process_task(task, worker_name)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                # Timeout is normal, continue loop
                continue
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_name}: {str(e)}")
        
        logger.info(f"Event processing worker {worker_name} stopped")
    
    def _process_task(self, task: Dict[str, Any], worker_name: str):
        """Process a single task"""
        task_type = task.get('type')
        
        try:
            if task_type == 'process_event':
                self._process_single_event(task, worker_name)
            elif task_type == 'batch_update':
                self._process_batch_update(task, worker_name)
            else:
                logger.error(f"Unknown task type: {task_type}")
                self._increment_error_count()
                
        except Exception as e:
            logger.error(f"Error processing task in {worker_name}: {str(e)}")
            self._increment_error_count()
            
            # Handle retry logic
            self._handle_task_retry(task, str(e))
    
    def _process_single_event(self, task: Dict[str, Any], worker_name: str):
        """Process a single operational event"""
        event_id = task['event_id']
        
        try:
            with transaction.atomic():
                # Get the event
                event = OperationalEvent.objects.get(id=event_id)
                
                # Process the event
                result = self.impact_service.process_single_event(event)
                
                # Log success
                processing_time = timezone.now() - task['queued_at']
                logger.info(
                    f"Worker {worker_name} processed event {event_id} "
                    f"(stope {event.stope_id}) in {processing_time.total_seconds():.2f}s"
                )
                
                self._increment_processed_count()
                
        except ObjectDoesNotExist:
            logger.error(f"Event {event_id} not found")
            self._increment_error_count()
        except Exception as e:
            logger.error(f"Error processing event {event_id}: {str(e)}")
            raise
    
    def _process_batch_update(self, task: Dict[str, Any], worker_name: str):
        """Process batch update task"""
        stope_ids = task['stope_ids']
        
        try:
            start_time = timezone.now()
            results = self.impact_service.batch_update_impact_scores(stope_ids)
            processing_time = timezone.now() - start_time
            
            logger.info(
                f"Worker {worker_name} completed batch update for "
                f"{len(stope_ids)} stopes in {processing_time.total_seconds():.2f}s"
            )
            
            self._increment_processed_count()
            
        except Exception as e:
            logger.error(f"Error in batch update: {str(e)}")
            raise
    
    def _handle_task_retry(self, task: Dict[str, Any], error_msg: str):
        """Handle task retry logic"""
        retry_count = task.get('retry_count', 0)
        max_retries = 3
        
        if retry_count < max_retries:
            # Increment retry count and requeue
            task['retry_count'] = retry_count + 1
            task['last_error'] = error_msg
            
            try:
                self.event_queue.put(task, timeout=1.0)
                logger.info(f"Requeued task (attempt {retry_count + 1}/{max_retries})")
            except queue.Full:
                logger.error("Cannot requeue task - queue is full")
        else:
            logger.error(f"Task failed after {max_retries} retries: {error_msg}")
    
    def _increment_processed_count(self):
        """Thread-safe increment of processed count"""
        with self._stats_lock:
            self.processed_count += 1
    
    def _increment_error_count(self):
        """Thread-safe increment of error count"""
        with self._stats_lock:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self._stats_lock:
            uptime = timezone.now() - self.start_time if self.start_time else timedelta(0)
            return {
                'running': self.running,
                'queue_size': self.event_queue.qsize(),
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'uptime_seconds': uptime.total_seconds(),
                'processing_rate': (
                    self.processed_count / uptime.total_seconds() 
                    if uptime.total_seconds() > 0 else 0
                ),
                'error_rate': (
                    self.error_count / (self.processed_count + self.error_count) 
                    if (self.processed_count + self.error_count) > 0 else 0
                )
            }


class EventProcessor:
    """
    High-level interface for event processing
    Manages the processing queue and provides convenient methods
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for event processor"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.queue = EventProcessingQueue()
        self._initialized = True
    
    def start_processing(self):
        """Start the background processing system"""
        self.queue.start()
    
    def stop_processing(self):
        """Stop the background processing system"""
        self.queue.stop()
    
    def queue_event_processing(self, event: OperationalEvent, priority: str = 'normal') -> bool:
        """Queue an event for background processing"""
        return self.queue.queue_event(event, priority)
    
    def queue_batch_processing(self, stope_ids: List[int]) -> bool:
        """Queue a batch update"""
        return self.queue.queue_batch_update(stope_ids)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.queue.get_stats()
    
    def is_running(self) -> bool:
        """Check if the processing system is running"""
        return self.queue.running
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.event_queue.qsize()


# Global instance
event_processor = EventProcessor()
