"""TensorBoard logging utility for training"""

import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir='./logs', experiment_name='experiment'):
        """Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save logs (default: ./logs)
            experiment_name: Name of the experiment (creates subdirectory)
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalar values"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram"""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """Log text"""
        self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush buffered logs"""
        self.writer.flush()
    
    def close(self):
        """Close writer"""
        self.writer.close()
    
    def get_log_dir(self):
        """Get log directory path"""
        return str(self.log_dir)


def create_logger(log_dir='./logs', experiment_name='experiment'):
    """Create and return logger instance
    
    Usage:
        logger = create_logger(experiment_name='exp_v1')
        logger.log_scalar('loss/train', 0.5, 0)
        logger.log_scalar('reward/episode', 10.5, 1)
    """
    return Logger(log_dir, experiment_name)
