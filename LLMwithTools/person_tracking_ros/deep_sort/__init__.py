"""
DeepSORT tracking module for person tracking ROS package.
"""

__version__ = "1.0.0"
__author__ = "person_tracking_ros"

# 导入主要的DeepSORT类以便直接访问
from .deep_sort.deep_sort import DeepSort

__all__ = ["DeepSort"]