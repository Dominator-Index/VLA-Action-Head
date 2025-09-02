"""
Action Heads Package

This package contains various action head implementations for VLA models.
The action heads are organized into separate modules for better maintainability.
"""

# Import all action heads for easy access
from .regression_heads import L1RegressionActionHead, VAEActionHead
from .diffusion_heads import DiffusionActionHead, EndToEndDiffusionActionHead
from .flow_matching_heads import (
    FlowMatchingActionHead, 
    OTFlowMatchingActionHead, 
    COTFlowMatchingActionHead
)
from .mean_flow_heads import MeanFlowActionHead
from .convex_flow_heads import ConvexFlowActionHead
from .shortcut_heads import ShortcutActionHead
from .normalizing_flow_heads import NormalizingFlowActionHead

# Export all action heads
__all__ = [
    'L1RegressionActionHead',
    'VAEActionHead', 
    'DiffusionActionHead',
    'EndToEndDiffusionActionHead',
    'FlowMatchingActionHead',
    'OTFlowMatchingActionHead',
    'COTFlowMatchingActionHead',
    'MeanFlowActionHead',
    'ConvexFlowActionHead',
    'ShortcutActionHead',
    'NormalizingFlowActionHead',
]