"""
Compatibility wrapper for stability predictor models.
This file provides backward compatibility for imports.
"""

# Import the main dual-branch model as the default stability predictor
from .dual_branch_stability_predictor import DualBranchStopeStabilityPredictor as StopeStabilityPredictor

# Export the main classes
__all__ = [
    'StopeStabilityPredictor',
]
