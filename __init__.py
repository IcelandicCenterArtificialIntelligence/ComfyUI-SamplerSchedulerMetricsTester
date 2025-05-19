# __init__.py for the SamplerSchedulerMetricsTester node

# Import the node class and mappings from your node file
from .sampler_scheduler_metrics_tester import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose them to ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("Loading SamplerSchedulerMetricsTester custom node...")
