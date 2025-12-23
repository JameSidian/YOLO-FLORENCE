"""
Flash Attention Stub Module

This is a stub module to bypass flash_attn import requirements for Florence-2 models.
Florence-2 models check for flash_attn during loading, but it may not be strictly
required for CPU inference or can fall back to standard attention.

WARNING: This is a workaround. Actual flash_attn functionality will not work.
For proper performance, flash_attn should be properly installed.
"""

import warnings

# Suppress the warning about missing flash_attn during model loading
warnings.filterwarnings("ignore", message=".*flash_attn.*")

# Create a minimal stub that satisfies the import check
class FlashAttentionFunction:
    """Stub for flash attention function"""
    @staticmethod
    def __call__(*args, **kwargs):
        raise RuntimeError("flash_attn is not actually installed. This is just a stub.")

# Try to provide basic attributes that might be checked
__version__ = "2.0.0"

def flash_attn_func(*args, **kwargs):
    """Stub flash attention function"""
    raise RuntimeError("flash_attn is not actually installed. This is just a stub.")

# Provide common attributes that might be accessed
__all__ = ['flash_attn_func', 'FlashAttentionFunction']




