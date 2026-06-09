import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.array(x)
    h, w = x.shape[-2], x.shape[-1]
    if len(x.shape) == 3:
        return np.sum(x, axis=(1, 2)) / (h*w)
    elif len(x.shape) == 4:
        return np.sum(x, axis=(2, 3)) / (h*w)
    else:
        raise ValueError(f"Expected numpy array size, got {x.shape}")
    
        