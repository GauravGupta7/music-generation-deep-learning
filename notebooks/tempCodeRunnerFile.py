import numpy as np
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict  # Fix for compatibility issues
