"""It normalizes arrays."""

import numpy as np


def normalization_0_1(x, axis):
    """
    Normalize between 0 and 1 along specified axis. It works with any dimension.

    Parameters
    ----------
    x : array
        Array can have any size, and normalization occurs in specified axis.
    
    Returns
    -------
    array: ndarray
       The normalized array between 0 and 1.
    """
    minv = np.min(x, axis=axis)
    num = x-np.expand_dims(minv, axis=axis)
    den = np.expand_dims(np.max(x, axis=axis)-minv, axis=axis)
    return num/den


def normalization_std(x, axis):
    """
    Normalize to 0 mean and 1 std along the last axis.

    Parameters
    ----------
    x : array
        Array can have any size, and normalization occurs in the last
        axis.
    
    Returns
    -------
    array: ndarray
       The normalized array with 0 mean and 1 std
    """
    x = x-np.expand_dims(np.mean(x, axis=axis), axis=axis)
    std = np.expand_dims(np.std(x, axis=axis), axis=axis)
    return x/std
