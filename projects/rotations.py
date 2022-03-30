"""Perform multiple rotation matrices."""

import numpy as np
from itertools import combinations
import random
import pdb


def basic_rot_matrix(angle, axes, size):
    """
    Produce rotation matrix of angle theta.

    Parameters
    ----------
    angle: float
       Angle of rotation in radians.
    axes: tuple
       Axis of rotation from and towards. It should be smaller than size.
    size: int
       Size of transformation at hand.

    Returns
    -------
    array: ndarray
       Matrix that performs a rotation by angle from axisa to axisb.
    """
    mat = np.identity(size)
    axisa, axisb = axes[0], axes[1]
    mat[axisa, axisa] = np.cos(angle)
    mat[axisb, axisb] = np.cos(angle)
    mat[axisa, axisb] = -np.sin(angle)
    mat[axisb, axisa] = np.sin(angle)
    return mat


def full_rotation_matrix(angles, axes, size):
    """
    Calculate full rotation matrix.

    Parameters
    ----------
    angles: list
       Each element contains rotation angle in degrees.
    axes: list
       Each element is a tuple that contains axis from and towards.
    size: int
       Size of the rotation matrix

    Returns
    -------
    array: ndarray
       Full rotation matrix.
    """
    mat = np.identity(size)
    for i, ang in enumerate(angles):
        ang = ang*np.pi/180
        mat = np.matmul(mat, basic_rot_matrix(ang, axes[i], size))
    return mat


def rotation_scaling(dim, prev_angle):
    """
    Rotate along main axes and performs scaling.

    Parameters
    ----------
    dim: int
       Dimension of the matrix. This code generates a square matrix.
    pre_angle: array
       Contains all angles for previous rotations.

    Returns
    -------
    array: ndarray
       Weight matrix with rotations and scaling.
    """
    axes = list(np.arange(dim))
    ax = list(combinations(axes, 2))
    size = np.random.randint(len(ax)-1, len(ax)+1) #tuned for superconductor dataset
#    size = np.random.randint(1, len(ax)+1)
    trans = random.sample(list(np.arange(len(ax))), size)
    trans = np.array(trans).astype(int)
    new_angle = prev_angle
    c_ang = new_angle[trans]
    angles = -np.random.randint(c_ang-5, c_ang+5)
    axes = [ax[i] for i in trans]
    rot_mat = full_rotation_matrix(angles, axes, dim)
    new_angle[trans] = new_angle[trans] + angles
    sigmas = np.random.uniform(.9, 1.1, size=dim)
    sigmas = np.diag(sigmas)
    rot_mat = np.matmul(sigmas, rot_mat)
    return rot_mat, new_angle
        

# dim = 3
# axes = list(np.arange(dim))
# ax = list(combinations(axes, 2))

# pre = np.zeros(len(ax))

# rot_mat, new_angle = rotation_scaling(dim, pre)

# test = np.matmul(rot_mat, np.array([[1],[0],[0]]))
# test1 = np.matmul(rot_mat, np.array([[0],[1],[0]]))
# test2 = np.matmul(rot_mat, np.array([[0],[0],[1]]))
