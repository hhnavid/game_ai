import numpy as np

def rotate_point_around_axis(p, axis_, theta):
    """
    rotate a 3d point around a given axis with angle theta
    Args:
        p (np array): 3d point to be rotated
        axis_ (np array): 3d rotation axis
        theta (float, rad): rotation angle

    Returns:
        np array: rotated point
    """    
    # Normalize axis
    axis_ = axis_ / np.linalg.norm(axis_)
    
    # Rodrigues' rotation formula
    p_rot = (
        p * np.cos(theta)
        + np.cross(axis_, p) * np.sin(theta)
        + axis_ * np.dot(axis_, p) * (1 - np.cos(theta))
    )    
    return p_rot