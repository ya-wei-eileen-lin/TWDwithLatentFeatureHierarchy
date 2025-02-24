import numpy as np
from sklearn.metrics import pairwise_distances

def hyperbolic_distance(x, y):
    """Calculate the hyperbolic distance between two points in hyperbolic space."""
    return 2 * np.arcsinh(np.linalg.norm(x - y) / (2 * np.sqrt(x[-1] * y[-1])))

def hyperbolic_distance_axis(x, y):
    """Calculate the distance along the same axis in hyperbolic space."""
    return np.abs(np.log(x[-1]) - np.log(y[-1]))

def midpoint_same_level(x, y, sanity_check=False):
    """Find the midpoint of two points at the same level in hyperbolic space."""
    if x[-1] != y[-1]:
        raise Warning("Points are not from the same level")

    t = x[-1]
    z = (x[:-1] + y[:-1]) / 2
    z_n = np.sqrt(np.sum(((x[:-1] - y[:-1]) / 2)**2) + t**2)
    z = np.concatenate((z, [z_n]))

    if sanity_check:
        print('Hyperbolic distance (x, z):', hyperbolic_distance(x, z))
        print('Hyperbolic distance (y, z):', hyperbolic_distance(y, z))

    return z, z_n

def midpoint_same_axis(x, y, sanity_check=False):
    """Find the midpoint of two points along the same axis in hyperbolic space."""
    n1_axis = x[:-1]
    level_x, level_y = x[-1], y[-1]
    z_n = np.sqrt(level_x * level_y)
    z = np.concatenate((n1_axis, [z_n]))

    if sanity_check:
        print('Hyperbolic distance (x, z):', hyperbolic_distance(x, z))
        print('Hyperbolic distance (y, z):', hyperbolic_distance(y, z))

    return z, z_n

def midpoint_set_same_axis(embedding):
    """Calculate the midpoints for a set of points along the same axis in hyperbolic space."""
    N, k = embedding.shape
    levels = embedding[:, -1]
    pair_midpoints = np.sqrt(np.outer(levels, levels))
    np.fill_diagonal(pair_midpoints, 0)
    return pair_midpoints

def midpoint_set_same_level(embedding):
    """Calculate the midpoints for a set of points at the same level in hyperbolic space."""
    N, k = embedding.shape
    emd = embedding[:, :-1]
    t = embedding[0, -1]
    pair_distances = pairwise_distances(emd)
    pair_radii = np.sqrt(pair_distances**2 / 4 + t**2)
    return pair_radii


def midpoint_same_axis_geometric_mean(points):
    """
    Calculate the midpoint for multiple points on the same axis in hyperbolic space.
    The points are of the form [0, 0, ..., 0, a_i].

    Parameters:
    points (numpy.ndarray): An array of points in the hyperbolic space.

    Returns:
    numpy.ndarray: The midpoint on the same axis.
    """
    if not all(np.all(points[:, :-1] == 0, axis=1)):
        raise ValueError("All points must be on the same axis (only the last coordinate can differ).")

    # Extracting the last element (hyperbolic radii) of each point
    radii = points[:, -1]

    # Calculating the geometric mean of the radii
    geometric_mean_radius = np.power(np.prod(radii), 1/len(radii))

    # Constructing the midpoint (with all but the last elements being zero)
    midpoint = np.zeros_like(points[0])
    midpoint[-1] = geometric_mean_radius

    return midpoint