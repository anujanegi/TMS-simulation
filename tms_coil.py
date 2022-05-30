"""Simple TMS coil modelling.
"""
import numpy as np


def get_tms_coil(dx=1, r_coil=50, type="fig8"):

    """Defines the coil positions for a given coil type.

    Args:
        dx (in mm, optional): precision. Defaults to 1.
        r_coil (in mm, optional): radius of coil. Defaults to 50 (N. Lang et al. 2006 - in Endnote).
        type (str, optional): Coil type. Defaults to "fig8".
    """
    dx = 1
    r_coil = 50
    data_pts = np.arange(0, 2 * np.pi * r_coil, dx)

    if type == "fig8":
        xs = np.hstack(
            (r_coil * (np.cos(data_pts - np.pi) + 1), r_coil * (np.cos(data_pts) - 1))
        )
        ys = np.hstack((r_coil * np.sin(data_pts), r_coil * np.sin(data_pts)))
        zs = np.zeros(xs.size)
        coil_pts = np.stack((xs, ys, zs)).T

    elif type == "circular":
        xs = r_coil * np.cos(data_pts)
        ys = r_coil * np.sin(data_pts)
        zs = np.zeros(xs.size)
        coil_pts = np.stack((xs, ys, zs)).T
    else:
        raise ValueError("Coil type option doesn't exist")

    return coil_pts, xs, ys
