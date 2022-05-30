"""Simple TMS coil modelling.
"""
from tkinter.tix import Tree
import numpy as np


def _coil_positions(coil_pts, list_positions):
    # Term explanation: 'Occipital' = towards back, 'Cranial' = towards vertex
    pos_list = ['Occipital', 'Occipital-Right', 'Right', 'Frontal-Right', 'Frontal', 
                'Frontal-Left', 'Left', 'Occipital-Left', 'Occipital-Cranial-Central', 'Cranial', 'Frontal-Cranial-Central', 
                'Frontal-Cranial-Right', 'Cranial-Right', 'Occipital-Cranial-Right', 
                'Occipital-Cranial-Left', 'Cranial-Left', 'Frontal-Cranial-Left']
    view_angles = []

    # Rotations and translations of the coil
    delta_x = [-110, -95, 0, 95, 105, 95, 0, -95, -75, 0, 65, 45, 0, -50, -50, 0, 45] # mm
    delta_y = [0, -50, -90, -55, 0, 55, 90, 50, 0, 0, 0, -45, -60, -50, 50, 60, 45]
    delta_z = [0, 0, 0, 0, 0, 0, 0, 0, 60, 85, 60, 50, 60, 60, 60, 60, 50]
    rot_x = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/4, 0, np.pi/4, 
            -np.pi/4, -np.pi/4, -np.pi/4, np.pi/4, np.pi/4, np.pi/4] # radians
    rot_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rot_z = [np.pi / 2, np.pi / 4, 0, -np.pi / 4, np.pi/2, np.pi / 4, 0, -np.pi / 4, np.pi/2, np.pi/2, np.pi/2, 
            -np.pi / 4, 0, np.pi / 4, -np.pi / 4, 0, np.pi / 4]
  

    # Initialize coil position lists
    coil_rot = []

    # For each coil position
    for pos in range(len(delta_x)):
        # Define series of transformation matrices and multiply
        # Translation vector
        delta_vec = np.zeros(3)
        delta_vec[0] = delta_x[pos]
        delta_vec[1] = delta_y[pos]
        delta_vec[2] = delta_z[pos]
        # Rotation about x-axis
        rot_mat_x = np.eye(3)
        rot_mat_x[1, 1] = rot_mat_x[2, 2] = np.cos(rot_x[pos])
        rot_mat_x[1, 2] = -np.sin(rot_x[pos])
        rot_mat_x[2, 1] = np.sin(rot_x[pos])
        # Rotation about y-axis
        rot_mat_y = np.eye(3)
        rot_mat_y[0, 0] = rot_mat_y[2, 2] = np.cos(rot_y[pos])
        rot_mat_y[0, 2] = -np.sin(rot_y[pos])
        rot_mat_y[2, 0] = np.sin(rot_y[pos])
        # Rotation about z-axis
        rot_mat_z = np.eye(3)
        rot_mat_z[0, 0] = rot_mat_z[1, 1] = np.cos(rot_z[pos])
        rot_mat_z[0, 1] = -np.sin(rot_z[pos])
        rot_mat_z[1, 0] = np.sin(rot_z[pos])
        # Full Transform
        rot_mat = rot_mat_x.dot(rot_mat_y).dot(rot_mat_z)

        # Apply to coils
        coil_rot.append(coil_pts.dot(rot_mat) + delta_vec)
        
    if list_positions:
        # Print list of coil positions and their index number
        list_print = list(range(len(pos_list)))
        print("\033[1mIndex numbers of coil positions\033[0m")
        for pos in list_print:
            print(pos,'\t',pos_list[pos])

    return coil_rot

def get_tms_coil(dx=1, r_coil=50, type="fig8", list_positions=True):

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
    
    # transform coil for different positions
    coil = _coil_positions(coil_pts, list_positions)

    return coil, xs, ys
