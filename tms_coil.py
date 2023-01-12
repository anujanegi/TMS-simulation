"""Simple TMS coil modelling.
"""
from tkinter.tix import Tree
import numpy as np


class TMS_coil:
    def __init__(self, dx=1, r_coil=50, type="fig8") -> None:
        """
        Args:
        dx (in mm, optional): precision. Defaults to 1.
        r_coil (in mm, optional): radius of coil. Defaults to 50 (N. Lang et al. 2006 - in Endnote).
        type (str, optional): Coil type. Defaults to "fig8".
        """
        self.dx = 1
        self.r_coil = 50
        self.type = type
        self.coil_rot = None
        self.coil_pts = None

    def init_coil_positions(self, list_positions):
        # Term explanation: 'Occipital' = towards back, 'Cranial' = towards vertex
        self.pos_list = [
            "Occipital",
            "Occipital-Right",
            "Right",
            "Frontal-Right",
            "Frontal",
            "Frontal-Left",
            "Left",
            "Occipital-Left",
            "Occipital-Cranial-Central",
            "Cranial",
            "Frontal-Cranial-Central",
            "Frontal-Cranial-Right",
            "Cranial-Right",
            "Occipital-Cranial-Right",
            "Occipital-Cranial-Left",
            "Cranial-Left",
            "Frontal-Cranial-Left",
        ]
        view_angles = []

        # Rotations and translations of the coil
        delta_x = [
            -110,
            -95,
            0,
            95,
            105,
            95,
            0,
            -95,
            -75,
            0,
            65,
            45,
            0,
            -50,
            -50,
            0,
            45,
        ]  # mm
        delta_y = [0, -50, -90, -55, 0, 55, 90, 50, 0, 0, 0, -45, -60, -50, 50, 60, 45]
        delta_z = [0, 0, 0, 0, 0, 0, 0, 0, 60, 85, 60, 50, 60, 60, 60, 60, 50]
        rot_x = [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            -np.pi / 4,
            -np.pi / 4,
            -np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
        ]  # radians
        rot_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        rot_z = [
            np.pi / 2,
            np.pi / 4,
            0,
            -np.pi / 4,
            np.pi / 2,
            np.pi / 4,
            0,
            -np.pi / 4,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            -np.pi / 4,
            0,
            np.pi / 4,
        ]

        # Initialize coil position lists
        self.coil_rot = []

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
            self.coil_rot.append(self.coil_pts.dot(rot_mat) + delta_vec)

        if list_positions:
            # Print list of coil positions and their index number
            list_print = list(range(len(self.pos_list)))
            print("\033[1mIndex numbers of coil positions\033[0m")
            for pos in list_print:
                print(pos, "\t", self.pos_list[pos])

    def get_tms_coil(self, list_positions=True):

        """Defines the coil positions for a given coil type."""
        data_pts = np.arange(0, 2 * np.pi * self.r_coil, self.dx)

        if self.type == "fig8":
            xs = np.hstack(
                (
                    self.r_coil * (np.cos(data_pts - np.pi) + 1),
                    self.r_coil * (np.cos(data_pts) - 1),
                )
            )
            ys = np.hstack(
                (self.r_coil * np.sin(data_pts), self.r_coil * np.sin(data_pts))
            )
            zs = np.zeros(xs.size)
            self.coil_pts = np.stack((xs, ys, zs)).T

        elif self.type == "circular":
            xs = self.r_coil * np.cos(data_pts)
            ys = self.r_coil * np.sin(data_pts)
            zs = np.zeros(xs.size)
            self.coil_pts = np.stack((xs, ys, zs)).T
        else:
            raise ValueError("Coil type option doesn't exist")

        # transform coil for different positions
        self.init_coil_positions(list_positions)

        return self.coil_rot, self.coil_pts, xs, ys

    def get_electric_field_strength(self, nodes, idx):
        """Calculates the strength of the electric field for a particular point

        The strength is calculated using a simple inverse square assumption of electric
        field strength. Since the points used to define the coil are equidistant, the
        total field strength at a particular point relative to this is defined as the
        normalized sum of distance from each point defining the coil. This assumes the TMS
        coil produces a homogeneous electric field (ref validity).

        Parameters
            nodes = coordinates of points being acted on by the electric field
                    dims = (n_nodes, 3), where second dimension is (x, y, z)
        """
        self.idx = idx
        coil_pts = self.coil_rot[self.idx]

        # Define the number of nodes being calculated for
        try:
            n_nodes = nodes.shape[0]
        except AttributeError:
            n_nodes = 1
        # Define number of coil points (used for normalization)
        if self.type == "fig8":
            n_pts = coil_pts.shape[0] / 4
        else:
            print(self.type)
            n_pts = coil_pts.shape[0]
        # Initialize strength of stimulus
        stim_strength = np.zeros(n_nodes)

        for i in range(n_nodes):
            # Calculate average distance of every point of coil to region point
            dist_sum = 0
            for j in range(len(coil_pts)):
                dist_sum += np.sqrt(
                    np.sum((nodes[i, :] - coil_pts[j]) ** 2, axis=0)
                )  # calculates distance between two 3D coordinates
            dist = dist_sum / coil_pts.shape[0]

            # Define cutoff value for maximum distance with impact; in MNI space coordinate
            if dist < 100:
                stim_strength[i] += (
                    np.mean(1 / np.linalg.norm(nodes[i, :] - coil_pts, axis=1) ** 2)
                    / n_pts
                )

        self.stim_strength = stim_strength
        return stim_strength

    def get_stimulus_distribution(self, field_strength, region_labels, idx=None):
        # Create a list with number, name and stimulus weighting of each region, descending by stimulus weighting
        reg_lab = region_labels
        reg_list = []
        for i in range(len(reg_lab)):
            reg_list.append(i)

        zip_list = list(zip(field_strength.tolist(), reg_list, reg_lab.tolist()))
        zip_list.sort(reverse=True)

        if idx:
            print(
                "\033[1mStimulus distribution at chosen coil position: %s"
                % self.pos_list[idx]
                + "\033[0m\n"
            )
        print(
            "\033[1m{0:16}{1:16}{2:16}\033[0m".format(
                "Region index", "Region label", "Stimulus weighting"
            )
        )
        for i in range(len(zip_list)):  # Show list with all regions
            # for i in range(10):    # Show list with top 10 regions
            print(
                str(
                    "{0:<16}{1:16}{2:16}".format(
                        zip_list[i][1], zip_list[i][2], zip_list[i][0]
                    )
                )
            )
