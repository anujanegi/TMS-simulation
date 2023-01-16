"""This script simulated the electric field for a subjects head model
"""

from simnibs import sim_struct, run_simnibs, read_msh, SIMNIBSDIR
import simnibs
import nilearn.image as img
import nilearn.plotting as niplot
import os
import numpy as np
from numpy import degrees, arcsin, arctan2, deg2rad, cos, sin
import config


def simulate_efield(subject, type):

    # Initalize a session
    s = sim_struct.SESSION()
    # Name of head mesh
    s.subpath = config.get_m2m_path(subject, type)
    # Output folder
    s.pathfem = os.path.join(config.get_subject_path(subject, type), "TMS_efield")
    s.fields = "e"
    s.map_to_surf = True
    # s.map_to_fsavg = True
    s.map_to_MNI = True
    s.open_in_gmsh = False

    if not os.path.exists(s.pathfem):
        os.makedirs(s.pathfem)

    # coil model
    # Initialize a list of TMS simulations
    tmslist = s.add_tmslist()
    # Select coil
    tmslist.fnamecoil = os.path.join("Drakaki_BrainStim_2022", "MagStim_D70.ccd")
    pos = tmslist.add_position()

    # Define M1 from MNI coordinates (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2034289/)
    # Select coil centre
    pos.centre = simnibs.mni2subject_coords(
        [-37, -21, 58], config.get_m2m_path(subject, type)
    )
    # Select coil direction
    # Point the coil handle posteriorly, we just add 10 mm to the original M1 y coordinate
    pos.pos_ydir = simnibs.mni2subject_coords(
        [-37, -21 - 10, 58], config.get_m2m_path(subject, type)
    )

    run_simnibs(s)


def average_efield_over_atlas(subject, type, atlas_name="HCP_MMP1", save=True):
    """Average the efield over the given atlas"""
    efield_head_mesh = read_msh(config.get_efield_head_mesh_path(subject, type))
    atlas = simnibs.subject_atlas(atlas_name, config.get_m2m_path(subject, type))

    EF_result = {}
    for areas, values in atlas.items():
        # define ROI
        if "unknown" in areas or "???" in areas:
            continue
        roi = atlas[areas]

        # calculate nodes areas in ROI for averaging
        node_areas = efield_head_mesh.nodes_areas()
        if not node_areas[roi].any():
            print(areas, node_areas[roi])
            EF_result[areas] = 0
            continue

        # Calculate mean electric field in each region
        field_name = "magnE"
        mean_normE = np.average(
            efield_head_mesh.field[field_name][roi], weights=node_areas[roi]
        )
        EF_result[areas] = mean_normE

    if save:
        import json

        with open(
            os.path.join(
                config.get_TMS_efield_path(subject, type),
                f"{subject}_efield_over_{atlas_name}_{field_name}.json",
            ),
            "w",
        ) as fp:
            json.dump(EF_result, fp)
    else:
        return EF_result


if __name__ == "__main__":

    for type in config.subjects:
        for subject in config.subjects[type]:
            simulate_efield(subject, type)
            average_efield_over_atlas(subject, type)
