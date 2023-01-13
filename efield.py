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


def simulatie_efield(subject, type):

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

    # for right M1 using electrode as reference
    # Select coil centre
    pos.centre = "FC6"  # right M1
    # Select coil direction
    pos.pos_ydir = "AF8"

    run_simnibs(s)


def average_efield_over_atlas(subject, type, atlas_name="HCP_MMP1", save=True):
    """Average the efield over the given atlas
    """
    efield_head_mesh = read_msh(config.get_efield_head_mesh_path(subject, type))
    atlas = simnibs.subject_atlas(atlas_name, config.get_m2m_path(subject, type))
    # Crop the mesh so we only have gray matter volume elements (tag 2 in the mesh)
    gray_matter = efield_head_mesh.crop_mesh(2)

    EF_result = {}
    for areas, values in atlas.items():
        # define ROI
        if "unknown" in areas or "???" in areas:
            continue
        roi = atlas[areas]

        # calculate nodes areas in ROI for averaging
        node_areas = gray_matter.nodes_areas()
        if not node_areas[roi].any():
            print(areas, node_areas[roi])
            EF_result[areas] = 0
            continue

        # Calculate mean electric field in each region
        field_name = "magnE"
        mean_normE = np.average(
            gray_matter.field[field_name][roi], weights=node_areas[roi]
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
            # simulatie_efield(subject, type)
            average_efield_over_atlas(subject, type)
