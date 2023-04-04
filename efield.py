"""This script simulated the electric field for a subjects head model
"""

from simnibs import sim_struct, run_simnibs, read_msh
import simnibs
import nilearn.image as img
import nilearn.plotting as niplot
import os
import numpy as np
import config
import sys


def simulate_efield(subject, type):

    # Initalize a session
    s = sim_struct.SESSION()
    # Name of head mesh
    s.subpath = config.get_m2m_path(subject, type)
    # Output folder
    s.pathfem = os.path.join(config.get_subject_path(subject, type), "TMS_efield")
    s.fields = "e"
    s.map_to_surf = True
    s.map_to_fsavg = True
    s.map_to_MNI = True
    s.open_in_gmsh = False

    if not os.path.exists(s.pathfem):
        os.makedirs(s.pathfem)
    # delete all .mat files in the folder
    for f in os.listdir(s.pathfem):
        if f.endswith(".mat"):
            os.remove(os.path.join(s.pathfem, f))

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
    # Point the coil handle posteriorly, just add 10 mm to the original M1 y coordinate
    pos.pos_ydir = simnibs.mni2subject_coords(
        [-37, -21 - 10, 58], config.get_m2m_path(subject, type)
    )

    run_simnibs(s)


def average_efield_over_atlas(
    subject, type, atlas_name="HCP_MMP1", save=True, field_name="magnE"
):
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


def calculate_leadfield(subject, type):
    lf = sim_struct.TDCSLEADFIELD()
    # subject folder
    lf.subpath = config.get_m2m_path(subject, type)
    # output directory
    lf.pathfem = os.path.join(config.get_subject_path(subject, type), "leadfield")
    # electrode positions
    lf.eeg_cap = os.path.join(
        config.get_m2m_path(subject, type), "eeg_positions", "easycap_BC_TMS64_X21.csv"
    )
    run_simnibs(lf)


if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    if "simulate_efield" in list_of_args:
        # simulates and saves the electric field for all subjects given in config.subjects
        # saves eflield average over atlas in json file
        for type in config.subjects:
            for subject in config.subjects[type]:
                simulate_efield(subject, type)
                average_efield_over_atlas(subject, type)

    elif "calculate_leadfield" in list_of_args:
        # calculates the leadfield for all subjects given in config.subjects
        for type in config.subjects:
            for subject in config.subjects[type]:
                calculate_leadfield(subject, type)

    else:
        print("Supported arguments:")
        print(
            "simulate_efield: Simulates and saves the electric field for all subjects given in config.subjects; also averages eflield over atlas and saves as a json file"
        )
        print(
            "calculate_leadfield: Calculates the leadfield for all subjects given in config.subjects"
        )
