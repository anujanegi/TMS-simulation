import os
import sys
import json
import numpy as np
import pandas as pd
import simnibs
import config
from itertools import combinations


def efield_group_stats_over_fsavg(subjects, field_name="magnE", show=False):
    """
    Will go through simulated efields and calculate
    the average and standard deviation of the given field 
    of the electric field in FsAverage space for each subject type

    Args:
        subjects (_type_): dictionary of {subject type: list of subject IDS}
        field_name (str, optional): name of the field to calculate the average for. Defaults to "magnE".
    """
    # calculate avg and std of efield for each subject type over fsavg
    averaged_efields = {}
    std_efields = {}
    for subject_type in subjects:
        fields = []
        for subject in subjects[subject_type]:
            results_fsavg = simnibs.read_msh(
                config.get_efield_fsavg_overlay_mesh_path(subject, subject_type)
            )
            fields.append(results_fsavg.field[field_name].value)
        fields = np.vstack(fields)
        averaged_efields[subject_type] = np.mean(fields, axis=0)
        std_efields[subject_type] = np.std(fields, axis=0)

    # plot and save
    for subject_type in subjects:
        # plot the average
        results_fsavg.nodedata = []  # cleanup fields
        results_fsavg.add_node_field(
            averaged_efields[subject_type], "avg"
        )  # add avg field

        if show:
            # show surface with the fields average
            results_fsavg.view(visible_fields="avg").show()
        results_fsavg.write(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_avg_{subject_type}_fsavg.msh",
            )
        )
        print(
            f"Saved {field_name}_avg_{subject_type}_fsavg.msh in {config.get_analysis_data_path()}"
        )

        # plot the std
        results_fsavg.nodedata = []  # cleanup fields
        results_fsavg.add_node_field(std_efields[subject_type], "std")  # add std field
        if show:
            # show surface with the fields std
            results_fsavg.view(visible_fields="std").show()
        results_fsavg.write(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_std_{subject_type}_fsavg.msh",
            )
        )
        print(
            f"Saved {field_name}_std_{subject_type}_fsavg.msh in {config.get_analysis_data_path()}"
        )


def efield_group_stats_over_atlas(subjects, field_name="magnE", atlas_name="HCP_MMP1"):
    """
    Will go through simulated efields and calculate
    the average and standard deviation of the given field
    of the electric field in the given atlas space for each subject type
    
    Args:
        subjects (_type_): dictionary of {subject type: list of subject IDS}
        field_name (str, optional): name of the field to calculate the average for. Defaults to "magnE".
        atlas_name (str, optional): name of the atlas to calculate the average over. Defaults to "HCP_MMP1".
    """
    # calculate avg and std of efield for each subject type over atlas
    averaged_efields = {}
    std_efields = {}
    for subject_type in subjects:
        fields = []
        for subject in subjects[subject_type]:
            with open(
                config.get_efield_atlas_avg_path(subject, subject_type), "r"
            ) as f:
                results_atlas = json.load(f)
            fields.append(results_atlas)  # collecting json dicts as a list

        fields_df = pd.DataFrame(fields)
        averaged_efields[subject_type] = dict(fields_df.mean())
        std_efields[subject_type] = dict(fields_df.std())

    # save the results as json
    for subject_type in subjects:
        with open(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_avg_{subject_type}_{atlas_name}.json",
            ),
            "w",
        ) as f:
            json.dump(averaged_efields[subject_type], f)
        print(
            f"Saved {field_name}_avg_{subject_type}_{atlas_name}.json in {config.get_analysis_data_path()}"
        )
        with open(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_std_{subject_type}_{atlas_name}.json",
            ),
            "w",
        ) as f:
            json.dump(std_efields[subject_type], f)
        print(
            f"Saved {field_name}_std_{subject_type}_{atlas_name}.json in {config.get_analysis_data_path()}"
        )


def efield_group_difference_over_fsavg(subjects, field_name="magnE", show=False):
    """
    Will go through group averaged efields and calculate
    the differences in the given field 
    of the electric field in FsAverage space

    Args:
        subjects (_type_): dictionary of {subject type: list of subject IDS}
        field_name (str, optional): name of the field to calculate the difference for. Defaults to "magnE".
    """
    # load average efield for each subject type over fsavg
    averaged_efields = {}
    for subject_type in subjects:
        mesh = simnibs.read_msh(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_avg_{subject_type}_fsavg.msh",
            )
        )
        averaged_efields[subject_type] = mesh.nodedata[0].value

    subject_type_combinations = list(combinations(subjects.keys(), 2))
    for combination in subject_type_combinations:
        # calculate the difference between the two
        fields = np.vstack(
            [averaged_efields[combination[0]], averaged_efields[combination[1]]]
        )
        diff_field = np.diff(fields, axis=0)
        # plot the difference
        mesh.nodedata = []  # cleanup fields
        mesh.add_node_field(
            diff_field[0], f"{field_name}_difference_{combination[1]}_{combination[0]}"
        )  # add difference field

        # show surface with the fields difference
        if show:
            mesh.view(
                visible_fields=f"{field_name}_difference_{combination[1]}_{combination[0]}"
            ).show()

        mesh.write(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_difference_{combination[1]}_{combination[0]}_fsavg.msh",
            )
        )
        print(f"Saved difference in fsavg space in {config.get_analysis_data_path()}")


def efield_group_diff_over_atlas(subjects, field_name="magnE", atlas_name="HCP_MMP1"):
    """
    Will go through group averaged efields and calculate
    the differences in the given field
    of the electric field in the given atlas space
    
    Args:
        subjects (_type_): dictionary of {subject type: list of subject IDS}
        field_name (str, optional): name of the field to calculate the difference for. Defaults to "magnE".
        atlas_name (str, optional): name of the atlas to calculate the difference over. Defaults to "HCP_MMP1".
    """
    # load average efield for each subject type over atlas
    averaged_efields = {}
    for subject_type in subjects:
        with open(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_avg_{subject_type}_{atlas_name}.json",
            ),
            "r",
        ) as f:
            averaged_efields[subject_type] = json.load(f)

    subject_type_combinations = list(combinations(subjects.keys(), 2))
    for combination in subject_type_combinations:
        diff_field = {
            key: averaged_efields[combination[1]][key]
            - averaged_efields[combination[0]].get(key, 0)
            for key in averaged_efields[combination[1]]
        }

        # save the results as json
        with open(
            os.path.join(
                config.get_analysis_data_path(),
                f"{field_name}_difference_{combination[1]}_{combination[0]}_{atlas_name}.json",
            ),
            "w",
        ) as f:
            json.dump(diff_field, f)
        print(
            f"Saved difference in {atlas_name} space in {config.get_analysis_data_path()}"
        )


if __name__ == "__main__":

    list_of_args = sys.argv[1:]

    if "efield_group_stats_over_fsavg" in list_of_args:
        efield_group_stats_over_fsavg(config.subjects)
    elif "efield_group_stats_over_atlas" in list_of_args:
        efield_group_stats_over_atlas(config.subjects)
    elif "efield_group_diff_over_fsavg" in list_of_args:
        efield_group_difference_over_fsavg(config.subjects)
    elif "efield_group_diff_over_atlas" in list_of_args:
        efield_group_diff_over_atlas(config.subjects)
    else:
        print("Supported commands:")
        print(
            "efield_group_stats_over_fsavg: calculates the average and standard deviaion of the efield for each subject type over fsavg"
        )
        print(
            'efield_group_stats_over_atlas: calculates the average and standard deviaion of the efield for each subject type over the given atlas (default: "HCP_MMP1")'
        )
        print(
            "efield_group_diff_over_fsavg: calculate the difference between the efields of the different subject types in fsavg space"
        )
        print(
            "efield_group_diff_over_atlas: calculate the difference between the efields of the different subject types in the given atlas space"
        )

# TODO: statistical significance btw groups
