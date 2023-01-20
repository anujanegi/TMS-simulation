import os
import sys
import numpy as np
import simnibs
import config
from itertools import combinations


def efield_group_difference(subjects, field_name="magnE"):
    """
    Will go through simulated efields and calculate
    the differences in the given field 
    of the electric field in FsAverage space

    Args:
        subjects (_type_): dictionary of {subject type: list of subject IDS}
        field_name (str, optional): name of the field to calculate the difference for. Defaults to "magnE".
    """
    # calculate avg efield for each subject type
    averaged_efields = {}
    for subject_type in subjects:
        fields = []
        for subject in subjects[subject_type]:
            results_fsavg = simnibs.read_msh(
                config.get_efield_fsavg_overlay_mesh_path(subject, subject_type)
            )
            fields.append(results_fsavg.field[field_name].value)
        fields = np.vstack(fields)
        avg_field = np.mean(fields, axis=0)
        averaged_efields[subject_type] = avg_field

    subject_type_combinations = list(combinations(subjects.keys(), 2))
    for combination in subject_type_combinations:
        # calculate the difference between the two
        fields = np.vstack(
            [averaged_efields[combination[0]], averaged_efields[combination[1]]]
        )
        diff_field = np.diff(fields, axis=0)
        # plot the difference
        results_fsavg.nodedata = []  # cleanup fields
        results_fsavg.add_node_field(
            diff_field[0], f"{field_name}_difference_{combination[1]}_{combination[0]}"
        )  # add difference field

        # show surface with the fields difference
        results_fsavg.view(
            visible_fields=f"{field_name}_difference_{combination[1]}_{combination[0]}"
        ).show()

        results_fsavg.write(
            os.path.join(
                config.get_analysis_path(),
                f"{field_name}_difference_{combination[1]}_{combination[0]}.msh",
            )
        )


if __name__ == "__main__":

    list_of_args = sys.argv[1:]

    if "efield_group_diff" in list_of_args:
        efield_group_difference(config.subjects)
    else:
        print("Supported commands:")
        print(
            "efield_group_diff: calculate the difference between the efields of the different subject types"
        )
