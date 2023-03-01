import numpy as np
import config
import json


def baseline_correction(data, time_interval=[1000, 1450]):
    if data["eeg"]:
        interval_baseline = np.where(
            np.logical_and(
                data["eeg"]["time"] >= time_interval[0],
                data["eeg"]["time"] < time_interval[1],
            )
        )
        baseline = []
        for e in range(data["eeg"]["data"].shape[2]):
            avg = np.mean(data["eeg"]["data"][:, 0, e, 0][interval_baseline])
            baseline.append(avg)

        for i, base in enumerate(baseline):
            data["eeg"]["data"][:, 0, i, 0] = data["eeg"]["data"][:, 0, i, 0] - base

        return data
    else:
        raise ValueError("No EEG monitor data found!")


def get_lim_efield_on_atlas():
    max_values = []
    for type in config.subjects:
        for subject in config.subjects[type]:
            with open(config.get_efield_atlas_avg_path(subject, type)) as f:
                efield = json.load(f)
                max_values.append(max(efield.values()))

    return [0, max(max_values)]


def get_lim_efield_in_type(type):
    max_values = []
    for subject in config.subjects[type]:
        from simnibs import read_msh

        mesh = read_msh(config.get_efield_head_mesh_path(subject, type))
        max_values.append(max(mesh.nodedata[0].value))

    return [0, max(max_values)]


def get_lim_stat_efield_in_type(type, stat):
    # gets the max value of the efield in the given type and stat over fsavg
    # NOTE: between fsavg and atlas, fsavg is the one with the highest values, hence choosing fsavg
    max_values = []
    for subject in config.subjects[type]:
        from simnibs import read_msh

        mesh = read_msh(config.get_efield_stats_path(type, stat=stat, over="fsavg"))
        max_values.append(max(mesh.nodedata[0].value))

    return [0, max(max_values)]


def get_lim_efield_difference(over="fsavg"):
    # gets the max value of the efield difference over fsavg
    # NOTE: between fsavg and atlas, fsavg is the one with the highest values, hence choosing fsavg
    max_values = []
    min_values = []
    import os

    if over == "fsavg":

        mesh_list = [
            fname
            for fname in os.listdir(config.get_analysis_data_path())
            if fname.endswith(".msh") and "difference" in fname and over in fname
        ]
        for mesh in mesh_list:
            from simnibs import read_msh

            mesh = read_msh(os.path.join(config.get_analysis_data_path(), mesh))
            max_values.append(max(mesh.nodedata[0].value))
            min_values.append(min(mesh.nodedata[0].value))
    else:
        json_list = [
            fname
            for fname in os.listdir(config.get_analysis_data_path())
            if fname.endswith(".json") and "difference" in fname and over in fname
        ]
        for json_file in json_list:
            with open(os.path.join(config.get_analysis_data_path(), json_file)) as f:
                efield = json.load(f)
                max_values.append(max(efield.values()))
                min_values.append(min(efield.values()))

    return [min(min_values), max(max_values)]
