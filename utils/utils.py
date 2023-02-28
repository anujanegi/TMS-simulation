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


def get_max_efield_on_atlas():
    max_values = []
    for type in config.subjects:
        for subject in config.subjects[type]:
            with open(config.get_efield_atlas_avg_path(subject, type)) as f:
                efield = json.load(f)
                max_values.append(max(efield.values()))

    return max(max_values)


def get_max_efield_in_type(type):
    max_values = []
    for subject in config.subjects[type]:
        from simnibs import read_msh

        mesh = read_msh(config.get_efield_head_mesh_path(subject, type))
        max_values.append(max(mesh.nodedata[0].value))

    return max(max_values)
