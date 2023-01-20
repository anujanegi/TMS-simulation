import numpy as np


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
