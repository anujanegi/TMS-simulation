"""Code utils to define and configre TVB objects required to run a simulation.
"""
from time import time as zeit

from tvb.simulator.lab import (
    models,
    connectivity,
    coupling,
    integrators,
    noise,
    cortex,
    local_connectivity,
    monitors,
    simulator,
)
import numpy as np


def get_brain_models(dt=2 ** -6):
    # Define the model
    neuron_model = models.Generic2dOscillator(a=np.array([0.3]), tau=np.array([2]))

    # Define the connectivity between regions
    white_matter = connectivity.Connectivity.from_file()
    white_matter.speed = np.array([4.0])
    white_matter.configure()

    # Define the coupling
    white_matter_coupling = coupling.Difference(a=np.array([7e-4]))

    # Define an integration method
    heunint = integrators.HeunStochastic(
        dt=dt, noise=noise.Additive(nsig=np.array([5e-5]))
    )

    # Define a cortical surface
    default_cortex = cortex.Cortex.from_file()
    default_cortex.coupling_strength = np.array([2 ** -10])
    default_cortex.local_connectivity = local_connectivity.LocalConnectivity.from_file()
    default_cortex.region_mapping_data.connectivity = white_matter
    default_cortex.configure()

    return neuron_model, heunint, default_cortex, white_matter, white_matter_coupling


def get_monitors(monitors_needed=["eeg"], **kwargs):
    """Current options are "raw", "tavg", "savg", "eeg" and "bold"."""
    all_mons = []
    if "raw" in monitors_needed:
        mon_raw = monitors.Raw()
        all_mons.append(mon_raw)
    if "tavg" in monitors_needed:
        mon_tavg = monitors.TemporalAverage(period=1.0)
        all_mons.append(mon_tavg)
    if "savg" in monitors_needed:
        mon_savg = monitors.SpatialAverage(period=1.0)
        all_mons.append(mon_savg)
    if "eeg" in monitors_needed:
        mon_eeg = monitors.EEG.from_file(period=0.1)
        all_mons.append(mon_eeg)
    if "bold" in monitors_needed:
        mon_bold = monitors.Bold(period=100)
        all_mons.append(mon_bold)

    return all_mons


def run_simulation(
    sim, duration, monitor_list
):  # Run the simulation - ADAPTED FROM OTHER SURFACE SIMULATION JNB
    start_time = zeit()

    monitor_data = {monitor: {"time": [], "data": []} for monitor in monitor_list}
    idx = list(range(len(monitor_list)))

    for data in sim(simulation_length=duration):
        for i in idx:
            if not data[i] is None:
                monitor_data[monitor_list[i]]["time"].append(data[i][0])
                monitor_data[monitor_list[i]]["data"].append(data[i][1])

    # convert lists to numpy
    for i in idx:
        monitor_data[monitor_list[i]]["time"] = np.array(
            monitor_data[monitor_list[i]]["time"]
        )
        monitor_data[monitor_list[i]]["data"] = np.array(
            monitor_data[monitor_list[i]]["data"]
        )

    print("The simulation took {}s to run".format(round((zeit() - start_time), 1)))
    return monitor_data
