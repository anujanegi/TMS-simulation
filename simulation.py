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


def get_brain_models(
    NMM="oscillator",
    path_to_SC=None,
    path_to_tract_lenghts=None,
    path_to_region_labels=None,
    dt=2 ** -6,
):
    # TODO: docstring
    """Get the brain models for the simulation."""
    # Define the neural mass model
    if NMM == "oscillator":
        neuron_model = models.Generic2dOscillator(a=np.array([0.3]), tau=np.array([2]))
        nsig = np.array([5e-5])
    elif NMM == "reduced_wong_wang":
        neuron_model = models.ReducedWongWang(
            a=np.array([0.27]), w=np.array([1.0]), I_o=np.array([0.3])
        )
        nsig = np.array([1e-5])
    elif NMM == "jansen_rit":
        neuron_model = models.JansenRit(mu=np.array([0.0]), v0=np.array([6.0]))
        phi_n_scaling = (
            neuron_model.a
            * neuron_model.A
            * (neuron_model.p_max - neuron_model.p_min)
            * 0.5
        ) ** 2 / 2.0
        nsig = np.zeros(6)
        nsig[3] = phi_n_scaling

    # Define the connectivity between regions
    if path_to_SC and path_to_region_labels and path_to_tract_lenghts:
        white_matter = connectivity.Connectivity(
            tract_lengths=np.loadtxt(path_to_tract_lenghts),
            weights=np.loadtxt(path_to_SC),
            region_labels=np.loadtxt(path_to_region_labels, dtype=str),
            centres=np.array([0]),
        )
    else:
        white_matter = connectivity.Connectivity.from_file()
        white_matter.speed = np.array([4.0])
    white_matter.configure()

    # Define the coupling
    if NMM == "oscillator":
        white_matter_coupling = coupling.Difference(a=np.array([7e-4]))
    elif NMM == "reduced_wong_wang":
        coupling.Linear(a=np.array([0.5 / 50.0])),
    elif NMM == "jansen_rit":
        white_matter_coupling = coupling.SigmoidalJansenRit(a=np.array([0.0]))
    white_matter_coupling.configure()

    # Define an integration method
    heunint = integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=nsig))

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
