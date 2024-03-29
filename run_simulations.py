import numpy as np
from tvb.simulator.lab import *

from simulation import *
from tms_coil import *
from visualise import *
from stimulus import *
from utils import *
from config import *
import json
import pickle as pkl
import scipy.io as sio
from utils.utils import find_roots
import multiprocessing as mp
import mne


def run_TMS_EEG_simulations(
    subject,
    type,
    do_resting_state_simulation=False,
    overwrite=False,
    efield_type="group_avg",
    dt=1,
    global_coupling=0.0,
    plotting=True,
):
    """
    runs TVB simulation for each subject :
        1. use individual structural connectivity matrices
        2. use group avg or individual TMS efield as stimulus to TVB
        3. run TVB simulation
        4. save TVB simulation results
        5. generate eeg using avg leadfield from Leon
        6. extract P30 and plot figs
    """
    # check if simulation results already exist
    if not overwrite and os.path.isfile(
        os.path.join(
            config.get_TVB_simulation_results_path(subject, type, gc=global_coupling),
            f"{type}_{subject}_{efield_type}_efield_tms_simulation.pkl",
        )
    ):
        print(
            "Loading existing TVB simulation results for subject: ",
            subject,
            " of type: ",
            type,
            "...",
        )
        tms_data = pkl.load(
            open(
                os.path.join(
                    config.get_TVB_simulation_results_path(
                        subject, type, gc=global_coupling
                    ),
                    f"{type}_{subject}_{efield_type}_efield_tms_simulation.pkl",
                ),
                "rb",
            )
        )

    else:

        print(
            "Simulating TMS-EEG protocol for subject: ",
            subject,
            " of type: ",
            type,
            "in TVB...",
        )

        # 1. use individual structural connectivity matrices
        PATH_TO_SC = config.get_subject_structural_connectivity_path(subject, type)
        PATH_TO_TRACT_LENGTHS = config.get_subject_tract_lengths_path(subject, type)
        PATH_TO_REGION_LABELS = config.get_region_labels_path("HCP_MMP1")
        if efield_type == "group_avg":
            PATH_TO_TMS_ELECTRIC_FIELD = config.get_group_average_efield_over_atlas_path(
                type, "HCP_MMP1", "magnE"
            )
        elif efield_type == "individual":
            PATH_TO_TMS_ELECTRIC_FIELD = config.get_efield_atlas_avg_path(
                subject, type, "HCP_MMP1", "magnE"
            )

        # define TVB models
        (
            neuron_model,
            heunint,
            default_cortex,
            white_matter,
            white_matter_coupling,
        ) = get_brain_models(
            NMM="jansen_rit",
            path_to_SC=PATH_TO_SC,
            path_to_region_labels=PATH_TO_REGION_LABELS,
            path_to_tract_lenghts=PATH_TO_TRACT_LENGTHS,
            dt=dt,
        )
        neuron_model.a_1 = np.array([1])
        neuron_model.a_2 = np.array([0.8])
        neuron_model.a_3 = np.array([0.25])
        neuron_model.a_4 = np.array([0.25])
        # neuron_model.nu_max = np.array([0.005])  #maximum firing rate of the neural population
        # neuron_model.r = np.array([0.6])  #Steepness of the sigmoidal transformation
        neuron_model.stvar = np.array([4, 5])

        # 2. use group avg TMS efield as stimulus to TVB
        # using HCP MMP1 atlas
        with open(PATH_TO_TMS_ELECTRIC_FIELD, "r") as f:
            electric_field_strength = json.load(f)

        # renaming to match white_matter.region_labels
        electric_field_strength = {
            f"L_{k[3:]}" if "lh." in k else f"R_{k[3:]}": v
            for k, v in electric_field_strength.items()
        }
        # rearranging values and filling with zeros to match white_matter.region_labels order
        electric_field_strength = np.array(
            [
                electric_field_strength[area]
                if area in electric_field_strength.keys()
                else 0
                for area in white_matter.region_labels
            ]
        )
        max = np.max(electric_field_strength)
        electric_field_strength = [
            i if i > 0.53 * max else 0 for i in electric_field_strength
        ]  # cut off values below 53% of max
        coil = TMS_coil(type="fig8")
        # field_scale = 5e1  # TODO: variable scaling
        electric_field_strength = np.asarray(electric_field_strength)
        coil.get_stimulus_distribution(
            electric_field_strength, white_matter.region_labels
        )

        frequency = 500.0
        stimulus_onset = 1e3
        duration = 1.5e3

        stimulus = get_stimulus(
            electric_field_strength * -1,
            white_matter,
            onset=stimulus_onset,
            frequency=frequency,
            duration=duration,
            type="rTMS",
            dt=dt,
        )

        # 3. run TVB simulation
        monitors = get_monitors(["raw"])
        if do_resting_state_simulation:
            rs_sim = simulator.Simulator(
                model=neuron_model,
                connectivity=white_matter,
                coupling=white_matter_coupling,
                integrator=heunint,
                monitors=monitors,
            )

            rs_sim.coupling.a = np.array([global_coupling])
            rs_sim.connectivity.speed = np.array([3.0])
            rs_sim.configure()
            rs_data = run_simulation(rs_sim, duration, ["raw"])

        # TMS simulation
        tms_sim = simulator.Simulator(
            model=neuron_model,
            connectivity=white_matter,
            coupling=white_matter_coupling,
            integrator=heunint,
            monitors=monitors,
            stimulus=stimulus,
        )
        tms_sim.configure()
        tms_sim.coupling.a = np.array([global_coupling])
        tms_sim.connectivity.speed = np.array([3.0])
        tms_sim.configure()
        tms_data = run_simulation(tms_sim, duration, ["raw"])

        # 4. save TVB simulation results
        # save rs_data as pickle
        if do_resting_state_simulation:
            filename = os.path.join(
                config.get_TVB_simulation_results_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_resting_state_simulation.pkl",
            )
            pkl.dump(rs_data, open(filename, "wb"))

        filename = os.path.join(
            config.get_TVB_simulation_results_path(subject, type, gc=global_coupling),
            f"{type}_{subject}_{efield_type}_efield_tms_simulation.pkl",
        )
        pkl.dump(tms_data, open(filename, "wb"))

    # extract data
    TIME, TMS_RAW = tms_data["raw"]["time"], tms_data["raw"]["data"]
    x0 = 1000 + 30  # P30
    # TODO: extract P30 amplitude from EEG data
    P30 = find_roots(TMS_RAW[:, 0, [9, 10, 11], 0].mean(axis=1), TIME - x0)
    pkl.dump(
        P30,
        open(
            os.path.join(
                config.get_TVB_simulation_results_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_P30_amplitude.pkl",
            ),
            "wb",
        ),
    )

    if plotting:
        # plot and save lfp
        plot_subject_lfp(
            TIME,
            TMS_RAW[:, 0, :, 0],
            plot_args={"title": f"LFP for {type}({subject})", "xlim": [900, 1500]},
            save_path=os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_lfp.png",
            ),
        )

    # 5. generate eeg using avg leadfield from Leon
    ## convert to eeg

    PATH_TO_LEADFIELD = f"/media/anujanegi/Anuja Negi/TMS-simulation/data/TVB_EducaseAD_molecular_pathways_TVB/_{type}/leadfield.mat"

    lead_field = sio.loadmat(PATH_TO_LEADFIELD)
    eeg_data = lead_field["lf_sum"].dot(TMS_RAW[:, 0, :, 0].T).T
    # save eeg data
    pkl.dump(
        eeg_data,
        open(
            os.path.join(
                config.get_TVB_simulation_results_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_eeg_data_educase_lf.pkl",
            ),
            "wb",
        ),
    )

    # convert to mne Evoked
    biosemi64_montage = mne.channels.make_standard_montage("biosemi64")
    # plot eeg cap layout
    f = biosemi64_montage.plot(show_names=True, show=False)
    if plotting:
        f.savefig(
            os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"biosemi64_eeg_cap_layout.png",
            )
        )

    info = mne.create_info(
        ch_names=biosemi64_montage.ch_names, sfreq=1000 / dt, ch_types="eeg"
    )
    evoked = mne.EvokedArray(eeg_data[900:1500, :].T, info, tmin=-100 / 1000)
    evoked.set_montage(biosemi64_montage)
    # save evoked
    pkl.dump(
        evoked,
        open(
            os.path.join(
                config.get_TVB_simulation_results_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_Evoked_educase_lf_biosemi64.pkl",
            ),
            "wb",
        ),
    )

    if plotting:
        # 6. extract P30 and plot figs
        print("Plotting and saving EEG plots...")
        plot_subject_eeg(
            evoked,
            title=f"TMS-EEG for {type} subject {subject}",
            save_path=os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_eeg_data_educase_lf_biosemi64.png",
            ),
        )

        plot_P30_topomap(
            evoked,
            title=f"Topomap of P30 TEP for {type} subject {subject}",
            save_path=os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_P30_topomap_educase_lf_biosemi64.png",
            ),
        )

        plot_TEP_butterfly(
            evoked,
            title=f"TMS evoked potential for {type} subject {subject}",
            save_path=os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_P30_butterfly_educase_lf_biosemi64.png",
            ),
        )

        plot_TEP_butterfly(
            evoked,
            times=np.array([0.03, 0.044, 0.1, 0.18]),
            title=f"TMS evoked potential for {type} subject {subject}",
            save_path=os.path.join(
                config.get_TVB_simulation_results_figures_path(
                    subject, type, gc=global_coupling
                ),
                f"{type}_{subject}_{efield_type}_efield_TEP_butterfly_educase_lf_biosemi64.png",
            ),
        )


if __name__ == "__main__":
    do_resting_state_simulation = False
    overwrite = True

    list_of_args = sys.argv[1:]

    if "simulation_with_avg_efield" in list_of_args:
        for type in config.subjects:
            for subject in config.subjects[type]:
                run_TMS_EEG_simulations(
                    subject=subject,
                    type=type,
                    overwrite=overwrite,
                    do_resting_state_simulation=do_resting_state_simulation,
                    efield_type="group_avg",
                )

    elif "simulation_with_ind_efield" in list_of_args:
        # for type in config.subjects:
        #     for subject in config.subjects[type]:
        type = "AD"
        subject = config.subjects[type][0]
        run_TMS_EEG_simulations(
            subject=subject,
            type=type,
            overwrite=overwrite,
            do_resting_state_simulation=do_resting_state_simulation,
            efield_type="individual",
        )

    else:
        print("Supported options:")
        print(
            "simulation_with_avg_efield: Runs TMS-EGG simulation for each subject using the group averaged TMS efied (using TVB)"
        )
        print(
            "simulation_with_ind_efield: Runs TMS-EGG simulation for each subject using the subject's own TMS efied (using TVB)"
        )
