import sys
import config
import numpy as np
import multiprocessing as mp
from itertools import repeat
from run_simulations import run_TMS_EEG_simulations


def simulation_run_for_all_subjects(global_coupling=0.0):
    do_resting_state_simulation = False
    overwrite = True
    plotting = True

    for type in config.subjects:
        for subject in config.subjects[type]:
            run_TMS_EEG_simulations(
                subject=subject,
                type=type,
                overwrite=overwrite,
                do_resting_state_simulation=do_resting_state_simulation,
                efield_type="individual",
                global_coupling=global_coupling,
                plotting=plotting,
            )


if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    if "explore_GC" in list_of_args:

        n_cores = 4
        p = mp.Pool(processes=n_cores)
        gc_range = np.arange(0.0001, 1, 0.025)
        p.map(simulation_run_for_all_subjects, gc_range)
        p.close()

    else:
        print("Supported arguments:")
        print("explore_GC: explore global coupling values")
