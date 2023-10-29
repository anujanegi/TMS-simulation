import sys
import config
import numpy as np
import multiprocessing as mp
from itertools import repeat
from run_simulations import run_TMS_EEG_simulations


def simulation_run_for_all_subjects(global_coupling=0.0, type=None):
    do_resting_state_simulation = False
    overwrite = False
    plotting = False

    if type is None:
        print("Please specify Subject type for simulation")
        return
    
    for subject in config.subjects[type]:
        run_TMS_EEG_simulations(
            subject=subject,
            type=type,
            overwrite=overwrite,
            do_resting_state_simulation=do_resting_state_simulation,
            efield_type="individual",
            global_coupling=global_coupling,
            plotting=plotting,
            experiment='test_new_lf'
        )


if __name__ == "__main__":
    list_of_args = sys.argv[1:]

    if "explore_GC" in list_of_args:
        
        gc_range = np.arange(0, 1.3, 0.1)

        # do AD
        n_cores = 4
        p = mp.Pool(processes=n_cores)
        p.starmap(simulation_run_for_all_subjects, zip(gc_range, repeat("AD")))
        p.close()
        
        # do MCI
        n_cores = 4
        p = mp.Pool(processes=n_cores)
        p.starmap(simulation_run_for_all_subjects, zip(gc_range, repeat("MCI")))
        p.close()
        
        # do HC
        n_cores = 4
        p = mp.Pool(processes=n_cores)
        p.starmap(simulation_run_for_all_subjects, zip(gc_range, repeat("HC")))
        p.close()
        
    else:
        print("Supported arguments:")
        print("explore_GC: explore global coupling values")
