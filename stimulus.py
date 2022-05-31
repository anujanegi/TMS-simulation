"""Defines various TMS stimuli types
"""
from tvb.simulator.lab import *
import numpy as np

from tvb.basic.neotraits.api import NArray, Attr


class MultiStimuliRegion(patterns.StimuliRegion):
    connectivity = Attr(connectivity.Connectivity, required=False)
    temporal = Attr(field_type=equations.TemporalApplicableEquation, required=False)
    weight = NArray(required=False)

    def __init__(self, *stimuli):
        super(MultiStimuliRegion, self).__init__()
        self.stimuli = stimuli

    def configure_space(self, *args, **kwds):
        [stim.configure_space(*args, **kwds) for stim in self.stimuli]

    def configure_time(self, *args, **kwds):
        [stim.configure_time(*args, **kwds) for stim in self.stimuli]

    def __call__(self, *args, **kwds):
        return np.array([stim(*args, **kwds) for stim in self.stimuli]).sum(axis=0)


def make_train(weighting, connectivity, **params):
    eqn_t = equations.PulseTrain()
    eqn_t.parameters.update(params)
    stim = patterns.StimuliRegion(
        temporal=eqn_t, connectivity=connectivity, weight=weighting
    )
    return stim


def get_stimulus(weighting, connectivity, duration=2.5e3, dt=2 ** -6, type="rTMS"):
    """Returns intialised TVB object based on stimuls type

    Args:
        type (str, optional): Options are dummy, rTMS and iTBS. Defaults to 'rTMS'.
    """
    if type == "SinglePulse":
        # Define the stimulus' temporal profile
        eqn_t = equations.PulseTrain()
        eqn_t.parameters["onset"] = 1.5e3  # Delay from beginning of sim [ms]
        eqn_t.parameters["T"] = 5000.0  # Frequency of pulse train [ms]
        eqn_t.parameters["tau"] = 5.0  # Period of pulse train [ms]
        eqn_t.parameters.update()

        # Create an object to stimulate the desired regions
        stim = patterns.StimuliRegion(
            temporal=eqn_t, connectivity=connectivity, weight=weighting
        )

    elif type == "rTMS":
        # rTMS
        # Define the stimulus' temporal profile
        eqn_t = equations.PulseTrain()
        eqn_t.parameters["onset"] = 1.5e3  # Delay from beginning of sim [ms]
        eqn_t.parameters["T"] = 200.0  # Frequency of pulse train [ms] #5Hz rTMS
        eqn_t.parameters["tau"] = 5.0  # Period of pulse train [ms]
        eqn_t.parameters.update()
        # Create an object to stimulate the desired regions
        stim = patterns.StimuliRegion(
            temporal=eqn_t, connectivity=connectivity, weight=weighting
        )

    elif type == "iTBS":
        # iTBS
        # triplet 50 Hz bursts, repeated at 5 Hz, 2s on / 8s off, 600 pps, 3min 9s, 1 session per day, 20 sessions ((Klomjai et al., 2015; Lefaucheur, 2019)

        onset = 1.5e3
        train1 = make_train(weighting, connectivity, onset=onset, T=200.0, tau=12.0)
        train2 = make_train(
            weighting, connectivity, onset=onset + 20, T=200.0, tau=10.0
        )
        train3 = make_train(
            weighting, connectivity, onset=onset + 40, T=200.0, tau=10.0
        )

        stim = MultiStimuliRegion(train1, train2, train3)

    else:
        raise ValueError(
            "Not a supported stimulus type. Current options are dummy, rTMS and iTBS"
        )

    # Configure the stimulus in space and time
    stim.configure_space()
    # Set the length of the stimulus
    stim.configure_time(np.arange(0.0, duration, dt))
    return stim
