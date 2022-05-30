"""Code utils to define and configre TVB objects required to run a simulation.
"""

from tvb.simulator.lab import models, connectivity, coupling, integrators, noise, cortex, local_connectivity
import numpy as np


def get_brain_models():
    # Define the model
    neuron_model = models.Generic2dOscillator(a=np.array([0.3]), tau=np.array([2]))

    # Define the connectivity between regions
    white_matter = connectivity.Connectivity.from_file()
    white_matter.speed = np.array([4.0])
    white_matter.configure()

    # Define the coupling
    white_matter_coupling = coupling.Difference(a=np.array([7e-4])) 

    # Define an integration method
    dt = 2**-6
    heunint = integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([5e-5])))

    # Define a cortical surface
    default_cortex = cortex.Cortex.from_file()
    default_cortex.coupling_strength = np.array([2**-10])
    default_cortex.local_connectivity = local_connectivity.LocalConnectivity.from_file()
    default_cortex.region_mapping_data.connectivity = white_matter
    default_cortex.configure()
    
    return neuron_model, heunint, default_cortex