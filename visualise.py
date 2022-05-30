"""Visualisation.
"""

import matplotlib.pyplot as plt

def plot_coil_shape(x_positions, y_positions, coil_type=""):
    plt.plot(x_positions, y_positions, '-o')
    plt.title("%s coil"%coil_type)
    plt.plot()
    