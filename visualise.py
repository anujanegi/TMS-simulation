"""Visualisation.
"""

from operator import pos
from re import S
import matplotlib.pyplot as plt
from tvb.simulator.plot.tools import plot_pattern


def plot_coil_shape(x_positions, y_positions, coil_type=""):
    plt.plot(x_positions, y_positions, "-o")
    plt.title("%s coil" % coil_type)
    plt.plot()


def plot_coil_on_cortical_surface(coil, ind, cortex, title=""):
    # Plot the coil position

    # Plot a representation of the cortical surface
    ax = plt.subplot(111, projection="3d")
    x, y, z = cortex.vertices.T
    ax.plot_trisurf(x, y, z, triangles=cortex.triangles, alpha=0.1, edgecolor="none")
    ax.view_init(30, -60)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot the rotated and translated coils
    ax.scatter(coil[ind][:, 0], coil[ind][:, 1], coil[ind][:, 2], c="C1")
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_stimulus(stimulus, position, type=""):
    if type == "iTBS":
        plt.imshow(stimulus(), interpolation="none", aspect="auto")
        plt.xlabel("Time")
        plt.title("%s Coil Position"%position)
        plt.ylabel("Space")
        plt.colorbar()
        plt.show()
    else:
        # Plot the stimulus in space and time
        plot_pattern(stimulus)
        fig = plt.gcf()
        fig.set_size_inches(6, 5)
        fig.suptitle("%s Coil Position" % position, fontsize=24, y=1.05)
        fig.tight_layout()
        plt.show()
