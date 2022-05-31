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
        plt.title("%s Coil Position" % position)
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


def plot_monitor_data(monitor_data, monitor_list, EOI=[3, 41, 42, 58], EOI_labels=None):
    if EOI_labels is None:
        EOI_labels = [str(i) for i in EOI]

    if "tavg" in monitor_list:
        tavg_time, TAVG = monitor_data["tavg"]["time"], monitor_data["tavg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(tavg_time, TAVG[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(tavg_time, TAVG[:, 0, :, 0].mean(axis=1), "r", alpha=1)
        plt.title("Temporal average")
        plt.xlabel("Time (ms)")
        plt.axis([1000, len(tavg_time), -1.2, 2.4])
        plt.axvspan(1500, len(tavg_time), color="whitesmoke")  # stimuli span
        plt.show()

    elif "savg" in monitor_list:
        savg_time, SAVG = monitor_data["savg"]["time"], monitor_data["savg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(savg_time, SAVG[:, 0, :, 0])
        plt.title("Spatial average")
        plt.xlabel("Time (ms)")
        plt.xlim(1000, len(tavg_time))
        plt.axvspan(1500, len(tavg_time), color="whitesmoke")  # stimuli span
        plt.show()

    elif "eeg" in monitor_list:
        eeg_time, EEG = monitor_data["eeg"]["time"], monitor_data["eeg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(eeg_time, EEG[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(eeg_time, EEG[:, 0, EOI, 0], alpha=1, label=EOI_labels)  # EOIs
        plt.title("EEG")
        plt.xlabel("Time (ms)")
        plt.xlim(1000, len(tavg_time))
        plt.axvspan(1500, len(tavg_time), color="whitesmoke")  # stimuli span
        plt.legend()
        plt.show()

    elif "bold" in monitor_list:
        bold_time, BOLD = monitor_data["bold"]["time"], monitor_data["bold"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(bold_time, BOLD[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(bold_time, BOLD[:, 0, EOI, 0], alpha=1, label=EOI_labels)  # EOIs
        plt.title("BOLD")
        plt.xlim(1000, len(tavg_time))
        plt.axvspan(1500, len(tavg_time), color="whitesmoke")  # stimuli span
        plt.legend()
        plt.show()
