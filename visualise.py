"""Visualisation.
"""

import sys
import numpy as np
from re import S
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.cm as cm
from tvb.simulator.plot.tools import plot_pattern
import pyvista as pv
import config
import os
import simnibs
import json
import utils.utils as utils


def plot_coil_shape(x_positions, y_positions, coil_type=""):
    plt.plot(x_positions, y_positions, "-o")
    plt.title("%s coil" % coil_type)
    plt.show()


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


def plot_stimulus(stimulus, position="", type=""):
    if type == "iTBS":
        plt.imshow(stimulus(), interpolation="none", aspect="auto")
        plt.xlabel("Time")
        plt.title("%s TMS stimulus" % position)
        plt.ylabel("Space")
        plt.colorbar()
        plt.show()
    else:
        # Plot the stimulus in space and time
        plot_pattern(stimulus)
        fig = plt.gcf()
        fig.set_size_inches(6, 5)
        fig.suptitle("%s TMS stimulus" % position, fontsize=24, y=1.05)
        fig.tight_layout()
        plt.show()


def plot_monitor_data(
    monitor_data, monitor_list, duration, EOI=[3, 41, 42, 58], EOI_labels=None
):
    if EOI_labels is None:
        EOI_labels = [str(i) for i in EOI]

    if "tavg" in monitor_list:
        tavg_time, TAVG = monitor_data["tavg"]["time"], monitor_data["tavg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(tavg_time, TAVG[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(tavg_time, TAVG[:, 0, :, 0].mean(axis=1), "r", alpha=1)
        plt.title("Temporal average")
        plt.xlabel("Time (ms)")
        plt.axis([1000, duration, -1.2, 2.4])
        plt.axvspan(1500, duration, color="whitesmoke")  # stimuli span
        plt.show()

    if "savg" in monitor_list:
        savg_time, SAVG = monitor_data["savg"]["time"], monitor_data["savg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(savg_time, SAVG[:, 0, :, 0])
        plt.title("Spatial average")
        plt.xlabel("Time (ms)")
        plt.xlim(1000, duration)
        plt.axvspan(1500, duration, color="whitesmoke")  # stimuli span
        plt.show()

    if "eeg" in monitor_list:
        eeg_time, EEG = monitor_data["eeg"]["time"], monitor_data["eeg"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(eeg_time, EEG[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(eeg_time, EEG[:, 0, EOI, 0], alpha=1, label=EOI_labels)  # EOIs
        plt.title("EEG")
        plt.xlabel("Time (ms)")
        plt.xlim(1000, duration)
        plt.axvspan(1500, duration, color="whitesmoke")  # stimuli span
        plt.legend()
        plt.show()

    if "bold" in monitor_list:
        bold_time, BOLD = monitor_data["bold"]["time"], monitor_data["bold"]["data"]

        plt.figure(figsize=(11, 3))
        plt.plot(bold_time, BOLD[:, 0, :, 0], "k", alpha=0.1)
        plt.plot(bold_time, BOLD[:, 0, EOI, 0], alpha=1, label=EOI_labels)  # EOIs
        plt.title("BOLD")
        plt.xlim(1000, duration)
        plt.axvspan(1500, duration, color="whitesmoke")  # stimuli span
        plt.legend()
        plt.show()


def plot_eeg_comparison(EEG_A, EEG_B, duration, title="EEG", labels=["RS", "TMS"]):
    """Plots EOI EEG data from 2 different simulations.

    Args:
        EEG_A (_type_): EEG monitor data from simulation A
        EEG_B (_type_): EEG monitor data from simulation A
    """

    eeg_time = EEG_A["time"]
    EEG_A, EEG_B = EEG_A["data"], EEG_B["data"]

    plt.figure(figsize=(11, 3))
    plt.plot(eeg_time, EEG_A[:, 0, [3, 41, 42, 58], 0], "powderblue", alpha=1)  # EOIs
    plt.plot(
        eeg_time,
        EEG_A[:, 0, [3, 41, 42, 58], 0].mean(axis=1),
        "b",
        alpha=1,
        label=labels[0],
    )  # EOIs
    plt.plot(eeg_time, EEG_B[:, 0, [3, 41, 42, 58], 0], "mistyrose", alpha=1)  # EOIs
    plt.plot(
        eeg_time,
        EEG_B[:, 0, [3, 41, 42, 58], 0].mean(axis=1),
        "r",
        alpha=1,
        label=labels[1],
    )  # EOIs
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.xlim(1400, duration)
    plt.axvspan(1500, duration, color="whitesmoke")  # stimuli span
    plt.legend()
    plt.show()


def plot_TEP(data, EOI=[3, 41, 42, 58], stimuli_onset=1500, stimulus="TMS"):
    eeg_time = data["time"]
    EEG = data["data"]
    plt.figure(figsize=(11, 3))

    plt.plot(eeg_time, EEG[:, 0, EOI, 0], "k", alpha=0.1)
    plt.plot(eeg_time, EEG[:, 0, EOI, 0].mean(axis=1), "r", alpha=1)  # EOIs

    plt.title("TEP for %s" % stimulus)
    plt.xlabel("Time (ms)")
    plt.xlim(stimuli_onset, stimuli_onset + 300)

    plt.axvline(x=1500 + 200, color="g", linestyle="--", label="P200")  # P200
    plt.axvline(x=1500 + 120, color="m", linestyle="--", label="N120")  # N100
    plt.legend()
    plt.show()


# Visualization
def plot_surface_mpl(
    vtx,
    tri,
    data=None,
    rm=None,
    reorient="tvb",
    view="superior",
    shaded=False,
    ax=None,
    figsize=(6, 4),
    title=None,
    lthr=None,
    uthr=None,
    nz_thr=1e-20,
    shade_kwargs={
        "edgecolors": "k",
        "linewidth": 0.1,
        "alpha": None,
        "cmap": "coolwarm",
        "vmin": None,
        "vmax": None,
    },
):

    r"""Plot surfaces, surface patterns, and region patterns with matplotlib

    This is a general-use function for neuroimaging surface-based data, and
    does not necessarily require construction of or interaction with tvb
    datatypes.

    See also:  plot_surface_mpl_mv
    taken from https://nbviewer.jupyter.org/urls/s3.amazonaws.com/replicating_spiegler2016/replicating_spiegler2016__html_nb.ipynb).


    Parameters
    ----------

    vtx           : N vertices x 3 array of surface vertex xyz coordinates

    tri           : N faces x 3 array of surface faces

    data          : array of numbers to colour surface with. Can be either
                    a pattern across surface vertices (N vertices x 1 array),
                    or a pattern across the surface's region mapping
                    (N regions x 1 array), in which case the region mapping
                    bust also be given as an argument.

    rm            : region mapping - N vertices x 1 array with (up to) N
                    regions unique values; each element specifies which
                    region the corresponding surface vertex is mapped to

    reorient      : modify the vertex coordinate frame and/or orientation
                    so that the same default rotations can subsequently be
                    used for image views. The standard coordinate frame is
                    xyz; i.e. first,second,third axis = left-right,
                    front-back, and up-down, respectively. The standard
                    starting orientation is axial view; i.e. looking down on
                    the brain in the x-y plane.

                    Options:

                      tvb (default)   : swaps the first 2 axes and applies a rotation

                      fs              : for the standard freesurfer (RAS) orientation;
                                        e.g. fsaverage lh.orig.
                                        No transformations needed for this; so is
                                        gives same result as reorient=None

    view          : specify viewing angle.

                    This can be done in one of two ways: by specifying a string
                    corresponding to a standard viewing angle, or by providing
                    a tuple or list of tuples detailing exact rotations to apply
                    around each axis.

                    Standard view options are:

                    lh_lat / lh_med / rh_lat / rh_med /
                    superior / inferior / posterior / anterior

                    (Note: if the surface contains both hemispheres, then medial
                     surfaces will not be visible, so e.g. 'rh_med' will look the
                     same as 'lh_lat')

                    Arbitrary rotations can be specied by a tuple or a list of
                    tuples, each with two elements, the first defining the axis
                    to rotate around [0,1,2], the second specifying the angle in
                    degrees. When a list is given the rotations are applied
                    sequentially in the order given.

                    Example: rotations = [(0,45),(1,-45)] applies 45 degrees
                    rotation around the first axis, followed by 45 degrees rotate
                    around the second axis.

    lthr/uthr     : lower/upper thresholds - set to zero any datapoints below /
                    above these values

    nz_thr        : near-zero threshold - set to zero all datapoints with absolute
                    values smaller than this number. Default is a very small
                    number (1E-20), which unless your data has very small numbers,
                    will only mask out actual zeros.

    shade_kwargs  : dictionary specifiying shading options

                    Most relevant options (see matplotlib 'tripcolor' for full details):

                      - 'shading'        (either 'gourand' or omit;
                                          default is 'flat')
                      - 'edgecolors'     'k' = black is probably best
                      - 'linewidth'      0.1 works well; note that the visual
                                         effect of this will depend on both the
                                         surface density and the figure size
                      - 'cmap'           colormap
                      - 'vmin'/'vmax'    scale colormap to these values
                      - 'alpha'          surface opacity

    ax            : figure axis

    figsize       : figure size (ignore if ax provided)

    title         : text string to place above figure




    Usage
    -----


    Basic freesurfer example:

    import nibabel as nib
    vtx,tri = nib.freesurfer.read_geometry('subjects/fsaverage/surf/lh.orig')
    plot_surface_mpl(vtx,tri,view='lh_lat',reorient='fs')



    Basic tvb example:

    ctx = cortex.Cortex.from_file(source_file = ctx_file,
                                  region_mapping_file =rm_file)
    vtx,tri,rm = ctx.vertices,ctx.triangles,ctx.region_mapping
    conn = connectivity.Connectivity.from_file(conn_file); conn.configure()
    isrh_reg = conn.is_right_hemisphere(range(conn.number_of_regions))
    isrh_vtx = np.array([isrh_reg[r] for r in rm])
    dat = conn.tract_lengths[:,5]

    plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat,view='inferior',title='inferior')

    fig, ax = plt.subplots()
    plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat, view=[(0,-90),(1,55)],ax=ax,
                     title='lh angle',shade_kwargs={'shading': 'gouraud', 'cmap': 'rainbow'})


    """

    # Copy things to make sure we don't modify things
    # in the namespace inadvertently.

    vtx, tri = vtx.copy(), tri.copy()
    if data is not None:
        data = data.copy()

    # 1. Set the viewing angle

    if reorient == "tvb":
        # The tvb default brain has coordinates in the order
        # yxz for some reason. So first change that:
        vtx = np.array([vtx[:, 1], vtx[:, 0], vtx[:, 2]]).T.copy()

        # Also need to reflect in the x axis
        vtx[:, 0] *= -1

    # (reorient == 'fs' is same as reorient=None; so not strictly needed
    #  but is included for clarity)

    # ...get rotations for standard view options

    if view == "lh_lat":
        rots = [(0, -90), (1, 90)]
    elif view == "lh_med":
        rots = [(0, -90), (1, -90)]
    elif view == "rh_lat":
        rots = [(0, -90), (1, -90)]
    elif view == "rh_med":
        rots = [(0, -90), (1, 90)]
    elif view == "superior":
        rots = None
    elif view == "inferior":
        rots = (1, 180)
    elif view == "anterior":
        rots = (0, -90)
    elif view == "posterior":
        rots = [(0, -90), (1, 180)]
    elif (type(view) == tuple) or (type(view) == list):
        rots = view

    # (rh_lat is the default 'view' argument because no rotations are
    #  for that one; so if no view is specified when the function is called,
    #  the 'rh_lat' option is chose here and the surface is shown 'as is'

    # ...apply rotations

    if rots is None:
        rotmat = np.eye(3)
    else:
        rotmat = get_combined_rotation_matrix(rots)
    vtx = np.dot(vtx, rotmat)

    # 2. Sort out the data

    # ...if no data is given, plot a vector of 1s.
    #    if using region data, create corresponding surface vector
    if data is None:
        data = np.ones(vtx.shape[0])
    elif data.shape[0] != vtx.shape[0]:
        data = np.array([data[r] for r in rm])

    # ...apply thresholds
    if uthr:
        data *= data < uthr
    if lthr:
        data *= data > lthr
    data *= np.abs(data) > nz_thr

    # 3. Create the surface triangulation object

    x, y, z = vtx.T
    tx, ty, tz = vtx[tri].mean(axis=1).T
    tr = Triangulation(x, y, tri[np.argsort(tz)])

    # 4. Make the figure

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # if shade = 'gouraud': shade_opts['shade'] =
    tc = ax.tripcolor(tr, np.squeeze(data), **shade_kwargs)

    ax.set_aspect("equal")
    ax.axis("off")

    if title is not None:
        ax.set_title(title)


def get_combined_rotation_matrix(rotations):
    """Return a combined rotation matrix from a dictionary of rotations around
    the x,y,or z axes"""
    rotmat = np.eye(3)

    if type(rotations) is tuple:
        rotations = [rotations]
    for r in rotations:
        newrot = get_rotation_matrix(r[0], r[1])
        rotmat = np.dot(rotmat, newrot)
    return rotmat


def get_rotation_matrix(rotation_axis, deg):

    """Return rotation matrix in the x,y,or z plane"""

    # (note make deg minus to change from anticlockwise to clockwise rotation)
    th = -deg * (np.pi / 180)  # convert degrees to radians

    if rotation_axis == 0:
        return np.array(
            [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
        )
    elif rotation_axis == 1:
        return np.array(
            [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
        )
    elif rotation_axis == 2:
        return np.array(
            [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
        )


def plot_activity_on_brain(vtx, tri, rm, monitor_data, time_stamps, title=""):

    time, MON = monitor_data["time"], monitor_data["data"]
    fig, ax = plt.subplots(ncols=10, nrows=2, figsize=(15, 3))
    cmap = cm.Reds  # JAN SAID cmap IS A DEFAULT VARIABLE, MAYBE NAME ERROR
    cmap.set_under(color="w")

    kws = {
        "edgecolors": "k",
        "vmin": 0.1,
        "cmap": cmap,
        "vmax": 0.6,
        "alpha": None,
        "linewidth": 0.01,
    }  # MAYBE SCALING IS THE PROBLEM? SECOND CALL # OUTLIERS CHANGING THIS?

    for t_it, t in enumerate(time_stamps):

        dat = np.absolute(MON[t, 0, :, 0])

        plot_surface_mpl(
            vtx=vtx,
            tri=tri,
            data=dat,
            rm=rm,
            ax=ax[0][t_it],  # VTX = VTX IS NOT DESIRABLE
            shade_kwargs=kws,
            view="lh_lat",
        )

        plot_surface_mpl(
            vtx=vtx,
            tri=tri,
            data=dat,
            rm=rm,
            ax=ax[1][t_it],
            shade_kwargs=kws,
            view="rh_lat",
        )

        ax[0][t_it].set_title("t=%1.1fms" % time[t])
    fig.suptitle(title, fontsize=15)


def _plot_msh(
    meshes, scalar_name, title=None, plot_args={}, save_path="./", save_name=None
):
    cmap = "seismic" if plot_args.get("cmap") is None else plot_args["cmap"]
    position = plot_args.get("position")
    camera_roll = plot_args.get("camera_roll")
    title = scalar_name if title is None else title
    save_name = title if save_name is None else save_name
    limit = (
        plot_args.get("limit")
        if plot_args.get("limit")
        else [min(meshes[0][scalar_name]), max(meshes[0][scalar_name])]
    )

    p = pv.Plotter(window_size=[800, 800], off_screen=True)
    sargs = {"color": "black", "title": title}
    for mesh in meshes:
        p.add_mesh(
            mesh, scalars=scalar_name, clim=limit, cmap=cmap, scalar_bar_args=sargs
        )
    # p.add_text(
    #         f"Simulated TMS E-field magnitude for {type}[{subject}]",
    #         font_size=10,
    #         color="black",
    #         position="upper_edge",
    #     )
    p.set_position(position)
    p.camera.roll = camera_roll
    p.screenshot(
        os.path.join(save_path, f"{save_name}.png"), transparent_background=True
    )

    print(f"Saved {save_name} in {save_path}")


def _plot_HCP_MMP1_atlas(
    atlas_json, title=None, plot_args={}, save_path="./", save_name=None
):
    ef_idxRegions = atlas_json.copy()
    ef_idxRegions = {
        "L_" + k[3:] if "lh." in k else "R_" + k[3:]: v
        for k, v in ef_idxRegions.items()
    }
    efield = [0]
    for key in ef_idxRegions:
        efield.append(ef_idxRegions[key])
    efield.insert(181, 0)  # region 0 and 181 are left and right subcortical areas

    # generate a glasser msh
    dir = config.get_glasser_msh_path()
    rl = np.loadtxt(os.path.join(dir, "_regions.txt"), dtype=str)
    ls_mesh = []
    for reg in rl:
        mesh = pv.read(os.path.join(dir, str(reg)))
        ls_mesh.append(mesh)

    for i in range(len(ls_mesh)):
        ls_mesh[i].cell_data["data"] = efield[i]

    plot_args["position"] = (
        plot_args.get("position")
        if plot_args.get("position")
        else (-515.2932430512969, 227.73440753777155, -153.4630638823102)
    )
    plot_args["camera_roll"] = (
        plot_args.get("camera_roll") if plot_args.get("camera_roll") else -13
    )
    _plot_msh(ls_mesh, "data", title, plot_args, save_path, save_name)


def plot_magnE_on_subject(subject, type, limit):
    mesh = pv.read(config.get_efield_head_mesh_path(subject, type))
    _plot_msh(
        [mesh],
        "magnE",
        title=f"E-field magnitude",
        plot_args={
            "limit": limit,
            "cmap": "rainbow",
            "position": (-471.96987340389745, 88.16278454405862, 268.98795757532633),
            "camera_roll": 93,
        },
        save_path=config.get_TMS_efield_path(subject, type),
        save_name=f"{subject}_magnE",
    )


def plot_magnE_on_atlas(subject, type, atlas_name="HCP_MMP1", limit=None):
    # load .json efield averaged over the atlas
    with open(config.get_efield_atlas_avg_path(subject, type), "r") as f:
        efield_avg_atlas = json.load(f)
    _plot_HCP_MMP1_atlas(
        efield_avg_atlas,
        title=f"E-field magnitude",
        plot_args={"limit": limit, "cmap": "rainbow"},
        save_path=config.get_TMS_efield_path(subject, type),
        save_name=f"{subject} magnE over {atlas_name}",
    )


# plot magnE on fsaverage
def plot_magnE_on_fsaverage(subject, type, limit):
    mesh = pv.read(config.get_efield_fsavg_overlay_mesh_path(subject, type))
    _plot_msh(
        [mesh],
        "magnE",
        title="E-field magnitude",
        save_path=config.get_TMS_efield_path(subject, type),
        save_name=f"{subject} magnE on fsaverage",
        plot_args={
            "limit": limit,
            "cmap": "rainbow",
            "position": (-471.96987340389745, 88.16278454405862, 268.98795757532633),
            "camera_roll": 93,
        },
    )


def _plot(x, y, title, xlim=[0, 0]):
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.plot(x, y)
    plt.xlim(xlim[0], xlim[1])
    lims = ax.get_xlim()
    i = np.where((x > lims[0]) & (x < lims[1]))[0]
    ax.set_ylim(y[i].min(), y[i].max())
    plt.title(title)

    return f, ax


def plot_subject_lfp(time, lfp, onset=1000, plot_args={}, save_path=None):
    f, ax = _plot(time, lfp, **plot_args)
    ax.axvline(x=onset, color="m", linestyle="--", label="TMS pulse")
    ax.axvline(x=onset + 30, color="b", linestyle="--", label="P30")
    ax.legend()
    ax.set_ylabel("Potential[y1-y2] (mV)")
    ax.set_xlabel("Time (ms)")
    if save_path:
        plt.savefig(save_path, transaprent=True)


def plot_subject_eeg(evoked, title=None, save_path=None):
    """Plots TMS-EEG using mne

    Args:
        evoked (_type_): MNE Evoked object
    """
    title = title if title else "TMS-EEG"
    f, ax = plt.subplots(figsize=(10, 4))
    ax.axvline(x=0, color="b", linestyle="--", label="TMS pulse")
    ax.legend()
    f = evoked.plot(show=False, time_unit="ms", axes=ax, titles=title)
    if save_path:
        f.savefig(save_path, transparent=True)


def plot_P30_topomap(evoked, title=None, save_path=None):
    """Plots P30 topomap

    Args:
        evoked (_type_): MNE Evoked object
        title (_type_, optional): _description_. Defaults to None.
        save_path (_type_, optional): _description_. Defaults to None.
    """
    title = title if title else "P30 topomap"
    fig = evoked.plot_topomap(
        times=np.array([0.03]),
        ch_type="eeg",
        time_unit="ms",
        show_names=False,
        show=False,
        size=4,
    )
    fig.axes[0].set_title(title, fontsize=16)
    if save_path:
        fig.savefig(save_path, transparent=True)


def plot_P30_butterfly(evoked, title, save_path):
    title = title if title else "P30 butterfly plot"
    topomap_args = dict(ch_type="eeg", time_unit="ms", show_names=False)
    ts_args = dict(time_unit="ms", show_names=False, titles=title)
    fig = evoked.plot_joint(
        times=np.array([0.03]), show=False, ts_args=ts_args, topomap_args=topomap_args
    )
    if save_path:
        fig.savefig(save_path, transparent=True)


if __name__ == "__main__":

    list_of_args = sys.argv[1:]

    if "plot_magnE_on_subject" in list_of_args:
        for type in config.subjects:
            limit = utils.get_lim_efield_in_type(type)
            for subject in config.subjects[type]:
                plot_magnE_on_subject(subject, type, limit=limit)

    elif "plot_subjects_magnE_on_atlas" in list_of_args:

        # limit = utils.get_lim_efield_on_atlas()
        for type in config.subjects:
            limit = utils.get_lim_efield_in_type(type)
            for subject in config.subjects[type]:
                plot_magnE_on_atlas(subject, type, limit=limit)

    elif "plot_subjects_magnE_on_fsaverage" in list_of_args:
        for type in config.subjects:
            limit = utils.get_lim_efield_in_type(type)
            for subject in config.subjects[type]:
                plot_magnE_on_fsaverage(subject, type, limit=limit)

    elif "plot_group_stats_on_fsaverage" in list_of_args:
        efield_type = "magnE"
        stats = ["avg", "std"]
        for type in config.subjects:
            # get avg and std efields and plot
            for stat in stats:
                limit = utils.get_lim_stat_efield_in_type(type, stat=stat)
                mesh = config.get_efield_stats_path(
                    type, stat=stat, over="fsavg", efield_type=efield_type
                )
                _plot_msh(
                    [pv.read(mesh)],
                    stat,
                    title=f"{stat} of Efield",
                    save_name=f"{efield_type}_{stat}_{type}_fsavg",
                    save_path=config.get_analysis_fig_path(),
                    plot_args={
                        "limit": limit,
                        "cmap": "rainbow",
                        "position": (
                            -471.96987340389745,
                            88.16278454405862,
                            268.98795757532633,
                        ),
                        "camera_roll": 93,
                    },
                )

    elif "plot_group_stats_on_atlas" in list_of_args:
        efield_type = "magnE"
        stats = ["avg", "std"]
        for type in config.subjects:
            # get avg and std efields and plot
            for stat in stats:
                limit = utils.get_lim_stat_efield_in_type(type, stat=stat)
                with open(
                    config.get_efield_stats_path(
                        type, stat=stat, over="HCP_MMP1", efield_type=efield_type
                    ),
                    "r",
                ) as f:
                    efield_atlas = json.load(f)
                _plot_HCP_MMP1_atlas(
                    efield_atlas,
                    title=f"{stat} of Efield",
                    plot_args={"limit": limit, "cmap": "rainbow"},
                    save_path=config.get_analysis_fig_path(),
                    save_name=f"{efield_type}_{stat}_{type}_HCP_MMP1",
                )

    elif "plot_efield_difference_on_fsaverage" in list_of_args:
        # load difference efields and plot
        mesh_list = [
            fname
            for fname in os.listdir(config.get_analysis_data_path())
            if fname.endswith(".msh") and "difference" in fname and "fsavg" in fname
        ]
        limit = utils.get_lim_efield_difference(over="fsavg")
        for mesh in mesh_list:
            _plot_msh(
                [pv.read(os.path.join(config.get_analysis_data_path(), mesh))],
                mesh[:-10],
                title=f"Efield difference",
                save_name=mesh[:-4],
                save_path=config.get_analysis_fig_path(),
                plot_args={
                    "limit": limit,
                    "cmap": "seismic",
                    "position": (
                        -471.96987340389745,
                        88.16278454405862,
                        268.98795757532633,
                    ),
                    "camera_roll": 93,
                },
            )

    elif "plot_efield_difference_on_atlas" in list_of_args:
        # load difference efields and plot
        json_list = [
            fname
            for fname in os.listdir(config.get_analysis_data_path())
            if fname.endswith(".json") and "difference" in fname and "HCP_MMP1" in fname
        ]
        limit = utils.get_lim_efield_difference(over="HCP_MMP1")
        for json_file in json_list:
            with open(
                os.path.join(config.get_analysis_data_path(), json_file), "r"
            ) as f:
                efield_atlas = json.load(f)
            _plot_HCP_MMP1_atlas(
                efield_atlas,
                title=f"Efield difference",
                plot_args={"limit": limit, "cmap": "seismic"},
                save_path=config.get_analysis_fig_path(),
                save_name=json_file[:-5],
            )

    else:
        print("Supported options:")
        print(
            "plot_magnE_on_subject: plots the magnitude of the E-field on the subject head"
        )
        print(
            "plot_subjects_magnE_on_fsaverage: plots the magnitude of the E-field on the fsavg head for all subjects"
        )
        print(
            "plot_subjects_magnE_on_atlas: plots the magnitude of the E-field on the fsavg head for a specific atlas for all subjects"
        )
        print(
            "plot_group_stats_on_fsaverage: plots the avg and std of efield for each group over FsAverage head"
        )
        print(
            "plot_group_stats_on_atlas: plots the avg and std of efield for each group over atlas"
        )
        print(
            "plot_efield_difference_on_fsaverage: plots the difference between the E-field magnitude between groups on an FsAverage head"
        )
        print(
            "plot_efield_difference_on_atlas: plots the difference between the E-field magnitude between groups on an atlas"
        )
