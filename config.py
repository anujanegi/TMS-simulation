import os

data_path = "/home/anujanegi/tj/TMS-simulation/data"
dataset_path = "/home/anujanegi/tj/TMS-simulation/data/ADNI"
subjects = {
    "AD": ["011_S_4547", "036_S_4715", "041_S_4974", "114_S_6039", "168_S_6142"],
    "MCI": ["002_S_1155", "002_S_4229", "002_S_4654", "002_S_1261", "003_S_1122"],
    "HC": ["002_S_0413", "002_S_5178", "002_S_4213", "002_S_1280", "002_S_4799"],
}


def get_m2m_path(subject, type):
    return os.path.join(dataset_path, type, subject, f"m2m_{subject[-4:]}")


# get efield head mesh path
def get_efield_head_mesh_path(subject, type):
    # Read the simulation result mapped to the gray matter surface
    return os.path.join(
        dataset_path,
        type,
        subject,
        "TMS_efield",
        "subject_overlays",
        f"{subject[-4:]}_TMS_1-0001_MagStim_D70_scalar_central.msh",
    )


# get efield transformed to fsavg pace head mesh path
def get_efield_fsavg_overlay_mesh_path(subject, type):
    return os.path.join(
        dataset_path,
        type,
        subject,
        "TMS_efield",
        "fsavg_overlays",
        f"{subject[-4:]}_TMS_1-0001_MagStim_D70_scalar_fsavg.msh",
    )


# get efield json for subject averaged over atlas path
def get_efield_atlas_avg_path(subject, type, atlas="HCP_MMP1", efield_type="magnE"):
    return os.path.join(
        dataset_path,
        type,
        subject,
        "TMS_efield",
        f"{subject}_efield_over_{atlas}_{efield_type}.json",
    )


def get_TMS_efield_path(subject, type):
    return os.path.join(dataset_path, type, subject, "TMS_efield")


def get_subject_path(subject, type):
    return os.path.join(dataset_path, type, subject)


def get_analysis_data_path():
    return os.path.join(dataset_path, "analysis", "data")


def get_analysis_fig_path():
    return os.path.join(dataset_path, "analysis", "figures")


def get_glasser_msh_path():
    return os.path.join(data_path, "Glasser_adapted_1_edit")


def get_efield_stats_path(type, efield_type="magnE", stat="avg", over="fsavg"):
    file_type = ".msh" if over == "fsavg" else ".json"
    return os.path.join(
        get_analysis_data_path(), f"{efield_type}_{stat}_{type}_{over}{file_type}"
    )


def get_subject_structural_connectivity_path(subject, type):
    return os.path.join(
        dataset_path, type, subject, "DWI_processing", "connectome_weights.csv"
    )


def get_subject_tract_lengths_path(subject, type):
    return os.path.join(
        dataset_path, type, subject, "DWI_processing", "connectome_lengths.csv"
    )


def get_region_labels_path(atls="HCP_MMP1"):
    if atls == "HCP_MMP1":
        return os.path.join(data_path, "HC", "region_labels_HCP.txt")
    else:
        print("Atlas not supported")


def get_group_average_efield_over_atlas_path(
    group, atlas="HCP_MMP1", efield_type="magnE"
):
    return os.path.join(
        get_analysis_data_path(), f"{efield_type}_avg_{group}_{atlas}.json"
    )
