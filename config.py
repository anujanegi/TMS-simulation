import os

data_path = "/home/anujanegi/tj/TMS-simulation/data/ADNI"
subjects = {
    "AD": ["011_S_4547", "036_S_4715", "041_S_4974", "114_S_6039", "168_S_6142"],
    "MCI": ["002_S_1155", "002_S_4229", "002_S_4654", "002_S_1261", "003_S_1122"],
    "HC": ["002_S_0413", "002_S_5178", "002_S_4213", "002_S_1280", "002_S_4799"],
}


def get_m2m_path(subject, type):
    return os.path.join(data_path, type, subject, f"m2m_{subject[-4:]}")


# get efield head mesh path
def get_efield_head_mesh_path(subject, type):
    # Read the simulation result mapped to the gray matter surface
    return os.path.join(
        data_path,
        type,
        subject,
        "TMS_efield",
        "subject_overlays",
        f"{subject[-4:]}_TMS_1-0001_MagStim_D70_scalar_central.msh",
    )


# get efield transformed to fsavg pace head mesh path
def get_efield_fsavg_overlay_mesh_path(subject, type):
    return os.path.join(
        data_path,
        type,
        subject,
        "TMS_efield",
        "fsavg_overlays",
        f"{subject[-4:]}_TMS_1-0001_MagStim_D70_scalar_fsavg.msh",
    )


def get_TMS_efield_path(subject, type):
    return os.path.join(data_path, type, subject, "TMS_efield")


def get_subject_path(subject, type):
    return os.path.join(data_path, type, subject)


def get_analysis_path():
    return os.path.join(data_path, "analysis")
