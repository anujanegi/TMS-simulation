import os

data_path = "/home/anujanegi/tj/TMS-simulation/data/ADNI"
subjects = {"AD": ["011_S_4547"], "MCI": ["002_S_4654"], "HC": ["002_S_0413"]}


def get_m2m_path(subject, type):
    return os.path.join(data_path, type, subject, f"m2m_{subject[-4:]}")


# get efield head mesh path
def get_efield_head_mesh_path(subject, type):
    return os.path.join(
        data_path,
        type,
        subject,
        "TMS_efield",
        "subject_overlays",
        f"{subject[-4:]}_TMS_1-0001_MagStim_D70_scalar_central.msh",
    )


def get_TMS_efield_path(subject, type):
    return os.path.join(data_path, type, subject, "TMS_efield")


def get_subject_path(subject, type):
    return os.path.join(data_path, type, subject)
