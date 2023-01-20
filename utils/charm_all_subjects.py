import sys
sys.path.append('../')
import config
import os

def main():
    # run command 'charm' on all subjects in confif.subjects
    for subject_type in config.subjects:
        for subject in config.subjects[subject_type]:
            print(f"Running charm on {subject}")
            os.chdir(config.get_subject_path(subject, subject_type))
            os.system(f"charm {subject[-4:]} T1w/T1w.nii.gz T2w/T2w.nii.gz --forceqform --forcerun")
            print(f"Done running charm on {subject}")

if __name__ == "__main__":
    main()  
