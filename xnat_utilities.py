import xnat
import sys
import os
import shutil
import os.path as op
from environment_variables import bids_root, xnat_host, xnat_project

def move_dir_contents(source_dir, destination_dir):
    """
    Move contents from source directory to destination directory.

    :param source_dir: Path to the source directory.
    :param destination_dir: Path to the destination directory.
    """
    try:
        if not op.exists(destination_dir):
            os.makedirs(destination_dir)

        for file_name in os.listdir(source_dir):
            file_path = op.join(source_dir, file_name)
            if op.isfile(file_path):
                shutil.move(file_path, destination_dir)
            elif op.isdir(file_path):
                shutil.copytree(file_path, op.join(destination_dir, file_name), dirs_exist_ok=True)

    except FileNotFoundError as e:
        print("Error: " + str(e))
    except Exception as e:
        print("An error occurred: " + str(e))

def xnat_download(subjects_to_download, overwrite=False):
    """
    Download subjects data from XNAT project.

    :param subjects_to_download: List of subjects to download.
    :param overwrite: Flag to overwrite existing data.
    """
    if not isinstance(xnat_project, str):
        print('project not provided or wrong type provided, must be a string.')
        sys.exit(1)
    
    if isinstance(subjects_to_download, str):
        subjects_to_download = [subjects_to_download]

    if not op.isdir(bids_root):
        os.makedirs(bids_root)

    if all([os.path.isdir(op.join(bids_root, subject)) for subject in subjects_to_download]) and not overwrite:
        print(f'The subjects are already present on your computer.')
        print(f'Set overwrite to true if you wish to overwrite them.')
        return

    # Loop to prompt for username and password until successful login
    while True:
        user = input("Enter your XNAT user name: ")
        password = input("Enter your XNAT password: ")
        
        try:
            session = xnat.connect(f'https://{xnat_host}.ae.mpg.de', user=user, password=password)
            break  # Exit the loop if the connection is successful
        except xnat.exceptions.XNATError as e:
            print("Login failed: " + str(e))
            print("Please try again.")

    project = session.projects.get(xnat_project)
    if not project:
        print(f'Project {xnat_project} not found.')
        sys.exit(1)

    # --------------------------------------------------------------------------------
    # Project level data:
    if overwrite or not os.path.isfile(op.join(bids_root, "dataset_description.json")):
        proj_bids_root = op.join(bids_root, xnat_project, 'resources', 'bids', 'files')
        project.resources['bids'].download_dir(bids_root)
        move_dir_contents(proj_bids_root, bids_root)
        shutil.rmtree(op.join(bids_root, xnat_project, 'resources'))
    else:
        print(f'The project data of {xnat_project} are already present on your computer.')
        print(f'Set overwrite to true if you wish to overwrite them.')

    # --------------------------------------------------------------------------------
    # Single subject:
    subjects_in_project = [s.label for s in project.subjects.values()]
    if subjects_to_download is None:
        subjects_to_download = subjects_in_project

    if not all([subj in subjects_in_project for subj in subjects_to_download]):
        print(f'Subjects {subjects_to_download} not found in project {xnat_project}.')
        sys.exit(1)
    # Download single subjects data:
    for subject in subjects_to_download:
        print(f'Downloading subject {subject}')
        subj_bids_root = op.join(bids_root, subject, 'resources', 'bids', 'files')

        if overwrite or not os.path.isdir(op.join(bids_root, subject)):
            session.projects.get(xnat_project).subjects.get(subject).resources['bids'].download_dir(bids_root)
            move_dir_contents(subj_bids_root, bids_root)
            shutil.rmtree(op.join(bids_root, subject))
        else:
            print(f'The project data of subject {subject} are already present on your computer.')
            print(f'Set overwrite to true if you wish to overwrite them.')

if __name__ == "__main__":
    xnat_download("sub-CE107")
