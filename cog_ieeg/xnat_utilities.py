import xnat
import sys
import os
import getpass
import shutil
import os.path as op
from cog_ieeg.utils import get_bids_root, get_xnat_host, get_xnat_project


def move_dir_contents(source_dir, destination_dir):
    """
    Move contents from the source directory to the destination directory.

    Parameters
    ----------
    source_dir : str
        Path to the source directory.
    destination_dir : str
        Path to the destination directory.
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


def xnat_download(subjects_to_download, to=None, overwrite=False):
    """
    Download subject data from an XNAT project.

    Parameters
    ----------
    subjects_to_download : str or list of str
        List of subject IDs to download. If a single subject ID is provided as a string, it will be converted to a list.
    to : str, optional
        Path to the directory where the data will be downloaded. If not provided, the BIDS root directory is used.
    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.

    Raises
    ------
    SystemExit
        If the XNAT project is not found or the subjects are not found in the project.
    """
    if to is None:
        to = get_bids_root()
        
    if not isinstance(get_xnat_project(), str):
        print('Project not provided or wrong type provided, must be a string.')
        sys.exit(1)
    
    if isinstance(subjects_to_download, str):
        subjects_to_download = [subjects_to_download]

    if not op.isdir(to):
        os.makedirs(to)

    if subjects_to_download is not None and all([os.path.isdir(op.join(to, subject)) for subject in subjects_to_download]) and not overwrite:
        print(f'The subjects are already present on your computer.')
        print(f'Set overwrite to true if you wish to overwrite them.')
        return

    # Loop to prompt for username and password until successful login
    while True:
        user = input("Enter your XNAT username: ")
        password = getpass.getpass("Enter your XNAT password: ")
        try:
            session = xnat.connect(f'https://{get_xnat_host()}.ae.mpg.de', user=user, password=password)
            break  # Exit the loop if the connection is successful
        except xnat.exceptions.XNATError as e:
            print("Login failed: " + str(e))
            print("Please try again.")

    project = session.projects.get(get_xnat_project())
    if not project:
        print(f'Project {get_xnat_project()} not found.')
        sys.exit(1)

    # --------------------------------------------------------------------------------
    # Project level data:
    if overwrite or not os.path.isfile(op.join(to, "dataset_description.json")):
        proj_bids_root = op.join(to, get_xnat_project(), 'resources', 'bids', 'files')
        project.resources['bids'].download_dir(to)
        move_dir_contents(proj_bids_root, to)
        shutil.rmtree(op.join(to, get_xnat_project()))
    else:
        print(f'The project data of {get_xnat_project()} are already present on your computer.')
        print(f'Set overwrite to true if you wish to overwrite them.')

    # --------------------------------------------------------------------------------
    # Single subject:
    subjects_in_project = [s.label for s in project.subjects.values()]
    if subjects_to_download is None:
        subjects_to_download = subjects_in_project

    if not all([subj in subjects_in_project for subj in subjects_to_download]):
        print(f'Subjects {subjects_to_download} not found in project {get_xnat_project()}.')
        sys.exit(1)
    # Download single subjects data:
    for subject in subjects_to_download:
        print(f'Downloading subject {subject}')
        subj_bids_root = op.join(to, subject, 'resources', 'bids', 'files')

        if overwrite or not os.path.isdir(op.join(to, subject)):
            session.projects.get(get_xnat_project()).subjects.get(subject).resources['bids'].download_dir(to)
            move_dir_contents(subj_bids_root, to)
            shutil.rmtree(op.join(to, subject, 'resources'))
        else:
            print(f'The project data of subject {subject} are already present on your computer.')
            print(f'Set overwrite to true if you wish to overwrite them.')

if __name__ == "__main__":
    xnat_download("sub-CF102")
