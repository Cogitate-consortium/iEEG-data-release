import xnat
import sys
import os
import shutil
import os.path as op
from environment_variables import bids_root, xnat_host, xnat_project


def move_dir_contents(source_dir, destination_dir):
    try:
        if not op.exists(destination_dir):
            os.makedirs(destination_dir)

        for file_name in os.listdir(source_dir):
            file_path = op.join(source_dir, file_name)
            if op.isfile(file_path):
                shutil.move(file_path, destination_dir)
            elif op.isdir(op.join(source_dir, file_name)):
                shutil.copytree(op.join(source_dir, file_name),
                                op.join(destination_dir, file_name),
                                dirs_exist_ok=True)

    except FileNotFoundError as e:
        print("Error: " + str(e))
    except Exception as e:
        print("An error occurred: " + str(e))


def xnat_download(subjects_to_download, myproject, overwrite=True):

    # Handle inputs:
    if not isinstance(myproject, str):
        print('project not provided or wrong type provided, must be a string.')
        sys.exit(1)
    if isinstance(subjects_to_download, str):
        subjects_to_download = [subjects_to_download]
    if not op.isdir(bids_root):
        os.makedirs(bids_root)

    # Check whether we actually need to download any subjects:
    if all([os.path.isdir(op.join(bids_root, subject)) 
            for subject in subjects_to_download]) and not overwrite:
        return
    else:
        user = input("Enter your XNAT user name: ")
        password = input("Enter your XNAT password: ")


    # Connect to XNAT:
    session = xnat.connect('https://%s.ae.mpg.de' % xnat_host, user=user, password=password)

    # Get the project:
    project = session.projects.get(myproject)

    # Check if it exists:
    if not project:
        print(f'Project {myproject} not found.')
        sys.exit(1)

    # --------------------------------------------------------------------------------
    # Project level files:
    if overwrite or not os.path.isfile(op.join(bids_root, "dataset_description.json")):
        # Get the directory in which xnat will download the data:
        proj_bids_root = op.join(bids_root, myproject, 'resources',
                                'bids', 'files')
        project.resources['bids'].download_dir(bids_root)
        move_dir_contents(proj_bids_root, bids_root)
        shutil.rmtree(op.join(bids_root, myproject, 'resources'))
    else:
        print(f'The project data of {myproject} are already present on your computer.')
        print(f'Set overwrite to true if you wish to overwrite them')

    # --------------------------------------------------------------------------------
    # Single subjects files:
    # Get the subjects in the project:
    subjects_in_project = [s.label for s in project.subjects.values()]

    if not all([subj in subjects_in_project for subj in subjects_to_download]):
        print(
            f'Subjects {subjects_to_download} not found in project {myproject}.')
        sys.exit(1)

    # Loop through each subject to download:
    for subject in subjects_to_download:
        print(f'Downloading subject {subject}')
        # Get the directory in which xnat will download the data:
        subj_bids_root = op.join(bids_root, subject, 'resources', 'bids', 'files')

        if overwrite or not os.path.isdir(op.join(bids_root, subject)):
            session.projects.get(myproject).subjects.get(subject).resources['bids'].download_dir(bids_root)
            # move the files to the actual location
            move_dir_contents(subj_bids_root, bids_root)

            # delete the subject folder, cleanup
            shutil.rmtree(op.join(bids_root, subject))

        else:
            print(f'The project data of subject are already present on your computer.')
            print(f'Set overwrite to true if you wish to overwrite them')


if __name__ == "__main__":
    download_subject("sub-CE107", xnat_project)

