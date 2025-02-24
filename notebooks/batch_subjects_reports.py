import os
import pandas as pd
from pathlib import Path
import papermill as pm
from nbconvert import HTMLExporter
import nbformat
import shutil
import sys
from cog_ieeg.utils import get_bids_root
from cog_ieeg.xnat_utilities import xnat_download


def copy_images(src, dst):

    if not os.path.exists(dst):
        os.makedirs(dst)

    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dst)


def subject_report_html(subject_id):
    # Get the current file directory:
    file_root = Path(__file__).resolve().parent
    # Set paths:
    input_nb = Path(file_root, 'ieeg-single-subject-report.ipynb') 
    output_nb = Path(file_root, f'output_{subject_id}.ipynb')
    report_dir = Path(file_root.parent, "subjects_reports")
    html_output = os.path.join(report_dir, f'report_{subject_id}.html')
    img_src = Path(__file__).resolve().parent

    # Ensure the reports directory exists
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Copy the images to the directory to make sure the notebook headers and footers don't break
    copy_images(img_src, os.path.join(report_dir, 'img'))
    # Execute the notebook for the given subject
    pm.execute_notebook(
        input_nb,
        output_nb,
        parameters=dict(subject=subject_id.split('-')[1])
    )

    # Convert the output notebook to HTML
    with open(output_nb) as f:
        nb = nbformat.read(f, as_version=4)
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(nb)

    with open(html_output, 'w') as f:
        f.write(body)

    # Delete the output notebook file
    os.remove(output_nb)
    print(f"Report for {subject_id} generated.")


if __name__ == "__main__":
    # subjects = pd.read_csv(Path(get_bids_root(), "participants.tsv"), sep='\t')["participant_id"].to_list()
    subjects = ["sub-CF102", "sub-CF103"]
    # Download the data if necessary:
    xnat_download([sub for sub in subjects], overwrite=False)
    for subject in subjects:
        print(subject)
        try:
            subject_report_html(subject)
        except:
            print(f"Failed for {subject}")
