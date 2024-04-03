import os
import papermill as pm
from nbconvert import HTMLExporter
import nbformat
import shutil


def copy_images(src, dst):
    src = 'img'  # Source directory
    dst = 'subjects_reports/img'  # Destination directory

    if not os.path.exists(dst):
        os.makedirs(dst)

    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dst)


def subject_report_html(subject_id):
    input_nb = 'ieeg-single-subject-report.ipynb'  # Replace with your notebook filename
    output_nb = f'output_{subject_id}.ipynb'
    report_dir = 'subjects_reports'
    html_output = os.path.join(report_dir, f'report_{subject_id}.html')

    # Ensure the reports directory exists
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Copy the images to the directory to make sure the notebook headers and footers don't break
    copy_images('img', os.path.join(report_dir, 'img'))
    # Execute the notebook for the given subject
    pm.execute_notebook(
        input_nb,
        output_nb,
        parameters=dict(subject_id=subject_id)
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
    subjects = ['CF102', 'CF103']  # Add your subjects here
    for subject in subjects:
        subject_report_html(subject)
