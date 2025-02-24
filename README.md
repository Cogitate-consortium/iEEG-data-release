<img src="img/iEEG_data_release_header.svg" width=1400 />


This repository demonstrates how to:

- Access and download iEEG data from the Cogitate database.
- Perform preprocessing and preliminary analyses on the data.
- Use the cog_ieeg Python package, which contains many utilities created for the Cogitate consortium.

It accompanies the scientific data we have recently submitted and is intended to help researchers quickly get started with the dataset and tools.

# 1. Setup guide:

## 1.1. Register for access
To download our data, you must first register for an account on our data portal. Watch this [video](https://www.youtube.com/embed/q-VRXeE6tUw?si=gFRrO4T_DPCXpnNn) to see how, or directly register [here](https://cogitate-data.ae.mpg.de/app/template/Login.vm#!).

## 1.2. Install Dependencies & cog_ieeg Package
In a dedicated Python environment (recommended), run:
```
pip install git+https://github.com/Cogitate-consortium/iEEG-data-release.git@main#egg=cog_ieeg
```
This single command will install all necessary dependencies and the cog_ieeg package. You will then be ready to go

# 2. Download the data

The various scripts presented below will download the data automatically in your home directory, under COGITATE/bids. If you wish to change this default parameter, you can adjust it with python:

```
from cog_ieeg.utils import set_bids_root
set_bids_root("YOUR/LOCAL/PATH")
```

This step is **optional**, everything else is ready to go. You also don't need to download any data manually, has we have automated download implemented. You should only register to our database [here](https://www.arc-cogitate.com/data-release) to get your credentials. You will simply need to specify the name of the subject you would like to download, input your credentials and the data will get downloaded on your machine. 

# 3. How to use this repository:
This repository contains Jupyter notebooks, analysis pipelines, and various utility functions. Depending on your goals, here are the key entry points:

### Understanding iEEG data and the cogitate data set
- Check out the [ieeg-data-release.ipynb notebook]((https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/notebooks/ieeg-data-release.ipynb)) to see the steps involved in iEEG data analysis, including how to work with the COGITATE data format.

### Generate Single-Subject Reports
- If you have decided you would like to do something with the cogitate data, you should go check this [notebook](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/notebooks/ieeg-single-subject-report.ipynb). It can be run on each subject to get an idea of channels localization, responses observed
- You can also run the [batch_subjects_reports.py](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/notebooks/batch_subjects_reports.py) to generate HTML reports for multiple subjects at once.

### Explore or Modify the cog_ieeg Package
- Visit [cog_ieeg](https://github.com/Cogitate-consortium/iEEG-data-release/tree/main/cog_ieeg) to see the source code for the Python package. This is where various custom functions are implemented and organized.

### Adopt Our Pipelines for Your Own Data
- If you have your own data and would like to use pipelines that are similar to ours, go check the various scripts [here](https://github.com/Cogitate-consortium/iEEG-data-release/tree/main/cog_ieeg/pipelines). The preprocessing pipelines relies heavily on the bids format, which means if your data also are in bids, you should be able to use our pipelines without much tweaking. 

My personal recommendation is to always start with this [notebook](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/ieeg-single-subject-report.ipynb) as it gives a really good overview of how things work in general. 

## How to cite us:
If you use the scripts found in this repository, you can use the DOI provided by Zenodo to cite us. And here is a bibtex:

```
@article{LepauvreEtAl2024,
  title = {COGITATE-iEEG-DATA-RELEASE},
  author = {Lepauvre, Alex and Henin, Simon and Bendtz, Katarina and Sripad, Praveen and Bonacchi, Niccolò and Kreiman, Gabriel and Melloni, Lucia},
  year = {2024},
  doi = {TO_BE_UPDATED},
}
```

If you use any the data for other purpose, you should cite the scientific data paper directly:

```
@article{SeedatEtAl2024,
  author = {Alia Seedat and Alex Lepauvre and Jay Jeschke and Urszula Gorska-Klimowska and Marcelo Armendariz and Katarina Bendtz and Simon Henin and Rony Hirschhorn and Tanya Brown and Erika Jensen and David Mazumder and Stephanie Montenegro and Leyao Yu and Niccol\`{o} Bonacchi and Praveen Sripad and Fatemeh Taheriyan and Orrin Devinsky and Patricia Dugan and Werner Doyle and Adeen Flinker and Daniel Friedman and Wendell Lake and Michael Pitts and Liad Mudrik and Melanie Boly and Sasha Devore and Gabriel Kreiman and Lucia Melloni},
  title = {Open multi-center iEEG dataset with task probing conscious visual perception},
  journal = {TO_BE_UPDATED},
  year = {TO_BE_UPDATED},
  volume = {TO_BE_UPDATED},
  number = {TO_BE_UPDATED},
  pages = {TO_BE_UPDATED},
  doi = {TO_BE_UPDATED},
}
```
## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/AlexLepauvre"><img src="https://avatars.githubusercontent.com/AlexLepauvre?v=4" width="100px;" alt=""/><br /><sub><b>Alex Lepauvre</b></sub></a><br /><a href="#code-AlexLepauvre" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/pravsripad"><img src="https://avatars.githubusercontent.com/pravsripad?v=4" width="100px;" alt=""/><br /><sub><b>Praveen Sripad</b></sub></a><br /><a href="#doc-pravsripad" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/simonhenin"><img src="https://avatars.githubusercontent.com/simonhenin?v=4" width="100px;" alt=""/><br /     ><sub><b>Simon Henin</b></sub></a><br /><a href="#doc-simonhenin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/KatBendtz"><img src="https://avatars.githubusercontent.com/KatBendtz?v=4" width="100px;" alt=""/><br /     ><sub><b>Katarina Bendtz</b></sub></a><br /><a href="#doc-KatBendtz" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/qian-chu"><img src="https://avatars.githubusercontent.com/qian-chu?v=4" width="100px;" alt=""/><br /     ><sub><b>Qian Chu</b></sub></a><br /><a href="#doc-qian-chu" title="Comments">💬</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


# Contributing to iEEG-data-release

Thank you for considering contributing to our project! Here are a few guidelines to help you get started:

## Pull Requests

1. **File an issue**: Before making any changes, please open an [issue](https://github.com/Cogitate-consortium/iEEG-data-release/issues) to discuss the proposed changes. This helps us keep track of what needs to be done and ensures that your efforts align with the project's goals.
2. **Fork the repository** and create your branch from `main`.
3. **Commit your changes** and push your branch to your fork.
4. **Submit a pull request** and request a review.

### Branch Protection

Our `main` branch is protected. All changes must be made via pull requests and require approval before being merged. This helps us maintain the quality and stability of the codebase.

## Code Reviews

All pull requests will be reviewed by a maintainer. Please be patient if we take a bit longer than expected. 

## Questions?

If you have any questions, feel free to open an issue or contact the maintainers (alex.lepauvre@).

## Additional notes
The folder WU_coregistration contains scripts that were used on subjects from Wisconsin university to co-register the trigger signal with the recorded iEEG signals (as both were recorded on different systems). This procedure was applied locally on the site before sharing the raw data, and is available here only for reference purposes. It should not be applied on the data at any points.

# Acknowledgments
This notebook is brought to you by the intracranial team of the COGITATE consortium.
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
   <div style="text-align: center;">
      <a href="https://www.arc-cogitate.com/our-team" target="_blank">
         <img src="img/IEEG TEAM.png" alt="iEEG team">
      </a>
   </div>
</div>
<br />
We would like to thank all the COGITATE consortium members:
<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
   <div style="text-align: center;">
      <a href="https://www.arc-cogitate.com/our-team" target="_blank">
         <img src="img/IEEG DP Authors.png" alt="COGITATE team">
      </a>
   </div>
</div>
<img style="float: right;" src="img/templeton_logo.png" width=200;>
<br />
<br />

This research was supported by Templeton World Charity Foundation ([TWCF0389](https://doi.org/10.54224/20389)) and the Max Planck Society. The opinions expressed in this publication are those of the authors and do not necessarily reflect the views of TWCF.