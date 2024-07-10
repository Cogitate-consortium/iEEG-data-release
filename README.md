<img src="img/iEEG_data_release_header.svg" width=1400 />

This repository contains scripts showcasing how to download iEEG data from the data base, perform preprocessing and preliminary analyses. It accompanies the scientific data paper published in nature: 

## Setup guide:
First, you should install the environment and activate it:
```
conda env create -f environment.yml
conda activate cog_ieeg_release
```

Then, you should adjust the path configuration in the environment_variable.py by setting the bids root to the local path where you would like to store your data:
```
bids_root = "YOUR/LOCAL/PATH"
```

That's the only setup you need to do. 

You don't need to download any data manually, has we have automated download implemented. You should only register to our database [here](https://www.arc-cogitate.com/data-release) to get your credentials. You will simply need to specify the name of the subject you would like to download, input your credentials and the data will get downloaded on your machine. 


## How to use this repository:
This repository contains jupyter notebooks, analysis pipelines and many different functions. Depending on your interest, here are the different places to go to first:

- If you are interested in iEEG data analysis and you would like to see what are the different steps involved OR if you are interested in the COGITATE data and would like to get a sense of how to work with them, go check this [notebook](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/ieeg-single-subject-report.ipynb)
- If you have decided you would like to do something with the cogitate data, you should go check this [notebook](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/ieeg-single-subject-report.ipynb). It can be run on each subject to get an idea of channels localization, responses observed... You can also used this [script](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/batch_subjects_reports.py) to specify a set of subjects to generate reports for each of them as HTML
- If you are a curious about the implementation of different functions we are using, you can go have a look at this [file](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/HelperFunctions.py). It contains all the single functions we are using (high gamma computations and so on)
- If you have your own data and would like to use pipelines that are similar to ours, go check the various scripts [here](https://github.com/Cogitate-consortium/iEEG-data-release/tree/main/pipelines). The preprocessing pipelines relies heavily on the bids format, which means if your data also are in bids, you should be able to use our pipelines without much tweaking. 

My personal recommendation is to always start with this [notebook](https://github.com/Cogitate-consortium/iEEG-data-release/blob/main/ieeg-single-subject-report.ipynb) as it gives a really good overview of how things work in general. 

## How to get in touch:
If you have any questions or any ideas to improve this repository, feel free to let us know by filing an issue in [github](https://github.com/Cogitate-consortium/iEEG-data-release/issues). We will try to get back to you as soon as we can.

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

# Contributors:
<!-- Copy-paste in your Readme.md file -->

<a href = "https://github.com/Tanu-N-Prabhu/Python/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo = Cogitate-consortium/iEEG-data-release"/>
</a>

Made with [contributors-img](https://contrib.rocks).

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