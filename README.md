# GEM-Experiments-POC
Presentation and analysis code associated with initial GEM Proof-of-Concept experiments.

If using anything from this repository, please cite:
Fink, Alexander, Janata (2022). The Groove Enhancement Machine (GEM): A multi-person adaptive metronome to manipulate sensorimotor synchronization and subjective enjoyment.

## Contents

#### Experiment_Presentation_Code
Contains the experiment manager files associated with experiments 1-5 of the paper.
- Exp. 1: 'GEM_expMngr_1person.py'
- Exp. 2: 'GEM_expMngr_1person_hearSelf.py'
- Exp. 3: 'GEM_expMngr_4person.py'
- Exp. 4: 'GEM_expMngr_4person_hearSelf.py'
- Exp. 5: 'GEM_expMngr_4person_hearAll.py'

Note that the code to create the GEM and is in a separate repository: [https://github.com/janatalab/GEM](https://github.com/janatalab/GEM)


#### Data
Contains all data necessary to recreate the statistical analyses reported in the paper.
- 'one_all.csv' contains all one player tapping data (Exps. 1 & 2)
- 'four_all.csv' contains all four player tapping data (Exps. 3-5)
- 'four_all_wAllIndDiffs_long.csv' contains data for individual players w/in the group experiments (Exps. 3-5)


#### Analysis_Code
Contains all code necessary to recreate the statistical analyses and related figures presented in the paper.
- The power analysis conducted prior to data collection: 'GEM_powerAnalysis.py'
- General functions to facilitate analyses: 'GEM_analysis_funcs.py'
- The Jupyter notebook that steps through all tapping data analyses: 'GEM_tapping_analyses.ipynb'
- The Jupyter notebook that steps through all subjective experience data analyses: 'GEM_subjective_analyses.ipynb'
- The R notebook that steps through all individual differences analyses: 'GEM_individualDifferences_analyses.Rmd'
