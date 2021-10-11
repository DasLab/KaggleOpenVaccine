# Kaggle OpenVaccine Data

## Kaggle_RYOS_data

This directory contains training set and test sets in .csv and in .json form.

`Kaggle_RYOS_trainset_prediction_output_Sep2021.txt` contains predictions from the Nullrecurrent code in this repository.

Model MCRMSEs were evaluated by uploading submissions to the Kaggle competition website at https://www.kaggle.com/c/stanford-covid-vaccine.

## mRNA_233x_data

This directory contains original data and scripts to reproduce model analysis from manuscript.

Because all the original formats are slightly different, the `reformat_*.py` scripts read in the original formats and reformats them in two forms for each prediction: "FULL" and "PCR" in the directory `formatted_predictions`.

"FULL" is per-nucleotide predictions for all the nucleotides. "PCR" has had the regions outside the RT-PCR sequencing set to NaN.

`python collate_predictions.py` reads in all the data and outputs `all_predictions_233x.csv`

`RegenerateFigure5.ipynb` reproduces the final scatterplot comparisons.

