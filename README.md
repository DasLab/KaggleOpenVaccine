# DeepDeg4

Code for training and inference of mRNA molecule stability models. DeepDeg4 contains non-linear machine learning models - one based on XGBoost and a more advnced neural netwoek one based on the winning solution of the Kaggle Open Vaccine challenge. 

DeepDeg4 models are advanced versions of the original DegScore model that can be found here: https://github.com/eternagame/DegScore

## Organization of the Repository

model_files: actual model files for the models used for inference
data: data used for the modeling and inference
notebooks: inference Python notebooks
scripts: inference Python scripts

## Dependencies

Install via `pip install requirements.txt` or `conda install --file requirements.txt`

*Not pip-installable:* You will need to install Arnie: a Python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. https://github.com/DasLab/arnie


## Setup

1. [Install git-lfs](https://git-lfs.github.com/) (best to do before git-cloning this KaggleOpenVaccine repo).

2. Install EternaFold, available for free noncommercial use [here](https://www.eternagame.org/about/software).

3. Git clone [Arnie](https://github.com/DasLab/arnie), follow instructions [here](https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md) to link EternaFold to it.

4. Add path to this repository as `KOV_PATH` (so that script can find path to stored model files):

```
export KOV_PATH='/path/to/KaggleOpenVaccine'
```

## Usage for one construct at a time

To run the nullrecurrent winning solution on one construct, given in `example.txt`:

```
CGC
```

Run

```
python scripts/OV_inference/OV_inference_2.py -i example.txt -o predict.txt
```

This write a text file of output predictions to `predict.txt`:

```
2.1289976365, 2.650808962, 2.1869660805000004
```


## Individual Kaggle Solutions

This code is based on the winning solution for the Open Vaccine Kaggle Competition Challenge. The competition can be found here:

https://www.kaggle.com/c/stanford-covid-vaccine/overview

This code is also the supplementary material for the Kaggle Competition Solution Paper. The individual Kaggle writeups for the top solutions that have been featured in that paper can be found in the following table:


| Team Name                       |  Team Memebers  | Rank  | Link to the solution                                            |
|---------------------------------|-----------------|-------|-----------------------------------------------------------------|
|Jiayang Gao                      | Jiayang Gao     |   1   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189620|
|                                 |                 |       |                                                                 |
|Kazuki ** 2                      |Kazuki Onodera   |   2   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709| 
|                                 |Kazuki Fujikawa  |       |                                                                 |
|                                 |                 |       |                                                                 |
|Striderl                         |Hanfei Mao       |   3   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189574|
|                                 |                 |       |                                                                 |
|FromTheWheel & Dyed & StoneShop  |Gilles Vandewiele|   4   |https://www.kaggle.com/group16/covid-19-mrna-4th-place-solution  |
|                                 |Michele Tinti    |       |                                                                 |
|                                 |Bram Steenwinckel|       |                                                                 |
|                                 |                 |       |                                                                 |
|tito                             |Takuya Ito       |   5   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189691|
|                                 |                 |       |                                                                 |
|nyanp                            |Taiga Noumi      |   6   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189241|
|                                 |                 |       |                                                                 |
|One architecture                 |Shujun He        |   7   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189564|
|                                 |                 |       |                                                                 |
|ishikei                          |Keiichiro Ishi   |   8   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/190314|
|                                 |                 |       |                                                                 |
|Keep going to be GM              |Youhan Lee       |   9   |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189845|
|                                 |                 |       |                                                                 |
|Social Distancing Please         |Fatih Öztürk     |   11  |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189571|
|                                 |Anthony Chiu     |       |                                                                 |
|                                 |Emin Ozturk      |       |                                                                 |
|                                 |                 |       |                                                                 |
|The Machine                      |Karim Amer       |   13  |https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189585|
|                                 |Mohamed Fares    |       |                                                                 |





