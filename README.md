# cultural-evolution-engine
This is a first take at a task and model agnostic CEE

## Installation Instructions 
To run, first create a python environment matching the `environment.yml` specs. You can do so by running:

```
conda env create -f environment.yml
```

Then do `conda activate cee` to activate the environment.

## Replicating experiments: 

```
# 2 agents solo setting
python baseline.py

# Cultural Evolution setting
python cee.py
```
