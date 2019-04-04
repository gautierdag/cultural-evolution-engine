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
python main.py
```


## Cultural Evolution Engine:

The cultural evolution engine consists of two main bases classes: `BaseAgent` and `BaseCEE`. 

- The `BaseAgent` class offers a wrapper for torch models that handles loading/saving, culling, and also storing potentially relevant attributes that one might want to track such as age, accuracy, loss. 

- The `BaseCEE` class on the other hand offers a class to manage the two populations, sample from them, and cull from them. 

Together they make up the Cultural Evolution Engine. 

Additionally standard metrics with which to evaluate messages and language are also implemented in the engine. Currently Representational Similarity Analysis (RSA) and the language Entropy.
 
