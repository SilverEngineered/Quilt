# Quilt


## Background
This repository is the companion code associated with the paper "Quilt" and contains all necessary code required to run the ensemble based Quilt model.  The datasets are not included but can be downloaded from the offical distributions of MNSIT, Fashion MNIST, and Cifar-10.\
To Do: Add automatic bash scripts for downloading data and setting up repository.

## Running Quilt
1.) Download the Dataset and place in the appropriate dataset directory.\
2.) Create and IBM Qiskit account and place token in dev_config0.json\
3.) Download all necessary libraries\
3.) Run the training procedure such as ```python main.py --model="Final_Divide" --run_type="train"```\
4.) Run an inference procedure like ```python3 main.py --model="Final_Divide" --run_type="test"```
