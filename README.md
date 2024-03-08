# Quilt


## Background
This repository is the companion code associated with the paper "Quilt" and contains all necessary code required to run the ensemble based Quilt model.  The datasets are not included but can be downloaded from the offical distributions of MNSIT, Fashion MNIST, and Cifar-10.\
To Do: Add automatic bash scripts for downloading data and setting up repository.
Requires: Python 3.10
## Running Quilt
1.) Download the Dataset and place in the appropriate dataset directory.\
2.) Pre-process the dataset using ```python fashion_mnist.py``` to split the data for ensembles
3.) Create and IBM Qiskit account and place token in dev_config0.json\
4.) Download all necessary libraries\
5.) Run the training procedure such as ```python main.py --model="Final_Divide" --run_type="train"```\
6.) Run an inference procedure like ```python3 main.py --model="Final_Divide" --run_type="test"```

### Validation
1) Run ```python main.py --model="Final_Divide" --run_type="train"``` to train data using Quilt
2) Run ```python main.py --model="Final_Divide" --run_type="test"``` to test and infer the bad bits with actual accuracy
3) Run ```python main.py --model="Ensemble" --run_type="train"``` and ```python main.py --model="Ensemble" --run_type="test"``` to get accuracy of Ensemble implementation
4) Update dataset name in nn/main.py and Run ```python nn/main.py ``` to check accuracy of OnevsOne
5) Compare the values across different classifications datasets and compare results

