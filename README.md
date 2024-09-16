
This is the code for our proposed method, SymbolicGPT. We tried to keep the implementation as simple and clean as possible to make sure it's understandable and easy to reuse. Please feel free to add features and submit a pull-request.

# Results/Models/Datasets
- Download via [link](https://www.dropbox.com/sh/yq03daorth1h4kj/AADolbgySCjOO18qGoP5Abqfa?dl=0)
- dataset size reduction can be done manually 


# Abstract:
Symbolic Regression remains an NP-Hard problem, with extensive research focusing on AI models for this task. We propose applying k-fold cross-validation to a transformer-based symbolic regression model trained on a significantly reduced dataset (15,000 data points, down from 500,000). This technique partitions the training data into multiple subsets (folds), iteratively training on some while validating on others. Our aim is to provide an unbiased estimate of model generalization and mitigate overfitting issues associated with smaller datasets. Results show that this process improves the model's output consistency and generalization by a relative a relative improvement o=in validation loss of 53.31%. Potentially enabling more efficient and accessible symbolic regression in resource-constrained environments.

# Setup the environment

you can install the following packages:
```bash
pip install numpy
pip install torch
pip install matplotlib
pip install scipy
pip install tqdm
pip install scikit-learn
pip install tabulate
```

# Dataset Generation

You can skip this step if you already downloaded the datasets using this [link](https://www.dropbox.com/sh/yq03daorth1h4kj/AADolbgySCjOO18qGoP5Abqfa?dl=0).

## How to generate the training data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py
- Change the seed to 2021 in the dataset.py 
- Change the seed to 2021 in the generateData.py 
- Generate the data using the following command:
```bash
$ python dataset.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Train/\*.json)

## Generate the validation data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py except the numSamples
- Make sure that the numSamples = 1000 // len(numVars) in the datasetTest.py 
- Change the seed to 2022 in the datasetTest.py 
- Change the seed to 2022 in the generateData.py 
- Generate the data using the following command:
```bash
$ python datasetTest.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Val/\*.json)

## Generate the test data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py except the numSamples
- Make sure that the numSamples = 1000 // len(numVars) in the datasetTest.py 
- Change the seed to 2023 in the datasetTest.py 
- Change the seed to 2023 in the generateData.py 
- Generate the data using the following command:
```bash
$ python datasetTest.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Test/\*.json)

# Train/Test the model

It's easy to train a new model and reproduce the results.
Use main.ipynb to download the requirements and run the script once the datasets and parameters are loaded

## Configure the parameters

Follow each dataset config file and change the corresponding parameters (numVars, numPoints etc.) in the symbolicGPT.py script. 

## Reproduce the experiments



### Use this in symbolicGPT.py to reproduce the results for 1 Variable Model
```python
numEpochs = 20 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=[30,31] # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=1 # the dimension of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
testBlockSize = 400
batchSize = 128 # batch size of training data
target = 'Skeleton' #'Skeleton' #'EQ'
const_range = [-2.1, 2.1] # constant range to generate during training only if target is Skeleton
decimals = 8 # decimals of the points only if target is Skeleton
trainRange = [-3.0,3.0] # support range to generate during training only if target is Skeleton
dataDir = os.path.join(os.getcwd(), 'datasets')
dataTestFolder = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points/Test'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
```

## Run the script to train and test the model
```bash
python symbolicGPT.py
```

# System Spec:
3 NVIDIA RTX 6000 ada Gpus
144 gb VRAM + 186 gb RAM
v48 CPU

# Citation:
```
@inproceedings{
    SymbolicGPT2021,
    title={SymbolicGPT: A Generative Transformer Model for Symbolic Regression},
    author={Mojtaba Valipour, Maysum Panju, Bowen You, Ali Ghodsi},
    booktitle={Preprint Arxiv},
    year={2021},
    url={https://arxiv.org/abs/2106.14131},
    note={Under Review}
}
```

# REFERENCES:
- https://github.com/mojivalipour/symbolicgpt
- https://github.com/agermanidis/OpenGPT-2
- https://github.com/imcaspar/gpt2-ml
- https://huggingface.co/blog/how-to-train
- https://github.com/bhargaviparanjape/clickbait
- https://github.com/hpandana/gradient-accumulation-tf-estimator
- https://github.com/karpathy/minGPT
- https://github.com/charlesq34/pointnet
- https://github.com/volpato30/PointNovo
- https://github.com/brencej/ProGED
- https://github.com/brendenpetersen/deep-symbolic-optimization

# License:
MIT
