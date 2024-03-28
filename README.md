# DeepSub

## Introduction
DeepSub is a tool designed to predict the number of subunits in a protein sequence for homo-oligomers.

## Installation
``` shell
$ git clone  https://github.com/tibbdc/DeepSub.git
```

``` shell
$ cd DeepSub 
```
``` shell
$ conda create -n deepsub python=3.9 
```
``` shell
$ conda activate deepsub 
```
``` shell
$ pip install -r requirements.txt
```

## Notebooks

1. **01_GetData.ipynb**
   - Obtaining and processing data sets .

2. **02_SeqIdentity.ipynb**
   - Sequence Identity Comparison Result.

3. **03_DeepSub.ipynb**
   - DeepSub model and cross-validation results.

5. **04_Queen.ipynb**
   - Queen model for model comparison.

6. **05_OpenSet.ipynb**
   - OpenSet Dataset Evaluation.

## Scripts

- **featurizer.py**
  - Sequence features are extracted before model training.

- **trainer.py**
  - Single training function.


## Notice
   
We have successfully trained the model, which is now stored at **DeepSub/model/deepsub.h5**. You can simply execute the **test.ipynb** notebook to start making predictions. Should you wish to retrain the model with your custom dataset, please refer to the instructions in the "Usage" section and adjust accordingly.





