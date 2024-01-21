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
$ conda create -n deepsub python=3.7 
```
``` shell
$ conda activate deepsub 
```
``` shell
$ pip install -r requirements.txt
```

## Usage

+  01.ipynb    load data 
+  02.ipynb  anlysis data 
+  03.ipynb  anlysis data 
+  python featurizer.py 
+  python trainer.py 
+  test.ipynb

## Notice
   
We have successfully trained the model, which is now stored at **DeepSub/model/deepsub.h5**. You can simply execute the **test.ipynb** notebook to start making predictions. Should you wish to retrain the model with your custom dataset, please refer to the instructions in the "Usage" section and adjust accordingly.





