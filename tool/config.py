
'''
Author: DengRui
Date: 2024-01-20 13:15:08
LastEditors: DengRui
LastEditTime: 2024-01-20 13:23:06
FilePath: /DeepSub/trainer.ipynb
Description:  trainer parameter
Copyright (c) 2024 by DengRui, All Rights Reserved. 
'''

DATA_PATH = './DATA/Dataset_0724.csv'
FEATURE_PATH = './featurebank/feature_esm2.feather'
FEATURE_PATH_CHECKPOINT = './featurebank/feature_esm2.feather'

MODEL_SAVE_PATH = './model/deepsub_20240120.h5'
INPUT_SHAPE = (1, 1280)
NUM_CLASSES = 10
BATCH_SIZE = 1024
EPOCHS = 200
TRAIN_TEST_SPLIT_SIZE = 0.2



