'''
Author: DengRui
Date: 2024-01-10 13:15:08
LastEditors: DengRui
LastEditTime: 2024-01-20 13:23:06
FilePath: /DeepSub/trainer.py
Description:  trainer
Copyright (c) 2024 by DengRui, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tool import model as md
from tool import config as cfg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_data():
    """
    Load and preprocess the dataset.

    Args:
        None.

    Returns:
        pd.DataFrame: Preprocessed dataset containing the following columns:
            - 'uniprot_id': UniProt ID.
            - 'seq': Sequence.
            - 'f1': Feature value.
            - 'new_label': Encoded label value using LabelEncoder.
            - 'label': Original label value.
    """
    dataset = pd.read_csv(cfg.DATA_PATH)
    feature = pd.read_feather(f'{cfg.FEATURE_PATH}feature_esm2.feather')
    dataset = dataset.rename(columns={'Entry': 'uniprot_id', 'Sequence': 'seq'})
    data_df = dataset.merge(feature, on='uniprot_id', how='left')
    data_df = data_df[~data_df.f1.isnull()]
    data_df['label'] = LabelEncoder().fit_transform(data_df['label'])

    return data_df

def reshape_features(data):
    """
    Reshape input data to have 3 dimensions.

    Args:
        data (np.ndarray): Input data to be reshaped.

    Returns:
        np.ndarray: Reshaped data with shape (n_samples, 1, n_features).
    """
    return np.array(data).reshape(data.shape[0],1,-1)


if __name__ =="__main__":

    # Load and preprocess the dataset.
    dataset = load_and_preprocess_data()

    # train_data, vali_data = train_test_split(dataset, test_size=cfg.TRAIN_TEST_SPLIT_SIZE,shuffle=False)
    train_data, vali_data = train_test_split(dataset, test_size=cfg.TRAIN_TEST_SPLIT_SIZE, stratify=dataset['label'], random_state=42)
    
    X_train = reshape_features(train_data.iloc[:,3:])
    X_val = reshape_features(vali_data.iloc[:,3:])
    
    print(f'X_train shape: {X_train.shape}')
    print(f'X_val shape: {X_val.shape}')

    # Train the model.
    gru_attention_model = md.GRUWithAttentionModel(input_shape=cfg.INPUT_SHAPE, num_classes=cfg.NUM_CLASSES)
    gru_attention_model.compile_model()

    history = gru_attention_model.train(
        X_train, 
        train_data['label'],
        X_val, 
        vali_data['label'],
        batch_size=cfg.BATCH_SIZE, 
        epochs=cfg.EPOCHS
    )
    
    # Save the model.
    folder_path = cfg.MODEL_SAVE_PATH
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    gru_attention_model.save_model(f'{cfg.MODEL_SAVE_PATH}deepsub.h5')

    val_predictions = gru_attention_model.model.predict(X_val, batch_size=cfg.BATCH_SIZE)
    ground_truth_labels = vali_data['label'].values
    predicted_labels = np.argmax(val_predictions, axis=1)
    
    # Export ground truth and predicted labels
    export_data = pd.DataFrame({'GroundTruth': ground_truth_labels, 'PredictedLabels': predicted_labels})
    export_data.to_csv('groundtruth_and_labels.csv', index=False)
