import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tool import model as md
from tool import config as cfg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from tool.att import Attention
def preprocess_data(dataset):
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


def run(dataset):
    # Load and preprocess the dataset.
    dataset = preprocess_data(dataset)
    X_val = reshape_features(dataset.iloc[:,3:])
    loaded_model = load_model("./model/deepsub.h5",custom_objects={"Attention": Attention},compile=False)
    predicted = loaded_model.predict(X_val)
    predicted_labels = np.argmax(predicted, axis=1)
    return dataset.label.values, predicted_labels