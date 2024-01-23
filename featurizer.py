'''
Author: DengRui
Date: 2024-1-20 13:15:08
LastEditors: DengRui
LastEditTime: 2024-1-20 13:23:06
FilePath: /DeepSub/embedding/esm_embedding_esm2.py
Description:  using esm2 embedding seqs
Copyright (c) 2024 by DengRui, All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import esm
import torch
import os
from tqdm import tqdm
from tool import config as cfg

# Set gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Load data
def load_data(path=cfg.DATA_PATH):
    """
    Load the dataset from the specified path and return a list, where each element is a tuple containing the uniprot_id and seq.

    Args:
        path (str): The path to the dataset, defaulting to cfg.DATA_PATH.

    Returns:
        List[Tuple[int, str]]: A list of tuples containing the uniprot_id and seq.            
    
    """  
    dataset = pd.read_csv(path)
    dataset = dataset.sample(1000)
    dataset = dataset.rename(columns={'Entry':'uniprot_id','Sequence':'seq'})
    df_data = list(zip(dataset.uniprot_id.index,dataset.seq))
    return df_data,dataset

# Set model
def set_model():
    """
    Set the model to be used for embedding.
    
    Args:
        None
        
    Returns:
        esm.pretrained.ESM: The pre-trained ESM model.
        
    """
    esm.pretrained.esm2_t33_650M_UR50D()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    
    return model,batch_converter,alphabet

# Sequences embedding
def get_rep_seq(sequences,model,batch_converter,alphabet):
    """
    Embedding sequences using the given model.
        
    Args:
        sequences (list): The list of sequences to be embedded.
        model (esm.pretrained.ESM): The pre-trained ESM model.
        batch_converter (esm.pretrained.BatchConverter): The batch converter for the given model.
        alphabet (esm.pretrained.Alphabet): The alphabet for the given model.
        
    Returns:
            pd.DataFrame: The embedding results for the given sequences.
         
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        # Average on the protein length, to obtain a single vector per fasta
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
       
    np_list = []
    # Detach the tensors to obtain a numpy array
    for i, ten in enumerate(sequence_representations):
        ten=ten.cpu().detach().numpy()
        np_list.append(ten)
    res = pd.DataFrame(np_list)
    res.columns = ['f'+str(i) for i in range (0,res.shape[1])]
    return res

def save_feature(folder_path_feature,res):
    """
    Save the embedding results to a feather file.
    
    Args:
        folder_path_feature (str): The path to the folder where the feather file will be saved.
        res (pd.DataFrame): The embedding results for the given sequences.
        
    Returns: None
    
    """
 
    res.to_feather(f'{folder_path_feature}feature_esm2.feather')



def main():
    
    """ Perform embedding on the given dataset and process it in batches.

    Args: 
        df_data (pd.DataFrame): The dataset to be processed, containing two columns: 
        sequences and uniprot_ids. stride (int, optional): The step size for batch processing. 
        Defaults to 2. num_iterations (int, optional): The number of iterations for processing. Defaults to None.

    Returns: None 
    """
    
    # Load data
    df_data,dataset = load_data()
    
    # Set model
    model,batch_converter,alphabet = set_model()
    
    # check dir
    folder_path_feature = cfg.FEATURE_PATH
    
    if not os.path.exists(folder_path_feature):
        os.makedirs(folder_path_feature)
    
    # Embedding
    stride =2
    num_iterations = len(df_data) // stride
    if len(df_data) % stride != 0:
        num_iterations += 1
        
    all_results = pd.DataFrame()

    for i in tqdm(range(num_iterations)):
        start = i * stride
        end = start + stride

        current_data = df_data[start:end]

        rep33 = get_rep_seq(current_data,model,batch_converter,alphabet)
        rep33['uniprot_id'] = dataset[start:end].uniprot_id.tolist()
        cols = list(rep33.columns)
        cols = [cols[-1]] + cols[:-1]
        rep33 = rep33[cols]
        all_results = pd.concat([all_results, rep33], ignore_index=True)
        if end%500 == 0:
            all_results.to_feather(f'{folder_path_feature}feature_esm2_checkpoint.feather')
            
    # save feature
    save_feature(folder_path_feature,all_results)
    
if __name__ == '__main__':
    main()