import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sys,os
from tool.att import Attention
from keras.models import load_model
import esm
import torch
from tqdm import tqdm
import requests
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = pd.read_csv("output/download_seq.csv")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(device)

# Esm2 embedding
def get_rep_seq(sequences):

    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
       
    np_list = []

    for i, ten in enumerate(sequence_representations):
        ten=ten.cpu().detach().numpy()
        np_list.append(ten)
    res = pd.DataFrame(np_list)
    res.columns = ['f'+str(i) for i in range (0,res.shape[1])]
    return res

df_data = list(zip(dataset.uniprot_id.index,dataset.seq))

# Run in batches
stride =2
num_iterations = len(df_data) // stride
if len(df_data) % stride != 0:
    num_iterations += 1
    
# Embedding
all_results = pd.DataFrame()

for i in tqdm(range(num_iterations)):
    
    start = i * stride
    end = start + stride

    current_data = df_data[start:end]

    rep33 = get_rep_seq(sequences=current_data)
    rep33['uniprot_id'] = dataset[start:end].uniprot_id.tolist()
    cols = list(rep33.columns)
    cols = [cols[-1]] + cols[:-1]
    rep33 = rep33[cols]
    all_results = pd.concat([all_results, rep33], ignore_index=True)

all_results.to_feather("output/seq_esm.feather")

# Deepsub
model = load_model("./model/deepsub_new.h5",custom_objects={"Attention": Attention},compile=False)
predicted = model.predict(np.array(all_results.iloc[:,1:]).reshape(all_results.shape[0],1,-1))
predicted_labels = np.argmax(predicted, axis=1)
label_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12}
y_test_transformed = [label_map[x] for x in predicted_labels]
np.save('output/openset_label.npy', np.array(y_test_transformed))
print("These are the predicted labels:")
print(y_test_transformed)