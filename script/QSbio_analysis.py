#coding=utf-8
from concurrent.futures import ThreadPoolExecutor
import os
from collections import Counter
import pandas as pd
import requests
import json
import time

def get_UniProt_entry(QSbio_data_file, QSbio_pdb_info_file):
    QSbio_data = pd.read_excel(QSbio_data_file)
    QSbio_data = QSbio_data[['code', 'nsub2']]
    QSbio_data['pdb_id'] = QSbio_data['code'].apply(lambda x: x.split('_')[0])
    QSbio_data['entity_id'] = QSbio_data['code'].apply(lambda x: x.split('_')[1])

    executor = ThreadPoolExecutor(max_workers=20)
    results = list(executor.map(get_info_by_pdbid, QSbio_data['code']))
    QSbio_data['uniprot_entry'] = results
    
    while True:
        if QSbio_data['uniprot_entry'].isnull().any():
            empty_entries = QSbio_data[QSbio_data['uniprot_entry'].isnull()].index

            print("Retrying entries with empty uniprot_entry:", empty_entries)

            for index in empty_entries:
                print(index)
                code = QSbio_data['code'][index]
                print(code)
                result = get_info_by_pdbid(code)
                QSbio_data['uniprot_entry'][index] = result
        else:
            break

    QSbio_data.to_csv(QSbio_pdb_info_file, index=None)


def get_info_by_pdbid(pdbid_entityid):
    print(pdbid_entityid)
    pdbid=pdbid_entityid.split('_')[0]
    entityid=pdbid_entityid.split('_')[1]
    url = 'https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/'+ pdbid +'/' + entityid
    try:
        response = requests.get(url) 
        content = response.content.decode('utf-8')
        content_json = json.loads(content)
        
        if content_json:
            return(content_json[pdbid]['data'][0]['accession'])
        else:
            return('error pdbid')
        
    except Exception:
        pass
    
