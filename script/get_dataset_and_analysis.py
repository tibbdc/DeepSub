#coding=utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Bio import SeqIO
from tqdm import tqdm
from lxml import etree
import pandas as pd
import xml.etree.ElementTree as ET
from pandarallel import pandarallel
pandarallel.initialize()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
def analysis_xml_basic_data(xml_file):
    """
    Read the XML file and extract basic data from each entry.
    
    Args:
        xml_file (str): Path to the XML file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the extracted basic data.   
          
    """
    records = []
    
    # Analyze the XML file throught the SEQIO module
    for record in tqdm(SeqIO.parse(xml_file, "uniprot-xml")):
        uniprot_id = record.id
        seq = record.seq
        sequence_length = 0
        comment_subunit =''
        
        if 'sequence_length' in record.annotations:
                sequence_length = record.annotations['sequence_length']
        if 'organism' in record.annotations:
                organism = record.annotations['organism']

        records.append({
            'uniprot_id': uniprot_id,
            'seq': str(seq),
            'seq_len': sequence_length,
            'organism': organism
        })
        
    return pd.DataFrame(records)

def get_subunit_infomation_lxml(xml_file):
    """
    Read the XML file and extract basic data from each entry.
    
    Args:
        xml_file (str): Path to the XML file.
        store_path (str): Path to save the extracted data as a CSV file.
        namespaces (dict): Namespace dictionary.
        
    Returns:
        pd.DataFrame: A DataFrame containing the extracted analysis data.    
         
    """
    namespaces = {'ns0': 'http://uniprot.org/uniprot'}
    parsed_data = []
    context = etree.iterparse(xml_file, events=('end',), tag='{http://uniprot.org/uniprot}entry')
    for event, entry in tqdm(context):
        accession_element = entry.find('ns0:accession', namespaces)
        accession_text = accession_element.text if accession_element is not None else None
        evidence_type = []
        evidence_key = []
        evidence_elements = entry.findall('ns0:evidence', namespaces)
        
        # Analysis evidence_elements
        for evidence_element in evidence_elements:
            evidencetype = evidence_element.get('type')
            evidencekey = evidence_element.get('key')
            evidence_type.append(evidencetype)
            evidence_key.append(evidencekey)
        
        # Analysis subunit_elements
        subunit_evidence = []
        subunit_text = []
        for comment in entry.findall("ns0:comment[@type='subunit']", namespaces):
            text_element = comment.find('ns0:text', namespaces)
            if text_element is not None:
                text_content = text_element.text
                evidence = text_element.get('evidence')
                if evidence is not None:
                    subunit_evidence = [int(num) for num in evidence.split()]
                else:
                    subunit_evidence = []
                subunit_text.append(text_content)
        
        # Analysis ec_number       
        ec_numbers = [ec.text for ec in entry.findall('.//ns0:protein//ns0:ecNumber', namespaces)]
        
        # Save the data
        parsed_data.append({'uniprot_id': accession_text, 'evidence_type': evidence_type,'evidence_key':evidence_key, 'subunit_evidence': subunit_evidence,'subunit_text':subunit_text,'ec_number':ec_numbers})
        
        # Clear the entry
        entry.clear()
        
    return pd.DataFrame(parsed_data)

def map_evidence(df_row):
    """
        map the evidence_type and evidence_key to the subunit_evidence
        
        Args:
            df_row (pd.Series): A row of the DataFrame containing the extracted analysis data.
            
        Returns:
            pd.Series: A Series containing the mapped evidence type and key to the subunit_evidence.
            
    """
    key_to_type = {key: ev_type for key, ev_type in zip(df_row['evidence_key'], df_row['evidence_type'])}
    return [key_to_type.get(str(key)) for key in df_row['subunit_evidence']]

def preprocessing_data(df_basic_data,df_subunit_data):
    """
        preprocessing the data
        
        Args:
            res (pd.DataFrame): A DataFrame containing the extracted analysis data.
            
        Returns:
            pd.DataFrame: A DataFrame containing the preprocessed data.
            
    """
    df_subunit_data['mapped_evidence_types'] = df_subunit_data.parallel_apply(map_evidence, axis=1)

    # Concat the data
    res = pd.concat([df_basic_data[['uniprot_id','seq','seq_len','organism']],df_subunit_data[['ec_number','evidence_type','evidence_key','subunit_evidence','subunit_text','mapped_evidence_types']]],axis=1)
    res = res.rename(columns={
        'uniprot_id': 'Entry',
        'subunit_text': 'Subunit structure',
        'seq_len':'Length',
        'seq':'Sequence',
        'ec_number':'EC number'
    })
    
    # Change the type of data
    for col in ['Subunit structure','EC number','mapped_evidence_types']:
        if res[col].apply(lambda x: isinstance(x, list)).any():
            res[col] = res[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
            
    return res


def analysis_uniprot_xml(xml_file, store_path):
    """
        Analysis the uniprot xml file
        
        Args:
            xml_file (str): Path to the XML file containing Uniprot data.
            store_path (str): Path to save the processed data as a feather file.
        Returns:
        None
    """
    # Anlysis data
    res = pd.DataFrame()
    
    if not os.path.exists(xml_file):
        print(f'Downloading {xml_file}...')
        
    else:
        # Analysis from xml
        df_basic_data   = analysis_xml_basic_data(xml_file)
        df_subunit_data = get_subunit_infomation_lxml(xml_file)
        res = preprocessing_data(df_basic_data, df_subunit_data)
        
        # Save uniprot_sprot_data.feather
        res.to_feather(store_path)
        
    return res
        

def get_dataset_from_uniprot(uniprot_data, dataset_outfile):
    """
    Read Uniprot data from a CSV file, preprocess the data, and filter out entries based on specific keywords.
    Assign labels to each entry based on the presence of keywords in the 'Subunit structure' column.
    Save the processed data to a new CSV file.

    Args:
        uniprot_data_file (str): Path to the CSV file containing Uniprot data.
        dataset_outfile (str): Path to save the processed dataset as a new CSV file.
    """
   
    # Preprocessing steps
    print(f'All data: {len(uniprot_data)}')
    # uniprot_data = uniprot_data.dropna(subset=['Subunit structure'])
    uniprot_data = uniprot_data[uniprot_data['Subunit structure'] != ''] 
    print(f'Filtered out empty data: {len(uniprot_data)}')

    uniprot_data = uniprot_data.astype(str)
    uniprot_data['Subunit structure'] = uniprot_data['Subunit structure'].apply(lambda x: x + ' ')
    uniprot_data['Subunit structure'] = uniprot_data['Subunit structure'].apply(lambda x: x.split('.')[0] + '.' if len(x.split('.')) > 1 and (x.split('.')[1] == '' or x.split('.')[1][0] == ' ') else x.split('.')[0] + '.' + x.split('.')[1] + '.') 

    mask = uniprot_data['mapped_evidence_types'].str.contains('ECO:0000250') & ~uniprot_data['Subunit structure'].str.contains('PubMed:')
    uniprot_data = uniprot_data[~mask]
    print(f'Removed ECO:0000250: {len(uniprot_data)}')

    uniprot_data = uniprot_data[uniprot_data['Subunit structure'].str.contains('Monomer|monomer|Homodimer|homodimer|Homotrimer|homotrimer|Homotetramer|homotetramer|Homopentamer|homopentamer|Homohexamer|homohexamer|Homoheptamer|homoheptamer|Homooctamer|homooctamer|Homodecamer|homodecamer|Homododecamer|homododecamer')]
    print(f'Extracting 10 subunit labels: {len(uniprot_data)}')

    uniprot_data = uniprot_data[~uniprot_data['Subunit structure'].str.contains('(By similarity)|(Probable)|(Potential)')]
    print(f'Filtered out data containing "(By similarity)", "(Probable)", or "(Potential)": {len(uniprot_data)}')

    # Update labels based on specific keywords
    uniprot_data['label'] = '0'
    keywords_Monomer = ['Active as a monomer','Binds DNA as a monomer','Monomer (in vitro) (PubMed:','Monomer (PubMed:','Monomer (Ref.','Monomer in solution','Monomer.','Monomer;','Monomeric in solution.','Binds DNA as monomer','Forms monomers in solution','Monomer (disintegrin)','Monomer (G-actin)']
    keywords_Homodimer = ['Forms head-to-head homodimers','Forms head-to-tail homodimers','Forms homodimer in solution','Forms homodimers','Homodimer (PubMed:','Homodimer (Ref.','Homodimer (via','Homodimer in solution','Homodimer of','Homodimer,','Homodimer.','Homodimer;','Forms a homodimer','Active as a homodimer.','Acts as homodimer','Binds DNA as a homodimer','Binds DNA as homodimer','Can form homodimer','Can homodimerize','Forms a functional homodimer','Forms an asymmetric homodimer','Forms disulfide-linked homodimers','Forms homodimer','Head to tail homodimer','Headphone-shaped homodimer','Homodimer (in vitro)','Homodimer formed by','Homodimer in','Homodimer that','Homodimers.','Homodimerizes.']
    keywords_Homotrimers = ['Forms homotrimers (PubMed:','Monomer (PubMed:','Monomer (Ref.','Homotrimer formed of','Homotrimer in solution','Homotrimer,','Homotrimer.','Homotrimer;','Can form homotrimer','Forms homotrimers','Homotrimer (PubMed:','Homotrimers of','Homotrimers.']
    keywords_Homotetramer = ['Forms homotetramers','Homotetramer composed of','Homotetramer consisting of','Homotetramer formed','Homotetramer in solution','Homotetramer,','Homotetramer.','Homotetramer:','Homotetramer;','A homotetramer formed','Binds DNA as a homotetramer','Homotetramer (in vitro)','Homotetramer (MAT-I)','Homotetramer (PubMed:','Homotetramer (Ref.','Homotetramer in']
    keywords_Homopentamer = ['Homopentamer (PubMed:','Homopentamer with','Homopentamer arranged in','Homopentamer.','Homopentamer;','Forms a homopentamer','Homopentamer (in vitro)','Homopentamer,']
    keywords_Homohexamer = ['Homohexamer (PubMed:','Homohexamer composed of','Homohexamer in solution','Homohexamer with','Homohexamer,','Homohexamer.','Homohexamer;','Homohexameric ring arranged as','A double ring-shaped homohexamer of','Forms a homohexamer','Forms homohexameric rings','Forms homohexamers','Homohexamer (dimer of homotrimers)']
    keywords_Homoheptamer = ['Homoheptamer.','Homoheptamer arranged in','Homoheptamer;']
    keywords_homooctamers = ['Forms only homooctamers','Homooctamer composed of','Homooctamer,','Homooctamer.','Homooctamer;','Homooctamer (isoform 2)','Homooctamer formed by','Homooctamer of','Homooctomer (PubMed:']
    keywords_Homodecamer = ['Homodecamer.','Homodecamer;','Homodecamer composed of','Forms an asymmetric tunnel-fold homodecamer','Homodecamer,','Homodecamer consisting of','Homodecamer of','Homodecamer; composed of']
    keywords_Homododecamer = ['Homododecamer (PubMed:','Homododecamer composed of','Homododecamer.','Homododecamer;']

    for index,row in uniprot_data.iterrows():
        if any(e in row['Subunit structure'] for e in keywords_Monomer):
            row['label'] = '1'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodimer):
            row['label'] = '2'
        elif any(e in row['Subunit structure'] for e in keywords_Homotrimers):
            row['label'] = '3'   
        elif any(e in row['Subunit structure'] for e in keywords_Homotetramer):
            row['label'] = '4'   
        elif any(e in row['Subunit structure'] for e in keywords_Homopentamer):
            row['label'] = '5'   
        elif any(e in row['Subunit structure'] for e in keywords_Homohexamer):
            row['label'] = '6'   
        elif any(e in row['Subunit structure'] for e in keywords_Homoheptamer):
            row['label'] = '7'   
        elif any(e in row['Subunit structure'] for e in keywords_homooctamers):
            row['label'] = '8'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodecamer):
            row['label'] = '10'   
        elif any(e in row['Subunit structure'] for e in keywords_Homododecamer):
            row['label'] = '12' 
                
    keywords_Monomer_and = [keyword + ' and' for keyword in keywords_Monomer]
    keywords_Homodimer_and = [keyword + ' and' for keyword in keywords_Homodimer]
    keywords_Homotrimers_and = [keyword + ' and' for keyword in keywords_Homotrimers]
    keywords_Homotetramer_and = [keyword + ' and' for keyword in keywords_Homotetramer]
    keywords_Homopentamer_and = [keyword + ' and' for keyword in keywords_Homopentamer]
    keywords_Homohexamer_and = [keyword + ' and' for keyword in keywords_Homohexamer]
    keywords_Homoheptamer_and = [keyword + ' and' for keyword in keywords_Homoheptamer]
    keywords_homooctamers_and = [keyword + ' and' for keyword in keywords_homooctamers]
    keywords_Homodecamer_and = [keyword + ' and' for keyword in keywords_Homodecamer]
    keywords_Homododecamer_and = [keyword + ' and' for keyword in keywords_Homododecamer]

    for index,row in uniprot_data.iterrows():
        if any( e in row['Subunit structure'] for e in keywords_Monomer_and):
            row['label'] = '1-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodimer_and):
            row['label'] = '2-and'
        elif any(e in row['Subunit structure'] for e in keywords_Homotrimers_and):
            row['label'] = '3-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homotetramer_and):
            row['label'] = '4-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homopentamer_and):
            row['label'] = '5-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homohexamer_and):
            row['label'] = '6-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homoheptamer_and):
            row['label'] = '7-and'   
        elif any(e in row['Subunit structure'] for e in keywords_homooctamers_and):
            row['label'] = '8-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodecamer_and):
            row['label'] = '10-and'   
        elif any(e in row['Subunit structure'] for e in keywords_Homododecamer_and):
            row['label'] = '12-and'
                
    keywords_Monomer_or = [keyword + ' or' for keyword in keywords_Monomer]
    keywords_Homodimer_or = [keyword + ' or' for keyword in keywords_Homodimer]
    keywords_Homotrimers_or = [keyword + ' or' for keyword in keywords_Homotrimers]
    keywords_Homotetramer_or = [keyword + ' or' for keyword in keywords_Homotetramer]
    keywords_Homopentamer_or = [keyword + ' or' for keyword in keywords_Homopentamer]
    keywords_Homohexamer_or = [keyword + ' or' for keyword in keywords_Homohexamer]
    keywords_Homoheptamer_or = [keyword + ' or' for keyword in keywords_Homoheptamer]
    keywords_homooctamers_or = [keyword + ' or' for keyword in keywords_homooctamers]
    keywords_Homodecamer_or = [keyword + ' or' for keyword in keywords_Homodecamer]
    keywords_Homododecamer_or = [keyword + ' or' for keyword in keywords_Homododecamer]

    for index,row in uniprot_data.iterrows():
        if any( e in row['Subunit structure'] for e in keywords_Monomer_or):
            row['label'] = '1-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodimer_or):
            row['label'] = '2-or'
        elif any(e in row['Subunit structure'] for e in keywords_Homotrimers_or):
            row['label'] = '3-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homotetramer_or):
            row['label'] = '4-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homopentamer_or):
            row['label'] = '5-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homohexamer_or):
            row['label'] = '6-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homoheptamer_or):
            row['label'] = '7-or'   
        elif any(e in row['Subunit structure'] for e in keywords_homooctamers_or):
            row['label'] = '8-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homodecamer_or):
            row['label'] = '10-or'   
        elif any(e in row['Subunit structure'] for e in keywords_Homododecamer_or):
            row['label'] = '12-or'  

    # Filter labels
    valid_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '12']
    uniprot_data = uniprot_data[uniprot_data['label'].isin(valid_labels)]
    print(f'Extracted data containing keywords: {len(uniprot_data)}')

    # Select required columns and save to the dataset_outfile
    uniprot_data = uniprot_data[['Entry', 'Sequence', 'label','organism','EC number']]
    uniprot_data.reset_index(drop=True)
    uniprot_data.to_csv(dataset_outfile, index=None)
    
    print('Done')
    print(uniprot_data.shape)
    return uniprot_data



def get_dataset_distribution(orig_data, subunit_num_distribution_png):
    """
    Generate a box plot to visualize the distribution of subunit numbers in the dataset.

    Args:
        dataset_outfile (str): The path of the dataset CSV file containing processed Uniprot data.
        subunit_num_distribution_png (str): The path where the generated box plot PNG file will be saved.
    """
    uniprot_data = orig_data.copy()
    uniprot_data['label'] = uniprot_data['label'].apply(lambda x: int(x))
    
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '12']
    subunit_data = []

    for i in x:
        subunit_data.append(uniprot_data[uniprot_data['label'] == int(i)]['label'].tolist())

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = np.arange(len(x))

    # Create the box plot
    subunit_box = ax.boxplot(subunit_data, positions=positions, widths=0.6, showfliers=False, patch_artist=True,
                             medianprops=dict(linestyle='None'), boxprops=dict(linewidth=0), zorder=1)  # Set zorder

    colors = ['lightblue']
    for box in subunit_box['boxes']:
        box.set(edgecolor='none', facecolor=colors[0])
        subunit_bar = ax.bar(positions, [len(subunit) for subunit in subunit_data], color='#4CAF50', alpha=0.7, edgecolor='none', zorder=2, capsize=3, error_kw=dict(lw=0, capthick=0))

    # Add value annotations on top of each bar
    for i, value in enumerate([len(subunit) for subunit in subunit_data]):
        ax.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=11)

    ax.set_xticks(positions)
    ax.tick_params(axis='both', which='both', labelsize=13)
    ax.set_xticklabels(x)
    ax.set_xlabel('Subunit number', fontsize=18)
    ax.set_ylabel('Number of Protein Sequences', fontsize=18)

    # Set a custom y-axis limit greater than the maximum value in the bar plot
    custom_y_limit = max(map(len, subunit_data)) + 5000
    ax.set_ylim(0, custom_y_limit)

    # plt.tight_layout()
    plt.savefig(subunit_num_distribution_png, dpi=300, bbox_inches='tight')
    plt.show()

def get_label_length_distribution(orig_data, label_length_distribution_png): 
    """
    Generate a box plot to visualize the distribution of sequence lengths and subunit number*sequence lengths in the dataset.

    Args:
        dataset_outfile (str): The path of the dataset CSV file containing processed Uniprot data.
        label_length_distribution_png (str): The path where the generated box plot PNG file will be saved.
    """
    uniprot_data = orig_data.copy()
    uniprot_data['length'] = uniprot_data['Sequence'].apply(lambda x: len(x))
    uniprot_data['subunit number*length'] = uniprot_data.apply(lambda row: row['label'] * row['length'], axis=1)
    uniprot_data = uniprot_data[['Entry','label','length','subunit number*length']]
    label = sorted(uniprot_data['label'].unique())
    length_data = []

    for l in label:
        lengths = uniprot_data[uniprot_data['label'] == l]['length'].tolist()
        length_data.append(lengths)

    label_length_data = [[l * length for length in lengths] for l, lengths in zip(label, length_data)]
    
    plt.figure(figsize=(8, 6), dpi=300) 
    fig, ax = plt.subplots()
    positions = np.arange(len(label))

    length_box = ax.boxplot(length_data, positions=positions-0.2, widths=0.3, showfliers=False, patch_artist=True)
    label_length_box = ax.boxplot(label_length_data, positions=positions+0.2, widths=0.3, showfliers=False, patch_artist=True)

    colors = ['lightblue', 'lightgreen']
    for box in length_box['boxes']:
        box.set(color='blue', linewidth=1.5)
        box.set(facecolor=colors[0])
    for box in label_length_box['boxes']:
        box.set(color='green', linewidth=1.5)
        box.set(facecolor=colors[1])

    ax.legend([length_box['boxes'][0], label_length_box['boxes'][0]], ['length', 'subunit number*length'], fontsize=16)

    ax.set_xticks(positions)
    ax.set_xticklabels(label)
    ax.set_xlabel('Subunit number', fontsize=20)
    ax.set_ylabel('Number of amino acids', fontsize=20)
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_ylim(0, 7000)

    plt.tight_layout()
    plt.savefig(label_length_distribution_png, dpi=300, bbox_inches='tight')
    plt.show()
    
def split_ECnumber(uniprot_EC_data):
    new_rows = []

    for index, row in uniprot_EC_data.iterrows():
        ec_numbers = row['EC number'].split(',')  
        
        for ec in ec_numbers:
            new_row = row.copy() 
            new_row['EC number'] = ec.strip() 
            new_rows.append(new_row) 

    uniprot_EC_data_new = pd.DataFrame(new_rows)

    return(uniprot_EC_data_new)   

def get_subunit_num_kinds_by_EC(uniprot_data):
    # uniprot_data = uniprot_data.head(10)
    grouped = uniprot_data.groupby('EC number')['uniprot_label'].apply(list).reset_index()

    EC_label = pd.DataFrame({
        'EC number': grouped['EC number'],
        'uniprot_label': grouped['uniprot_label']
    })

    return(EC_label)

def get_ec_subunit_num_ratio(ori_data,ec_subunit_num_ratio_png):
    uniprot_EC_data = ori_data[['Entry','EC number']]
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number']!='']
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number'].notna()]
    
    uniprot_EC_data_new = split_ECnumber(uniprot_EC_data)
    uniprot_EC_data_new = uniprot_EC_data_new[~uniprot_EC_data_new['EC number'].str.contains('-')]

    uniprot_data = ori_data[['Entry','label']]
    uniprot_data.rename(columns={'label': 'uniprot_label'}, inplace=True)

    uniprot_data = pd.merge(uniprot_data,uniprot_EC_data_new,how='left')
    uniprot_data = uniprot_data[uniprot_data['EC number'].notna()]

    EC_label = get_subunit_num_kinds_by_EC(uniprot_data)

    EC_label['length'] = EC_label['uniprot_label'].apply(lambda x:len(x))
    EC_label['uniprot_label_set'] = EC_label['uniprot_label'].apply(lambda x: list(set(x))).apply(sorted)
    EC_label['length_set'] = EC_label['uniprot_label_set'].apply(lambda x:len(x))

    value_counts = EC_label['length_set'].value_counts()
    EC_label.to_excel("2.xlsx", index=False)
    plt.figure(figsize=(8, 6))  
    patches, texts = plt.pie(
            value_counts.values,
            labels=[ str(round(x/sum(value_counts.values)*100,2))+"%" for x in value_counts.values],
            startangle=140,
            colors=sns.color_palette(palette='Accent',desat=0.7),
            explode=(0, 0.1, 0.2, 0.3, 0.5), 
            labeldistance=1.2,
            radius=0.9,
            counterclock=False,
            textprops={'color':'black',
                    'fontsize':15,
                        }
            )


    # 图例信息
    legend_title = 'The number of different subunit structures'
    legend_name = [1,2,3,4,5]

    plt.legend(patches, legend_name,
            title=legend_title,
            title_fontsize=16,
            loc="center left",
            bbox_to_anchor=(0, 1),
            ncol=5,
            fontsize=12
            )

    # plt.title('Ration of Subunit Types for EC',size=20)
    plt.savefig(ec_subunit_num_ratio_png,dpi =300,bbox_inches='tight')
    plt.show()
    

def plot_heatmap(heatmap_data, ec_subunit_num_heatmap_png):
    plt.figure(figsize=(8, 6))
    np.fill_diagonal(heatmap_data.values, 0)
    sns.heatmap(data=heatmap_data,               
                cmap=plt.get_cmap('Greens'),
                center=230,
                linewidths=1,
                
                cbar=True,
                cbar_kws={'label': 'EC count',
                            'orientation': 'vertical',
                            "ticks":np.arange(0,500,100),
                            }
                )

    # plt.title('Subunit Number frequency Heatmap',size =20)
    plt.yticks(size = 16)
    
    plt.gca().tick_params(axis='both', which='both', labelsize=14)
    plt.savefig(ec_subunit_num_heatmap_png,dpi =300,bbox_inches='tight')
    plt.show()


def get_ec_subunit_num_heatmap(ori_data,ec_subunit_num_heatmap_png):
    uniprot_EC_data = ori_data.copy()
    uniprot_EC_data = uniprot_EC_data[['Entry','EC number']]
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number']!='']
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number'].notna()]

    uniprot_EC_data_new = split_ECnumber(uniprot_EC_data)
    uniprot_EC_data_new = uniprot_EC_data_new[~uniprot_EC_data_new['EC number'].str.contains('-')]

    uniprot_data = ori_data[['Entry','label']]
    uniprot_data.rename(columns={'label': 'uniprot_label'}, inplace=True)

    uniprot_data = pd.merge(uniprot_data,uniprot_EC_data_new,how='left')
    uniprot_data = uniprot_data[uniprot_data['EC number'].notna()]

    EC_label = get_subunit_num_kinds_by_EC(uniprot_data)

    EC_label['length'] = EC_label['uniprot_label'].apply(lambda x:len(x))
    EC_label['uniprot_label_set'] = EC_label['uniprot_label'].apply(lambda x: list(set(x))).apply(sorted)
    EC_label['length_set'] = EC_label['uniprot_label_set'].apply(lambda x:len(x))
    
    EC_label=EC_label[EC_label['length_set']>1]
    
    # 构建热图数据
    labels_list = ['Monomer',
    'Homodimer',
    'Homotrimers',
    'Homotetramer',
    'Homopentamner',
    'Homohexamer',
    'Homoheptamer',
    'Homooctamers',
    'Homodecamer',
    'Homododecamer']
    label_dict = {}

    for i in range(len(labels_list)):
        label_dict[labels_list[i]] = i + 1
    label_dict['Homodecamer'] = 10
    label_dict['Homododecamer'] = 12


    data = np.zeros((10, 10))  # 初始化一个全零的10x10数组
    # for i,xlabel in enumerate(labels_list):
    #     for j,ylabel in enumerate(labels_list):
    #         pass
    heatmap_data = pd.DataFrame(data, index=labels_list, columns=labels_list)

    for index_name in labels_list:
        for columns_name in labels_list:
            value = 0
            index, columns = label_dict[index_name], label_dict[columns_name]
            value = EC_label[EC_label['uniprot_label_set'].apply(lambda x:index in x and columns in x)].shape[0]
            heatmap_data.loc[index_name,columns_name] = value
    heatmap_data.to_excel("1.xlsx",index=False)
    plot_heatmap(heatmap_data,ec_subunit_num_heatmap_png)
    
def get_distribution_among_species(ori_data, distribution_among_species_png):
    # uniprot_data = pd.read_csv(dataset_outfile,sep=',')
    # organsim_df = pd.read_csv(organsim_file,sep='\t')
    # subunit_with_organism_df = pd.merge(uniprot_data,organsim_df[['Entry','Organism']],how='left',on='Entry')
    uniprot_data = ori_data[ori_data['organism'].notna()]

    species_list = ['Homo sapiens', 'Mus musculus', 'Saccharomyces cerevisiae', 'Escherichia coli']
    species_dict = {}
    label_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    color_palette = sns.hls_palette(10, l=.7, s=.8)
    legend_labels = ['Monomer', 'Homodimer', 'Homotrimers', 'Homotetramer', 'Homopentamner', 'Homohexamer', 'Homoheptamer', 'Homooctamers', 'Homodecamer', 'Homododecamer']
    legend_colors = color_palette[:len(label_sequence)]  # Select colors for legend based on label sequence
    
    for species in species_list:
        group_df = uniprot_data[uniprot_data['organism'].str.contains(species)].groupby('label')
        count_list = [0] * 10
        for i, df in group_df:
            index = label_sequence.index(i)
            count_list[index] = df.shape[0]
        species_dict[species] = count_list

    x_label = [1, 2, 3, 4]
    x = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []  # Collect legend handles
    
    for species in species_list:
        x += 1
        bottom = 0
        y_list = [y / sum(species_dict[species]) for y in species_dict[species]]
        for i, y in enumerate(y_list):
            bar = plt.bar(x, y, width=0.6, bottom=bottom, color=color_palette[i])
            bottom += y
            handles.append(bar)  # Append each bar to legend handles

    ax.set_ylabel('Ratio', fontsize=16)
    # ax.set_title('Distribution of ratio by Label', fontsize=18)
    ax.set_xticks(x_label)
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_xticklabels(species_list, rotation=30, fontsize=14)
    ax.set_ylim(0, 1.2)

    # Add legend
    plt.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Subunit', fontsize=12, title_fontsize=16)

    plt.savefig(distribution_among_species_png, dpi=300, bbox_inches='tight')
    plt.show()