#coding=utf-8
import os
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def download_uniprot_data(url, store_path):
    """
    Download Uniprot data from the given URL and save it to a file.

    Args:
        url (str): The URL to fetch Uniprot data.
        store_path (str): The path to store the downloaded data.
    """
    response = requests.get(url)

    if response.status_code == 200:
        data = response.content.decode('utf-8')
        lines = data.split('\n')
        with open(store_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            for line in lines:
                writer.writerow(line.split('\t'))
        print("Data saved to", store_path)
    else:
        print("Failed to retrieve data. Status code:", response.status_code)


def get_dataset_from_uniprot(uniprot_data_file, dataset_outfile):
    """
    Read Uniprot data from a CSV file, preprocess the data, and filter out entries based on specific keywords.
    Assign labels to each entry based on the presence of keywords in the 'Subunit structure' column.
    Save the processed data to a new CSV file.

    Args:
        uniprot_data_file (str): Path to the CSV file containing Uniprot data.
        dataset_outfile (str): Path to save the processed dataset as a new CSV file.
    """
    uniprot_data = pd.read_csv(uniprot_data_file, sep='\t')
    
    # Preprocessing steps
    uniprot_data = uniprot_data.dropna(subset=['Subunit structure'])
    uniprot_data = uniprot_data.astype(str)
    uniprot_data['Subunit structure'] = uniprot_data['Subunit structure'].apply(lambda x: x + ' ')
    uniprot_data['Subunit structure'] = uniprot_data['Subunit structure'].apply(
        
        lambda x: x.split('.')[0] + '.' 
        if len(x.split('.')) > 1 and x.split('.')[1] and x.split('.')[1][0] == ' ' 
        else x.split('.')[0] + '.' + x.split('.')[1] + '.' 
        if len(x.split('.')) > 1 
        else x
    )
    # uniprot_data['Subunit structure'] = uniprot_data['Subunit structure'].apply(lambda x: x.split('.')[0] + '.' if x.split('.')[1][0] == ' ' else x.split('.')[0] + '.' + x.split('.')[1] + '.')
    uniprot_data = uniprot_data[uniprot_data['Subunit structure'].str.contains('Monomer|monomer|Homodimer|homodimer|Homotrimer|homotrimer|Homotetramer|homotetramer|Homopentamer|homopentamer|Homohexamer|homohexamer|Homoheptamer|homoheptamer|Homooctamer|homooctamer|Homodecamer|homodecamer|Homododecamer|homododecamer')]
    uniprot_data = uniprot_data[~uniprot_data['Subunit structure'].str.contains('(By similarity)|(Probable)|(Potential)')]
    uniprot_data['label'] = '0'
    
    # Update labels based on specific keywords
    keywords_Monomer = ['Active as a monomer','Binds DNA as a monomer','Monomer (in vitro) (PubMed:','Monomer (PubMed:','Monomer (Ref.','Monomer in solution','Monomer.','Monomer;','Monomeric in solution.','Binds DNA as monomer','Forms monomers in solution','Monomer (disintegrin)','Monomer (G-actin)']
    keywords_Homodimer = ['Forms head-to-head homodimers','Forms head-to-tail homodimers','Forms homodimer in solution','Forms homodimers','Homodimer (PubMed:','Homodimer (Ref.','Homodimer (via','Homodimer in solution','Homodimer of','Homodimer,','Homodimer.','Homodimer;','Forms a homodimer','Active as a homodimer.','Acts as homodimer','Binds DNA as a homodimer','Binds DNA as homodimer','Can form homodimer','Can homodimerize','Forms a functional homodimer','Forms an asymmetric homodimer','Forms disulfide-linked homodimers','Forms homodimer','Head to tail homodimer','Headphone-shaped homodimer','Homodimer (in vitro)','Homodimer formed by','Homodimer in','Homodimer that']
    keywords_Homotrimers = ['Forms homotrimers (PubMed:','Monomer (PubMed:','Monomer (Ref.','Homotrimer formed of','Homotrimer in solution','Homotrimer,','Homotrimer.','Homotrimer;','Can form homotrimer','Forms homotrimers','Homotrimer (PubMed:','Homotrimers of']
    keywords_Homotetramer = ['Forms homotetramers','Homotetramer composed of','Homotetramer consisting of','Homotetramer formed','Homotetramer in solution','Homotetramer,','Homotetramer.','Homotetramer:','Homotetramer;','A homotetramer formed','Binds DNA as a homotetramer','Homotetramer (in vitro)','Homotetramer (MAT-I)','Homotetramer (PubMed:','Homotetramer (Ref.','Homotetramer in']
    keywords_Homopentamer = ['Homopentamer (PubMed:','Homopentamer with','Homopentamer arranged in','Homopentamer.','Homopentamer;','Forms a homopentamer','Homopentamer (in vitro)','Homopentamer,']
    keywords_Homohexamer = ['Homohexamer (PubMed:','Homohexamer composed of','Homohexamer in solution','Homohexamer with','Homohexamer,','Homohexamer.','Homohexamer;','Homohexameric ring arranged as','A double ring-shaped homohexamer of','Forms a homohexamer','Forms homohexameric rings','Forms homohexamers','Homohexamer (dimer of homotrimers)']
    keywords_Homoheptamer = ['Homoheptamer.','Homoheptamer arranged in','Homoheptamer;']
    keywords_homooctamers = ['Forms only homooctamers','Homooctamer composed of','Homooctamer,','Homooctamer.','Homooctamer;','Homooctamer (isoform 2)','Homooctamer formed by','Homooctamer of','Homooctomer (PubMed:']
    keywords_Homodecamer = ['Homodecamer.','Homodecamer;','Homodecamer composed of','Forms an asymmetric tunnel-fold homodecamer','Homodecamer,','Homodecamer consisting of','Homodecamer of','Homodecamer; composed of']
    keywords_Homododecamer = ['Homododecamer (PubMed:','Homododecamer composed of','Homododecamer.','Homododecamer;']

    for index,row in uniprot_data.iterrows():
        if  any(e in row['Subunit structure'] for e in keywords_Monomer):
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
        if  any( e in row['Subunit structure'] for e in keywords_Monomer_and):
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
        if  any( e in row['Subunit structure'] for e in keywords_Monomer_or):
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

    # Select required columns and save to the dataset_outfile
    uniprot_data = uniprot_data[['Entry', 'Sequence', 'label']]
    uniprot_data.to_csv(dataset_outfile, index=None)
    
    print('Done')
    print(uniprot_data.shape)


def get_dataset_distribution(dataset_outfile, subunit_num_distribution_png):
    """
    Generate a bar chart to visualize the distribution of subunit numbers in the dataset.

    Args:
        dataset_outfile (str): The path of the dataset CSV file containing processed Uniprot data.
        subunit_num_distribution_png (str): The path where the generated bar chart PNG file will be saved.
    """
    uniprot_data = pd.read_csv(dataset_outfile)
    uniprot_data['label'] = uniprot_data['label'].apply(lambda x: int(x))
    
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '12']
    y = []
    for i in x:
        y.append(len(uniprot_data[uniprot_data['label'] == int(i)]))
        
    
    font1 = {'weight': 'normal', 'size': 20}
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.rcParams['font.size'] = 15
    plt.title("Distribution of subunit number", font1)
    plt.xlabel('Subunit number', font1)
    plt.ylabel('Num', font1)
    
    p1 = plt.bar(x, y, width=0.8, color="#4CAF50")
    
    plt.bar_label(p1, label_type='edge')
    
    plt.savefig(subunit_num_distribution_png,dpi =300,bbox_inches='tight')
    plt.show()


def get_label_length_distribution(dataset_outfile,label_length_distribution_png): 
    """
    Generate a box plot to visualize the distribution of sequence lengths and subunit number*sequence lengths in the dataset.

    Args:
        dataset_outfile (str): The path of the dataset CSV file containing processed Uniprot data.
        label_length_distribution_png (str): The path where the generated box plot PNG file will be saved.
    """
    uniprot_data = pd.read_csv(dataset_outfile)
    uniprot_data['length'] = uniprot_data['Sequence'].apply(lambda x: len(x))
    uniprot_data['subunit number*length'] = uniprot_data.apply(lambda row: row['label'] * row['length'], axis=1)
    uniprot_data = uniprot_data[['Entry','label','length','subunit number*length']]
    label = sorted(uniprot_data['label'].unique())
    length_data = []

    for l in label:
        lengths = uniprot_data[uniprot_data['label'] == l]['length'].tolist()
        length_data.append(lengths)

    label_length_data = [[l * length for length in lengths] for l, lengths in zip(label, length_data)]
    plt.figure(figsize=(10, 10), dpi=300)
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

    ax.legend([length_box['boxes'][0], label_length_box['boxes'][0]], ['length', 'subunit number*length'])

    ax.set_xticks(positions)
    ax.set_xticklabels(label)
    ax.set_xlabel('Subunit number')
    ax.set_ylabel('Number of amino acids')
    ax.set_ylim(0,7000)
    plt.title('Boxplot of Length and subunit number*Length')

    plt.tight_layout()
    plt.savefig(label_length_distribution_png,dpi =300,bbox_inches='tight')
    plt.show()
    
def split_ECnumber(uniprot_EC_data):
    new_rows = []

    for index, row in uniprot_EC_data.iterrows():
        ec_numbers = row['EC number'].split(';')  
        
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

def get_ec_subunit_num_ratio(dataset_outfile,entry_EC_file,ec_subunit_num_ratio_png):
    uniprot_EC_data = pd.read_csv(entry_EC_file,sep='\t')
    uniprot_EC_data = uniprot_EC_data[['Entry','EC number']]
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number'].notna()]
    uniprot_EC_data_new = split_ECnumber(uniprot_EC_data)
    uniprot_EC_data_new = uniprot_EC_data_new[~uniprot_EC_data_new['EC number'].str.contains('-')]

    uniprot_data = pd.read_csv(dataset_outfile)
    uniprot_data = uniprot_data[['Entry','label']]
    uniprot_data.rename(columns={'label': 'uniprot_label'}, inplace=True)

    uniprot_data = pd.merge(uniprot_data,uniprot_EC_data_new,how='left')
    uniprot_data = uniprot_data[uniprot_data['EC number'].notna()]

    EC_label = get_subunit_num_kinds_by_EC(uniprot_data)

    EC_label['length'] = EC_label['uniprot_label'].apply(lambda x:len(x))
    EC_label['uniprot_label_set'] = EC_label['uniprot_label'].apply(lambda x: list(set(x))).apply(sorted)
    EC_label['length_set'] = EC_label['uniprot_label_set'].apply(lambda x:len(x))

    value_counts = EC_label['length_set'].value_counts()
    
    plt.figure(figsize=(8, 8))  
    patches, texts = plt.pie(
            value_counts.values,
            labels=value_counts.index,
            startangle=140,
            colors=sns.color_palette(palette='Accent',desat=0.7),
            explode=(0, 0.1, 0.1, 0.1, 0.2), 
            labeldistance=1.2,
            radius=0.9,
            counterclock=False,
            textprops={'color':'b',
                    'fontsize':20,
                        }
            )

    # 图例信息
    legend_title = 'Type of Homomeric Oligmers'
    legend_name = [1,2,3,4,5]

    plt.legend(patches, legend_name,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            ncol=2,
            )

    plt.title('Ration of Subunit Types for EC')
    plt.savefig(ec_subunit_num_ratio_png,dpi =300,bbox_inches='tight')
    plt.show()
    

def plot_heatmap(heatmap_data, ec_subunit_num_heatmap_png):
    plt.figure(dpi=120)
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

    plt.title('Subunit Number frequency Heatmap')

    plt.savefig(ec_subunit_num_heatmap_png,dpi =300,bbox_inches='tight')
    plt.show()


def get_ec_subunit_num_heatmap(dataset_outfile,entry_EC_file,ec_subunit_num_heatmap_png):
    uniprot_EC_data = pd.read_csv(entry_EC_file,sep='\t')
    uniprot_EC_data = uniprot_EC_data[['Entry','EC number']]
    uniprot_EC_data = uniprot_EC_data[uniprot_EC_data['EC number'].notna()]
    uniprot_EC_data_new = split_ECnumber(uniprot_EC_data)
    uniprot_EC_data_new = uniprot_EC_data_new[~uniprot_EC_data_new['EC number'].str.contains('-')]

    uniprot_data = pd.read_csv(dataset_outfile)
    uniprot_data = uniprot_data[['Entry','label']]
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

    plot_heatmap(heatmap_data,ec_subunit_num_heatmap_png)
    
def get_distribution_among_species(dataset_outfile,organsim_file,distribution_among_species_png):
    uniprot_data = pd.read_csv(dataset_outfile,sep=',')
    organsim_df = pd.read_csv(organsim_file,sep='\t')
    subunit_with_organism_df = pd.merge(uniprot_data,organsim_df[['Entry','Organism']],how='left',on='Entry')
    subunit_with_organism_df = subunit_with_organism_df[subunit_with_organism_df['Organism'].notna()]
    
    species_list = ['Homo sapiens','Mus musculus','Saccharomyces cerevisiae','Escherichia coli']
    species_dict = {}
    label_sequence = [1,2,3,4,5,6,7,8,10,12]
    for species in species_list:
        group_df = subunit_with_organism_df[subunit_with_organism_df['Organism'].str.contains(species)].groupby('label')
        count_list = [0] * 10
        for i,df in group_df:
            index = label_sequence.index(i)
            count_list[index] = df.shape[0]
        species_dict[species] = count_list
        
    x_label = [1,2,3,4]
    x = 0
    fig, ax = plt.subplots()
    for species in species_list:
        x += 1
        bottom = 0
        y_list = [y/sum(species_dict[species]) for y in species_dict[species]]
        print()
        for i,y in enumerate(y_list):
            plt.bar(x,y, width=0.6, bottom=bottom,color=sns.hls_palette(10,l=.7,s=.8)[i]
                    )
            bottom += y
            
    ax.set_ylabel('Ration')
    ax.set_title('Distribution of ration by Label')
    ax.set_xticks(x_label,)
    ax.set_xticklabels(species_list,rotation = 30,fontsize = 'small')
    ax.set_ylim(0,1.2)
    
    plt.savefig(distribution_among_species_png,dpi =300,bbox_inches='tight')
    plt.show()