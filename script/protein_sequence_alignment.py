#coding=utf-8
from Bio import pairwise2
from Bio.Seq import Seq
from pandarallel import pandarallel
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from pandarallel import pandarallel

def calculate_and_save_sequence_identity(result1, sequence_alignment_outfile):
    """
    Calculate sequence identity for each row in result1 DataFrame and save it to a CSV file.

    Args:
    result1 (pd.DataFrame): DataFrame containing sequence alignment data.
    sequence_alignment_outfile (str): Path to the output CSV file.

    Returns:
    pd.DataFrame: Updated DataFrame with sequence identity values.
    """
    pandarallel.initialize(progress_bar=True, nb_workers=30)

    result1['identity'] = result1.parallel_apply(sequence_alignment, axis=1)

    result1.to_csv(sequence_alignment_outfile, index=None)

    return result1


def plot_and_save_histogram_same_label(result1, histogram_same_label_file):
    """
    Plot a histogram of sequence identity scores for entries with the same label and save it to a file.

    Args:
    result1 (pd.DataFrame): DataFrame containing sequence alignment data.
    histogram_same_label_file (str): Path to save the histogram image file.

    Returns:
    None
    """
    # Filter rows with label_match equal to True
    result1_True = result1[result1['label_match'] == True]

    # Plot histogram for sequence alignment with the same label
    sns.histplot(data=result1_True, x='identity', bins='auto', kde=False, color='blue')
    plt.xlabel('Identity')
    plt.ylabel('Counts')
    plt.title('Frequency Histogram of Sequence Alignment Identity Scores with Same Label')

    plt.savefig(histogram_same_label_file,dpi =300,bbox_inches='tight')

    plt.show()


def plot_and_save_histogram_diff_label(result1, histogram_diff_label_file):
    """
    Plot a histogram of sequence identity scores for entries with different labels and save it to a file.

    Args:
    result1 (pd.DataFrame): DataFrame containing sequence alignment data.
    histogram_diff_label_file (str): Path to save the histogram image file.

    Returns:
    None
    """
    result1_False = result1[result1['label_match'] == False]

    sns.histplot(data=result1_False, x='identity', bins='auto', kde=False, color='red')
    plt.xlabel('Identity')
    plt.ylabel('Counts')
    plt.title('Frequency Histogram of Sequence Alignment Identity Scores with Different Label')

    plt.savefig(histogram_diff_label_file,dpi =300,bbox_inches='tight')

    plt.show()
    

def analyze_uniprot_data(uniprot_file, Swiss_Prot_Entry_EC_file):
    """
    Function to analyze Uniprot data.

    Args:
    uniprot_file (str): Path to the Uniprot data file.
    Swiss_Prot_Entry_EC_file (str): Path to the Swiss-Prot Entry and EC number data file.

    Returns:
    pd.DataFrame: DataFrame containing analysis results.
    """

    # Read Uniprot data
    uniprot_data = pd.read_csv(uniprot_file)
    uniprot_data = uniprot_data[['Entry', 'label']]
    uniprot_data.rename(columns={'label': 'uniprot_label'}, inplace=True)
    print('Loaded Uniprot data shape:', uniprot_data.shape)

    # Read Uniprot EC data
    uniprot_EC_data = pd.read_csv(Swiss_Prot_Entry_EC_file, sep='\t')
    uniprot_EC_data = uniprot_EC_data[['Entry', 'EC number']]

    # Filter out rows with NaN values in EC number column
    uniprot_EC_data = uniprot_EC_data.dropna(subset=['EC number'])
    print('Filtered out rows with NaN in Swiss-Prot EC number column. Remaining rows:', uniprot_EC_data.shape[0])

    # Merge Uniprot data with Uniprot EC data
    uniprot_data = pd.merge(uniprot_data, uniprot_EC_data, how='left')
    uniprot_data = uniprot_data.dropna(subset=['EC number'])
    print('Merge Uniprot dataset with Swiss-Prot:', uniprot_data.shape)
    
    # Split EC numbers
    uniprot_EC_data_new = split_ECnumber(uniprot_data)
    uniprot_EC_data_new = uniprot_EC_data_new[~uniprot_EC_data_new['EC number'].str.contains('-')]
    
    # Get subunit count and types information by EC number
    EC_label = get_subunit_num_kinds_by_EC(uniprot_EC_data_new)

    # Calculate length-related information
    EC_label['length'] = EC_label['uniprot_label'].apply(lambda x: len(x))
    EC_label['uniprot_label_set'] = EC_label['uniprot_label'].apply(lambda x: list(set(x))).apply(sorted)
    EC_label['length_set'] = EC_label['uniprot_label_set'].apply(lambda x: len(x))

    # Filter data with length greater than 2
    EC_label_greater_than_2 = EC_label[EC_label['length_set'] > 2]
    print('Filtered data with length greater than 2. Number of rows:', EC_label_greater_than_2.shape[0])
    
    return EC_label_greater_than_2


def create_protein_pairs_with_sequences(EC_label_greater_than_2, uniprot_file):
    """
    Create pairs of proteins with sequences for further analysis.

    Args:
    EC_label_greater_than_2 (pd.DataFrame): DataFrame containing EC label data.

    Returns:
    pd.DataFrame: DataFrame containing protein pairs and their sequences.
    """

    result = pd.DataFrame(columns=['EC number', 'entry1', 'entry1_label', 'entry2', 'entry2_label', 'identity', 'label_match'])

    for _, row in EC_label_greater_than_2.iterrows():
        ec_number = row['EC number']
        uniprot_labels = row['uniprot_label']
        entries = row['Entry']

        cartesian_product = list(product(entries, repeat=2))

        uniprot_label_pairs = list(product(uniprot_labels, repeat=2))

        new_rows = []
        for i in range(len(cartesian_product)):
            # Exclude combinations of the same entry
            if cartesian_product[i][0] != cartesian_product[i][1]:
                entry1_label = uniprot_label_pairs[i][0]
                entry2_label = uniprot_label_pairs[i][1]

                label_match = (entry1_label == entry2_label)

                new_row = {
                    'EC number': ec_number,
                    'entry1_label': entry1_label,
                    'entry2_label': entry2_label,
                    'entry1': cartesian_product[i][0],
                    'entry2': cartesian_product[i][1],
                    'label_match': label_match
                }
                new_rows.append(new_row)

        result = result.append(new_rows, ignore_index=True)

    # Add protein sequences
    uniprot_data = pd.read_csv(uniprot_file, usecols=['Entry', 'Sequence'])

    result1 = result.merge(uniprot_data, left_on='entry1', right_on='Entry', how='left')
    result1.rename(columns={'Sequence': 'seq1'}, inplace=True)
    result1.drop('Entry', axis=1, inplace=True)

    result1 = result1.merge(uniprot_data, left_on='entry2', right_on='Entry', how='left')
    result1.rename(columns={'Sequence': 'seq2'}, inplace=True)
    result1.drop('Entry', axis=1, inplace=True)

    new_order = ['EC number', 'entry1', 'entry1_label', 'seq1', 'entry2', 'entry2_label', 'seq2', 'identity', 'label_match']
    result1 = result1.reindex(columns=new_order)

    return result1


def split_ECnumber(uniprot_EC_data):
    new_rows = []

    # 遍历每一行
    for index, row in uniprot_EC_data.iterrows():
        ec_numbers = row['EC number'].split(';')  # 按分号分割EC号码
        
        # 对于每个EC号码创建一个新的行
        for ec in ec_numbers:
            new_row = row.copy()  # 复制原始行的内容
            new_row['EC number'] = ec.strip()  # 更新EC号码列
            new_rows.append(new_row)  # 将新行添加到列表中

    # 创建包含拆分后行的新数据框
    uniprot_EC_data_new = pd.DataFrame(new_rows)

    # 打印结果
    return(uniprot_EC_data_new)    


def get_subunit_num_kinds_by_EC(uniprot_data):
    # uniprot_data = uniprot_data.head(10)
    grouped = uniprot_data.groupby('EC number')['uniprot_label'].apply(list).reset_index()
    grouped2 = uniprot_data.groupby('EC number')['Entry'].apply(list).reset_index()
    # 创建新的数据框，存储合并后的结果
    EC_label = pd.DataFrame({
        'EC number': grouped['EC number'],
        'uniprot_label': grouped['uniprot_label'],
        'Entry':grouped2['Entry']
    })

    # 打印结果
    return(EC_label)


def sequence_alignment(row):
    seq1 = row['seq1']
    seq2 = row['seq2']
    # 进行全局比对
    alignments = pairwise2.align.globalxx(seq1, seq2)

    # 获取最佳比对结果
    best_alignment = alignments[0]
    aligned_seq1 = best_alignment[0]
    aligned_seq2 = best_alignment[1]

    identity = sum(aa1 == aa2 for aa1, aa2 in zip(aligned_seq1, aligned_seq2)) / len(aligned_seq1) * 100

    return identity