import os
import argparse
import numpy as np
import pandas as pd
from VFI import gaussian_convolution_vfi

# Molecular Weight
mw_map = {
    'S': 105.09,
    'T': 119.12,
    'Q': 146.15,
    'N': 132.12,
    'Y': 181.19, 
    'C': 121.16,
    'G': 75.07,
    'A': 89.09,
    'V': 117.15,
    'L': 131.17,
    'I': 131.17,
    'M': 149.21,
    'P': 115.13,
    'F': 165.19, 
    'W': 204.23,
    'D': 133.10 ,
    'E': 147.13, 
    'K': 146.19,
    'R': 174.20,
    'H': 155.16,
}

# Volume
v_map = {
    'K': 68.0,
    'H': 49.2,
    'R': 70.8,
    'D': 31.3,
    'E': 47.2,
    'N': 35.4,
    'Q': 51.3,
    'S': 18.1,
    'T': 34.0,
    'C': 28.0,
    'G': 0.0,
    'A': 15.9,
    'P': 41.0,
    'V': 47.7,
    'M': 62.8,
    'I': 63.6,
    'L': 63.6,
    'Y': 78.5,
    'F': 77.2,
    'W': 100.0
}

# Standardized Polarizability
p_map = {
    'K': 64.2,
    'H': 43.2,
    'R': 51.9,
    'D': 100.0,
    'E': 93.8,
    'N': 63.0,
    'Q': 45.7,
    'S': 32.1,
    'T': 21.0,
    'C': 7.4,
    'G': 37.0,
    'A': 25.9,
    'P': 21.0,
    'V': 8.6,
    'M': 4.9,
    'I': 0.0,
    'L': 0.0,
    'Y': 9.9,
    'F': 1.2,
    'W': 4.9
}

# Punto isométrico estandarizado
ip_e_map = {
    'K': 86.9 ,
    'H': 59.2,
    'R': 100.0,
    'D': 0.0,
    'E': 3.2,
    'N': 31.3,
    'Q': 34.4,
    'S': 34.8,
    'T': 45.7,
    'C': 26.3,
    'G': 38.5,
    'A': 39.2,
    'P': 40.2,
    'V': 38.5,
    'M': 35.7,
    'I': 39.2,
    'L': 38.6,
    'Y': 34.4,
    'F': 38.6,
    'W': 37.7
}

# Standardized Hydophobicity
hf_e_map = {
    'K': 43.5,
    'H': 23.1,
    'R': 22.6,
    'D': 17.5,
    'E': 17.8,
    'N': 2.4,
    'Q': 0.0,
    'S': 1.9,
    'T': 1.9,
    'C': 40.3,
    'G': 2.7,
    'A': 23.1,
    'P': 73.5,
    'V': 49.6,
    'M': 44.3,
    'I': 83.6,
    'L': 57.6,
    'Y': 70.8,
    'F': 76.1,
    'W': 100.0 
}

# Mean solvent accesible surface area
msa_map = {
    'K': 54.3,
    'H': 28.1,
    'R': 50.1,
    'D': 45.0,
    'E': 48.6,
    'N': 46.1,
    'Q': 43.6,
    'S': 40.5,
    'T': 35.3,
    'C': 7.4,
    'G': 54.0,
    'A': 37.4,
    'P': 66.2,
    'V': 19.6,
    'M': 3.9,
    'I': 7.5,
    'L': 10.1,
    'Y': 30.1,
    'F': 5.5,
    'W': 13.8 
}

# Hydrophobicity
hf_map = {
    'K': -3.9,
    'H': -3.2,
    'R': -4.5,
    'D': -3.5,
    'E': -3.5,
    'N': -3.5,
    'Q': -3.5,
    'S': -0.8,
    'T': -0.7,
    'C': 2.5,
    'G': -0.4,
    'A': 1.8,
    'P': -1.6,
    'V': 4.2,
    'M': 1.9,
    'I': 4.5,
    'L': 3.8,
    'Y': -1.3,
    'F': 2.8,
    'W': -0.9
}

# Secondary Structure. Output of proteus2
secondary_structure_KCNQ2 = 'CCCCCCCCCCCCCCHHHHHHHHCCECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHHHHCCCCHHHTTTTTTTTTTTTTTTTTTTTTTCHHHHHHHHHCTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCHHHHHHHCTTTTTTTTTTTTTTTTTTCHHHHHHHHHHHHCTTTTTTTTTTTTTTTTCCHHHHHHHHHHHHCCCTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCTTTTTTTTTTTTTCCCCCCCCCCCTTTTTTTTTTTTTTTTTTTCTTTTTTTTCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCECHHHHHHHHHHHHHHHHHCCCCCHHHHHHHHHHCHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCECCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEEEEEEECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCEEECCCCHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
ss_map = {
    'C':'coil',
    'T':'membrane_helix',
    'H':'helix',
    'E':'beta_strand'  
}

# Topological Domain
def domain_map_KCNQ2(x):
    
    if x in np.r_[1:92, 144:167, 219:232, 313:332, 351:357, 367:535, 560:563, 595:622, 648:873]: return 'Cytoplasmic'
    elif x in np.r_[113:123, 188:196, 253:265, 286:292]: return 'Extracelullar'
    elif x >= 92 and x <= 112: return 'S1'
    elif x >= 123 and x <= 143: return 'S2'
    elif x >= 167 and x <= 187: return 'S3'
    elif x >= 196 and x <= 218: return 'S4'
    elif x >= 232 and x <= 252: return 'S5'
    elif x >= 265 and x <= 285: return 'Pore'
    elif x >= 292 and x <= 312: return 'S6'
    elif x >= 332 and x <= 350: return 'hA'
    elif x >= 357 and x <= 366: return 'hTW'
    elif x >= 535 and x <= 559: return 'hB'
    elif x >= 563 and x <= 594: return 'hC'
    elif x >= 622 and x <= 647: return 'hD'

# Functional Domain 
def functional_map_KCNQ2(x):
    
    if x in np.r_[1:92, 219:232, 313:332, 560:563, 648:873]: return 'unknown_function'
    elif x >= 92 and x <= 218: return 'voltage_domain'
    elif x in np.r_[232:277, 283:313]: return 'pore_domain'
    elif x >= 277 and x <= 282: return 'selectivity_filter'
    elif x >= 332 and x <= 559: return 'CaM_interaction'
    elif x >= 563 and x <= 647: return 'SID_domain'

# Charge
def charge_map(x):
    
    if x in ['S', 'T', 'Q', 'N', 'Y', 'C', 'G', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W']: return 'neutral'
    elif x in ['D', 'E']: return 'negative_acidic'
    elif x in ['K', 'R', 'H']: return 'positive_basic'

def d_charge_map(x, y):
    
    if x=='positive_basic' and y=='positive_basic': return 'pos_to_pos'
    elif x=='positive_basic' and y=='negative_acidic': return 'pos_to_neg'
    elif x=='positive_basic' and y=='neutral': return 'pos_to_neu'
    elif x=='negative_acidic' and y=='positive_basic': return 'neg_to_pos'
    elif x=='negative_acidic' and y=='negative_acidic': return 'neg_to_neg'
    elif x=='negative_acidic' and y=='neutral': return 'neg_to_neu'
    elif x=='neutral' and y=='positive_basic': return 'neu_to_pos'
    elif x=='neutral' and y=='negative_acidic': return 'neu_to_neg'
    elif x=='neutral' and y=='neutral': return 'neu_to_neu'

# Polarity
def polar_map(x):
    
    if x in ['S', 'T', 'Q', 'N', 'C', 'G', 'D', 'E', 'K', 'R', 'H']: return 'polar'
    elif x in ['Y', 'A', 'V', 'L', 'I', 'M', 'P', 'F', 'W']: return 'non_polar'

def d_pol_map(x, y):
    
    if x=='polar' and y=='polar': return 'p_to_p'
    elif x=='polar' and y=='non_polar': return 'p_to_np'
    elif x=='non_polar' and y=='polar': return 'np_to_p'
    elif x=='non_polar' and y=='non_polar': return 'np_to_np'

# Aromaticity
def aromatic_map(x):
    
    if x in ['A', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V']: return 'non_aromatic'
    elif x in ['Y', 'F', 'W']: return 'aromatic'

def d_aro_map(x, y):
    
    if x=='aromatic' and y=='aromatic': return 'a_to_a'
    elif x=='aromatic' and y=='non_aromatic': return 'a_to_na'
    elif x=='non_aromatic' and y=='aromatic': return 'na_to_a'
    elif x=='non_aromatic' and y=='non_aromatic': return 'na_to_na'

# Gaps and fragments
def str_pos_map_KCNQ2(x):
    
    if x >= 1 and x <= 69: return 'GAP1'
    elif x >= 70 and x <= 184: return 'FRAG1'
    elif x >= 185 and x <= 194: return 'GAP2'
    elif x >= 195 and x <= 367: return 'FRAG2'
    elif x >= 368 and x <= 534: return 'GAP3'
    elif x >= 535 and x <= 595: return 'FRAG3'
    elif x >= 596 and x <= 872: return 'GAP4'

def featurize(input_file: str, output_dir: str, vfi_sigma: float, position_col: str,
              variant_col: str, label_col: str, gene_info_file: str) -> None:
    """Given a file containing a set of KCNQ2 variants, obtains features for each of these variants and
    saves them to a csv file in the specified directory.

    Args:
        input_file (str): Name of the file containing the set of variants.
        output_dir (str): Name of directory where featurized samples will be saved.
        vfi_sigma (float): Value of the sigma parameter used to compute the VFI feature.
        position_col (str): Name of the column that that contains the position where the variant is located.
        variant_col (str): Name of the column that contains the variant, in initial amino-acid/position/final
            amino-acid format (e.g. G12K).
        label_col (str): Name of the column that contains the label (pathogenic or benign) of the variant.
        gene_info_file (str): Name of the file that contains additional information about the gene: allele
            frequency, residue conservation and pLDDT.
    """

    data  = pd.read_csv(input_file)[[position_col, variant_col, label_col, 'Group']]
    data.drop_duplicates(subset=[variant_col], keep='first', inplace=True, ignore_index=True)
    data.columns = ['Position', 'Variant', 'MyLabel', 'Group']

    info = pd.read_csv(gene_info_file)

    # Amino Acid Types
    data['initial_aa'] = data['Variant'].apply(lambda x: x[0])
    data['final_aa'] = data['Variant'].apply(lambda x: x[-1])

    # Assign per aminoacid features
    data['topological_domain'] = data['Position'].apply(domain_map_KCNQ2)
    data['functional_domain'] = data['Position'].apply(functional_map_KCNQ2)
    data['d_charge'] = data.apply(lambda x: d_charge_map(charge_map(x.initial_aa), charge_map(x.final_aa)), axis=1)
    data['d_pol'] = data.apply(lambda x: d_pol_map(polar_map(x.initial_aa), polar_map(x.final_aa)), axis=1)
    data['d_aro'] = data.apply(lambda x: d_aro_map(aromatic_map(x.initial_aa), aromatic_map(x.final_aa)), axis=1)
    data['d_size'] = data['final_aa'].apply(lambda x: mw_map[x]) - data['initial_aa'].apply(lambda x: mw_map[x])
    data['d_vol'] = data['final_aa'].apply(lambda x: v_map[x]) - data['initial_aa'].apply(lambda x: v_map[x])
    data['d_pol_e'] = data['final_aa'].apply(lambda x: p_map[x]) - data['initial_aa'].apply(lambda x: p_map[x])
    data['d_ip_e'] = data['final_aa'].apply(lambda x: ip_e_map[x]) - data['initial_aa'].apply(lambda x: ip_e_map[x])
    data['d_hf_e'] = data['final_aa'].apply(lambda x: hf_e_map[x]) - data['initial_aa'].apply(lambda x: hf_e_map[x])
    data['d_msa'] = data['final_aa'].apply(lambda x: msa_map[x]) - data['initial_aa'].apply(lambda x: msa_map[x])
    data['d_hf'] = data['final_aa'].apply(lambda x: hf_map[x]) - data['initial_aa'].apply(lambda x: hf_map[x])

    # Evolutionary features
    condf = info[['Position', 'Conservation']]
    data = data.merge(condf, how='left', on='Position')

    # Structural features
    data['start_codon'] = np.where(data.Position == 1, 1, 0)

    struct_df = pd.DataFrame(np.vstack([np.arange(1, len(secondary_structure_KCNQ2) + 1), np.array(list(secondary_structure_KCNQ2))]).T, columns=['Residue', 'secondary_str'])
    struct_df['Residue'] = struct_df['Residue'].astype(int)
    struct_df['secondary_str'] = struct_df['secondary_str'].apply(lambda x: ss_map[x])
    data = data.merge(struct_df, how='left', left_on='Position', right_on='Residue').drop(columns=['Residue'])

    data['str_pos'] = data['Position'].apply(lambda x: str_pos_map_KCNQ2(x))
    
    # pLDDT
    plddt = info[['Position', 'pLDDT']]
    data = data.merge(plddt, how='left', on='Position')
    
    # VFI
    if vfi_sigma is not None:
        vfi_df = info[['Position', 'Freq']]
        
        vfi_df[f'VFI_{vfi_sigma}'] = gaussian_convolution_vfi(vfi_df['Freq'].to_numpy(), sigma=vfi_sigma, alpha=5e-7)
        data = data.merge(vfi_df, how='left', on='Position')
        data.drop(columns=['Freq'], inplace=True)
    
    # One hot encode
    excpt = ['Position', 'Variant', 'MyLabel', 'Group']
    str_cols = [col for col in data.columns if data[col].dtype == object and col not in excpt]

    cols = []

    for col in data.columns:
        
        if col in str_cols:
            tmp_df = pd.get_dummies(data[col], dtype=int)
            
            if col == 'initial_aa':
                tmp_df.columns = [col + '_i' for col in tmp_df.columns]
            elif col == 'final_aa':
                tmp_df.columns = [col + '_f' for col in tmp_df.columns]
            
            cols.append(tmp_df)
        else:
            cols.append(data[col])

    data_oh = pd.concat(cols, axis=1)
    
    for group, df in data_oh.groupby('Group'):

        df.drop(columns=['Group'], inplace=True)
        df.to_csv(os.path.join(output_dir, f'{group}_Featurized.csv'))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True, help='File contining variants to be featurized.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output file containing the featurized version of the input variant file.')
    parser.add_argument('-g', '--gene', type=str, choices={'KCNQ1', 'KCNQ2', 'KCNQ3', 'KCNQ4', 'KCNQ5'}, help='Name of the gene.')
    parser.add_argument('-s', '--vfi_sigma', type=int, required=True, help='Value of sigma used to define the VFI feature.')
    parser.add_argument('-v', '--variant_col', type=str, default='Variant', help='Name of the column in the input file that contains the names of the variants to featurize.')
    parser.add_argument('-p', '--position_col', type=str, default='Position', help='Name of the column in the input file that contains the positions of the variants to featurize.')
    parser.add_argument('-l', '--label_col', type=str, default='MyLabel', help='Name of the column in the input file that contains the labels of the variants to featurize.')
    parser.add_argument('-f', '--gene_info', type=str, default='data/gene_info/KCNQ2_info.csv', help='File containing evolutionary conservation, allele frequency and pLDDT score information for each position in the aminoacid sequence.')
    
    args = parser.parse_args()

    for name, file in vars(args).items():

        if name in ('output_dir', 'gene', 'vfi_sigma', 'variant_col', 'position_col', 'label_col'):
            continue

        assert os.path.isfile(file), f'File "{name}" does not exist or cannot be found.'

    featurize(args.input, args.output_dir, args.gene, args.vfi_sigma, args.position_col, args.variant_col,
              args.label_col, args.gene_info)
