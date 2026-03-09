

from Bio.PDB import DSSP, HSExposureCB, PPBuilder, is_aa, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
import torch
from Bio.SeqUtils import seq1

from Bio.PDB import FastMMCIFParser, is_aa
import foldseek_extract_pdb_features


import pandas as pd
from pathlib import Path, PurePath
import json
import argparse
import logging
import os

# MODEL
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'

# WARNING change accordingly depending on the execution path
model_dir = '3di_model'


def encoder_features(residues, virt_cb=(270, 0, 2), full_backbone=True):
    """
    Calculate 3D descriptors for each residue of a PDB file.
    """
    coords, valid_mask = foldseek_extract_pdb_features.get_atom_coordinates(residues, full_backbone=full_backbone)

    coords = foldseek_extract_pdb_features.move_CB(coords, virt_cb=virt_cb)

    partner_idx = foldseek_extract_pdb_features.find_nearest_residues(coords, valid_mask)
    features, valid_mask2 = foldseek_extract_pdb_features.calc_angles_forloop(coords, partner_idx, valid_mask)

    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([features, log_dist])

    return vae_features, valid_mask2

def discretize(centroids, x):
    return np.argmin(foldseek_extract_pdb_features.distance_matrix(x, centroids), axis=1)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file', help='mmCIF or PDB file')
    parser.add_argument('-conf_file', help='Configuration and parameters file', default=None)
    parser.add_argument('-out_dir', help='Output directory', default='.')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Set the logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    # fileHandler = logging.FileHandler("{}/info.log".format(args.out_dir))
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)

    # Load the config file
    # If not provided, set the path to "configuration.json", which is in the same folder of this Python file
    src_dir = str(PurePath(os.path.realpath(__file__)).parent)
    config_file = src_dir + "/configuration.json" if args.conf_file is None else args.configuration
    with open(config_file) as f:
        config = json.load(f)

    # Fix configuration paths (identified by the '_file' or '_dir' suffix in the field name)
    # If paths are relative it expects they refer to the absolute position of this file
    for k in config:
        if ('_file' in k or '_dir' in k) and k[0] != '/':
            config[k] = src_dir + '/' + config[k]

    # Start
    pdb_id = Path(args.pdb_file).stem
    logging.info("{} processing".format(pdb_id))
    ###################################################################################################################################
    # Calc_3di
    encoder = torch.load('{}/encoder.pt'.format(model_dir), weights_only=False)
    centroids = np.loadtxt('{}/states.txt'.format(model_dir))
    encoder.eval()

    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, args.pdb_file)

    data = []
    for chain in structure[0]:
        residues = list(chain.get_residues())

        feat, mask = encoder_features(residues)
        res_features = feat[mask]

        with torch.no_grad():
            res = encoder(torch.tensor(res_features, dtype=torch.float32)).detach().numpy()

        valid_states = discretize(centroids, res)

        states = np.full(len(mask), -1)
        states[mask] = valid_states

        for i, state in enumerate(states):
            if state != -1:
                data.append((pdb_id, chain.id, *residues[i].id[1:], seq1(residues[i].get_resname()), state, LETTERS[state],
                      *feat[i]))

    df3di = pd.DataFrame(data, columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter',
                                     'cos_phi_12', 'cos_phi_34', 'cos_phi_15', 'cos_phi_35',
                                     'cos_phi_14', 'cos_phi_23', 'cos_phi_13', 'd', 'seq_dist', 'log_dist'])
    
    ###################################################################################################################################
    # Ramachandran regions
    regions_matrix = []
    with open(config["rama_file"]) as f:
        for line in f:
            if line:
                regions_matrix.append([int(ele) for ele in line.strip().split()])

    # Atchely scales
    atchley_scale = {}
    with open(config["atchley_file"]) as f:
        next(f)
        for line in f:
            line = line.strip().split("\t")
            atchley_scale[line[0]] = line[1:]

    # Get valid residues
    residues = [residue for residue in structure[0].get_residues() if is_aa(residue) and residue.id[0] == ' ']
    if not residues:
        logging.warning("{} no valid residues error  (skipping prediction)".format(pdb_id))
        raise ValueError("no valid residues")

    # Calculate DSSP
    dssp = {}
    try:
        dssp = dict(DSSP(structure[0], args.pdb_file, dssp=config["dssp_file"]))
    except Exception:
        logging.warning("{} DSSP error".format(pdb_id))
        raise

    # Calculate Half Sphere Exposure
    hse = {}
    try:
        hse = dict(HSExposureCB(structure[0]))
    except Exception:
        logging.warning("{} HSE error".format(pdb_id))

    # Calculate ramachandran values
    rama_dict = {}  # {(chain_id, residue_id): [phi, psi, ss_class], ...}
    ppb = PPBuilder()
    for chain in structure[0]:
        for pp in ppb.build_peptides(chain):
            phi_psi = pp.get_phi_psi_list()  # [(phi_residue_1, psi_residue_1), ...]
            for i, residue in enumerate(pp):
                phi, psi = phi_psi[i]
                ss_class = None
                if phi is not None and psi is not None:
                    for x, y, width, height, ss_c, color in config["rama_ss_ranges"]:
                        if x <= phi < x + width and y <= psi < y + height:
                            ss_class = ss_c
                            break
                rama_dict[(chain.id, residue.id)] = [phi, psi, ss_class]

    # Generate contacts and add features
    data = []
    ns = NeighborSearch([atom for residue in residues for atom in residue])
    for residue_1, residue_2 in ns.search_all(config["distance_threshold"], level="R"):
        index_1 = residues.index(residue_1)
        index_2 = residues.index(residue_2)

        if abs(index_1 - index_2) >= config["sequence_separation"]:

            aa_1 = seq1(residue_1.get_resname())
            aa_2 = seq1(residue_2.get_resname())
            chain_1 = residue_1.get_parent().id
            chain_2 = residue_2.get_parent().id

            data.append((pdb_id,
                    chain_1,
                    *residue_1.id[1:],
                    aa_1,
                    *dssp.get((chain_1, residue_1.id), [None, None, None, None])[2:4],
                    *hse.get((chain_1, residue_1.id), [None, None])[:2],
                    *rama_dict.get((chain_1, residue_1.id), [None, None, None]),
                    *atchley_scale[aa_1],
                    chain_2,
                    *residue_2.id[1:],
                    aa_2,
                    *dssp.get((chain_2, residue_2.id), [None, None, None, None])[2:4],
                    *hse.get((chain_2, residue_2.id), [None, None])[:2],
                    *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                    *atchley_scale[aa_2]))

    if not data:
        logging.warning("{} no contacts error (skipping prediction)".format(pdb_id))
        raise ValueError("no contacts error (skipping prediction)")

    # TODO add sequence separation


    # Create a DataFrame and save to file
    df = pd.DataFrame(data, columns=['pdb_id',
                                     's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_ss3', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                                     't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_ss3', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']).round(3)
   
    # Merge dataframes
    merged_df = df.merge(df3di[['ch', 'resi', '3di_state']].rename(columns={'3di_state': 's_3di_state'}), left_on=['s_ch', 's_resi'], right_on=['ch', 'resi'], how='left').drop(columns=['ch', 'resi'])

    final_df = merged_df.merge(df3di[['ch', 'resi', '3di_state']].rename(columns={'3di_state': 't_3di_state'}), left_on=['t_ch', 't_resi'], right_on=['ch', 'resi'], how='left').drop(columns=['ch', 'resi'])

    col = final_df.pop('s_3di_state')
    final_df.insert(17, 's_3di_state', col)

    # MODEL
    # Preprocessing
    final_df.replace(' ', np.nan, inplace=True)

    # Deal with nan values
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = final_df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_columns:
        if final_df[col].isna().all():
            # If column is entirely NaN, fill with 0 or another strategy ??
            final_df[col] = 0

    final_df[numeric_columns] = imputer.fit_transform(final_df[numeric_columns])

    excluded_columns = ['pdb_id', 't_resi', 's_a5', 't_a5', 't_ch', 's_up', 's_down', 't_up', 't_down', 's_ss3', 't_ss3']
    categorical_columns = [col for col in final_df.columns if col not in excluded_columns and col not in numeric_columns]
    imputer_cat = SimpleImputer(strategy='most_frequent')
    final_df[categorical_columns] = imputer_cat.fit_transform(final_df[categorical_columns])

    # Label encoding for categorical columns
    label_encoder = LabelEncoder()

    for col in categorical_columns:
        final_df[col] = label_encoder.fit_transform(final_df[col])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(final_df[numeric_columns])
    final_df[numeric_columns] = features_scaled

    features_scaled = scaler.fit_transform(final_df[categorical_columns])
    final_df[categorical_columns] = features_scaled

    # Extract features
    features = final_df.drop(columns=excluded_columns)

    # Import model
    model = joblib.load("random_forest_model.joblib")

    # Predict
    y_pred = model.predict(features)

    labels = ['HBOND', 'VDW', 'IONIC', 'PICATION', 'PIHBOND', 'PIPISTACK', 'SSBOND']

    score_columns = pd.DataFrame(y_pred, columns=[f'{labels[i]}' for i in range(len(labels))])
    final_df = pd.concat([final_df.reset_index(drop=True), score_columns], axis=1)
    final_df.to_csv("{}/{}.tsv".format(args.out_dir, pdb_id), sep="\t", index=False)




