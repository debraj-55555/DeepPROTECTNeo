
import os, re, math, random, shlex, subprocess, tempfile, json, hashlib, argparse
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from peptides import Peptide
import torch
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
PAD = "-"  # padding
MASK = "."  # mask token
UNK = "?"   # unknown token
SEP = "|"   # separator
CLS = "*"   # classification token
AA_VOCAB = STANDARD_AA + "X" + PAD + MASK + UNK + SEP + CLS
VOCAB_SIZE = len(AA_VOCAB)

#########################################
# Extra Feature Functions: Peptide Embedding & Feature Names Plotting
#########################################
def generate_peptide_embedding(sequence, save_path="./cache"):
    embed_cache = os.path.join(save_path, "embed_cache")
    os.makedirs(embed_cache, exist_ok=True)
    seq_hash = hashlib.md5(sequence.encode("utf-8")).hexdigest()
    filename = os.path.join(embed_cache, f"peptide_{seq_hash}.npy")
    if os.path.exists(filename):
        return np.load(filename)
    try:
        peptide = Peptide(sequence)
        embedding = np.array(list(peptide.descriptors().values()))
        feature_names = list(peptide.descriptors().keys())
        fn_csv = os.path.join(save_path, "peptide_feature_names.csv")
        if not os.path.exists(fn_csv):
            pd.DataFrame({"feature_names": feature_names}).to_csv(fn_csv, index=False)
            print(f"Peptide feature names saved to {fn_csv}")
    except Exception as e:
        print(f"Error generating peptide embedding: {e}")
        embedding = np.random.rand(102)
    np.save(filename, embedding)
    return embedding



#########################################
# ANARCI Alignment Integration with Caching
#########################################
def get_numbering(seqs):
    """
    Get the IMGT numbering of CDR3 using ANARCI, with caching in ./cache/.
    Only prints whether it’s loading from cache or running ANARCI.
    Suppresses pandas SettingWithCopyWarning.
    """
    # Suppress pandas SettingWithCopyWarning
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    # Normalize sequences: strip whitespace, remove empties
    cleaned = [s.strip() for s in seqs]
    unique_seqs = sorted({s for s in cleaned if s})

    # Compute hash key for this set
    key = ";".join(unique_seqs)
    key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()

    # Prepare cache directory
    cache_dir = os.path.join(os.getcwd(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'anarci_aligned.csv')

    # Try loading from cache
    if os.path.exists(cache_file):
        df_cache = pd.read_csv(cache_file)
        if 'hash' in df_cache.columns and \
           df_cache['hash'].eq(key_hash).all():
            print("Loading cached ANARCI alignments.")
            mapping = dict(zip(df_cache['cdr3'], df_cache['cdr3_align']))
            return [mapping.get(s, '') for s in cleaned]

    # Not cached => run ANARCI once
    print("Running ANARCI…")
    fasta_path = os.path.join(cache_dir, f"tmp_faketcr_{key_hash}.fasta")
    align_prefix = os.path.join(cache_dir, f"tmp_align_{key_hash}")
    out_csv = f"{align_prefix}_B.csv"

    # Write fake TCR FASTA
    template = [
        "GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGTTDQGEVPNGYNVSRSTIEDFPLRLLSAAPSQTSVYF",
        "GEGSRLTVL"
    ]
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(unique_seqs):
            f.write(f">{i}\n{template[0]}{seq}{template[1]}\n")

    # Run ANARCI command
    cmd = f"ANARCI -i {fasta_path} -o {align_prefix} --csv -p 20"
    os.system(cmd)

    # Load alignment output
    if not os.path.exists(out_csv):
        raise FileNotFoundError("ANARCI failed to produce output CSV")
    df = pd.read_csv(out_csv)

    # Collect aligned CDR3 (columns 104–118)
    cols = ['104','105','106','107','108','109','110','111','111A','111B',
            '112C','112B','112A','112','113','114','115','116','117','118']
    aligned = []
    for idx in range(len(df)):
        chars = [df.at[idx, c] if c in df.columns else '-' for c in cols]
        aligned.append(''.join(chars))

    # Build and save cache table
    df_cache = pd.DataFrame({
        'hash': [key_hash] * len(unique_seqs),
        'cdr3': unique_seqs,
        'cdr3_align': aligned
    })
    df_cache.to_csv(cache_file, index=False)

    # Return aligned list in original order
    mapping = dict(zip(df_cache['cdr3'], df_cache['cdr3_align']))
    return [mapping.get(s, '') for s in cleaned]


def plot_peptide_feature_names(csv_path, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(df)), np.arange(len(df)), color='skyblue')
    plt.yticks(range(len(df)), df["feature_names"])
    plt.xlabel("Feature Index")
    plt.title("Peptide Descriptor Feature Names")
    plt.tight_layout()
    out_path = os.path.join(save_path, "peptide_feature_names.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Peptide feature names plot saved to {out_path}")
# (You may substitute with your full AAfea_phy dictionary.)
AAfea_phy_dict = {
    "A": [0.83, 0.36, 0.02, 0, 0, 1, 1.19, 0.86, 0.81, 0, 4.6, 0.92, 0.5, -1.404, 1.08, -0.44, 1.16, -0.06, -0.67, -0.4, 11.5, 6, 0.71, 1.08, 32, 5, -0.27, -0.04],
    "R": [0.93, -0.52, -0.42, 4, 1, 1.7, 1, 1.15, 0.85, 1, 6.5, 0.93, 0.8, -0.921, 0.93, -0.13, 1.72, -0.84, 3.89, -0.59, 14.28, 10.76, 1.09, 0.976, -95, -57, 2, 0.07],
    "N": [0.89, -0.9, -0.77, 2, 0, 1, 0.94, 0.6, 0.62, 0, 5.9, 0.6, 0.8, -1.178, 1.05, 0.05, 1.97, -0.48, 2.27, -0.92, 12.82, 5.41, 0.95, 1.197, -73, -77, 0.61, 0.13],
    "D": [0.54, -1.09, -1.04, 1, 0, 0.7, 1.07, 0.66, 0.71, -1, 5.7, 0.48, -8.2, -1.162, 0.86, -0.2, 2.66, -0.8, 1.57, -1.31, 11.68, 2.77, 1.43, 1.266, -29, 45, 0.5, 0.19],
    "C": [1.19, 0.7, 0.77, 0, 0, 1, 0.95, 0.91, 1.17, 0, -1, 1.16, -6.8, -1.365, 1.22, 0.13, 0.5, 1.36, -2, 0.17, 13.46, 5.05, 0.65, 0.733, 182, 224, -0.23, -0.38],
    "Q": [1.1, -1.05, -1.1, 2, 0, 1, 1.32, 1.11, 0.98, 0, 6.1, 0.95, -4.8, -1.116, 0.95, -0.58, 3.87, -0.73, 2.12, -0.91, 14.45, 5.65, 0.87, 1.05, -95, -67, 1, 0.14],
    "E": [0.37, -0.83, -1.14, 1, 0, 0.7, 1.64, 0.37, 0.53, -1, 5.6, 0.61, -16.9, -1.163, 1.09, -0.28, 2.4, -0.77, 1.78, -1.22, 13.57, 3.22, 1.19, 1.085, -74, -8, 0.33, 0.23],
    "G": [0.75, -0.82, -0.8, 0, 0, 1.5, 0.6, 0.86, 0.88, 0, 7.6, 0.61, 0, -1.364, 0.85, 0.08, 1.63, -0.41, 0, -0.67, 3.4, 5.97, 1.07, 1.104, -22, -47, -0.22, 0.09],
    "H": [0.87, 0.16, 0.26, 1, 1, 1, 1.03, 1.07, 0.92, 0, 4.5, 0.93, -3.5, -1.215, 1.02, 0.09, 0.86, 0.49, 1.09, -0.64, 13.69, 7.59, 1.13, 0.906, -25, -50, 0.37, -0.04],
    "I": [1.6, 2.17, 1.81, 0, 0, 1, 1.12, 1.17, 1.48, 0, 2.6, 1.81, 13.9, -1.189, 0.98, -0.04, 0.57, 1.31, -3.02, 1.25, 21.4, 6.02, 1.05, 0.583, 106, 83, -0.8, -0.34],
    "L": [1.3, 1.18, 1.14, 0, 0, 1, 1.18, 1.28, 1.24, 0, 3.25, 1.3, 8.8, -1.315, 1.04, -0.12, 0.51, 1.21, -3.02, 1.22, 21.4, 5.98, 0.84, 0.789, 104, 82, -0.44, -0.37],
    "K": [0.74, -0.56, -0.41, 2, 1, 1.7, 1, 1.15, 0.85, 1, 6.5, 0.93, 0.8, -0.921, 0.93, -0.13, 1.72, -0.84, 3.89, -0.59, 14.28, 10.76, 1.09, 0.976, -95, -57, 2, 0.07],
    "M": [1.05, 1.21, 1, 0, 0, 1, 1.49, 1.15, 1.05, 0, 1.4, 1.19, 4.8, -1.303, 1.11, -0.21, 0.4, 1.27, -1.67, 1.02, 16.25, 5.74, 0.8, 0.812, 82, 83, -0.31, -0.3],
    "F": [1.38, 1.01, 1.35, 0, 0, 1, 1.02, 1.34, 1.2, 0, 3.2, 1.25, 13.2, -1.135, 0.96, -0.13, 0.43, 1.27, -3.24, 1.92, 19.8, 5.48, 0.95, 0.685, 132, 117, -0.55, -0.38],
    "P": [0.55, -0.06, -0.09, 0, 0, 0.1, 0.68, 0.61, 0.61, 0, 7, 0.4, 6.1, -1.236, 0.91, -0.48, 2.04, 0, -1.75, -0.49, 17.43, 6.3, 1.7, 1.412, -82, -103, 0.36, 0.19],
    "S": [0.75, -0.6, -0.97, 1, 0, 1, 0.81, 0.91, 0.92, 0, 5.25, 0.82, 1.2, -1.297, 0.95, 0.27, 1.61, -0.5, 0.1, -0.55, 9.47, 5.68, 0.65, 0.987, -34, -41, 0.17, 0.12],
    "T": [1.19, -1.2, -0.77, 1, 0, 1, 0.85, 1.14, 1.18, 0, 4.8, 1.12, 2.7, -1.252, 1.15, 0.47, 1.48, -0.27, -0.42, -0.28, 15.77, 5.66, 0.086, 0.784, 20, 79, 0.18, 0.03],
    "W": [1.37, 1.31, 1.71, 1, 0, 1, 1.18, 1.13, 1.18, 0, 4, 1.54, 14.9, -1.03, 1.17, -0.22, 0.75, 0.88, -2.86, 0.5, 21.67, 5.89, 1.25, 0.755, 118, 130, 0.05, -0.33],
    "Y": [1.47, 1.05, 1.11, 1, 0, 1, 0.77, 1.37, 1.23, 0, 4.35, 1.53, 6.1, -1.03, 0.8, -0.11, 1.72, 0.33, 0.98, 1.67, 18.03, 5.66, 0.85, 0.665, 44, 27, 0.48, -0.29],
    "V": [1.7, 1.21, 1.13, 0, 0, 1, 0.74, 1.31, 1.66, 0, 3.4, 1.81, 2.7, -1.254, 1.03, 0.06, 0.59, 1.09, -2.18, 0.91, 21.57, 5.96, 1.12, 0.546, 113, 117, -0.65, -0.29],
    PAD: [0.0] * 28,
    "X": [0.0] * 28
}
#########################################
# BLOSUM62 and Physicochemical Encodings
#########################################
def get_blosum62_dict() -> dict:
    BLOSUM62_STR = """A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,*
A,4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,-4
R,-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,-4
N,-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,-4
D,-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,-4
C,0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4
Q,-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,-4
E,-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
G,0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,-4
H,-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,-4
I,-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,-4
L,-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,-4
K,-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,-4
M,-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,-4
F,-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,-4
P,-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,-4
S,1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,-4
T,0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,-4
W,-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,-4
Y,-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,-4
V,0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,-4
B,-2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,-4
Z,-1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
X,0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,-4
*,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1"""
    lines = BLOSUM62_STR.strip().splitlines()
    header = [x.strip() for x in lines[0].split(",")]
    blosum_dict = {}
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        letter = parts[0]
        vec = [int(x) for x in parts[1:]]
        blosum_dict[letter] = torch.tensor(vec, dtype=torch.float)
    return blosum_dict

BLOSUM62_DICT = get_blosum62_dict()

def encode_sequence_blosum(seq: str, max_len: int) -> torch.Tensor:
    seq = seq.upper()
    if len(seq) < max_len:
        seq = seq.ljust(max_len, PAD)
    else:
        seq = seq[:max_len]
    embeds = [BLOSUM62_DICT.get(aa, BLOSUM62_DICT["X"]) for aa in seq]
    return torch.stack(embeds)

def encode_sequence_physico(seq: str, max_len: int) -> torch.Tensor:
    seq = seq.upper()
    if len(seq) < max_len:
        seq = seq.ljust(max_len, PAD)
    else:
        seq = seq[:max_len]
    feats = [torch.tensor(AAfea_phy_dict.get(aa, AAfea_phy_dict["X"]), dtype=torch.float) for aa in seq]
    return torch.stack(feats)