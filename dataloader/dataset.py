
 
from torch.utils.data import Dataset
from .utils import generate_peptide_embedding, encode_sequence_blosum, encode_sequence_physico, get_numbering
from .tokenizer import TCRBertTokenizer
import torch
import pandas as pd
#########################################
# Global Constants & Dummy Physico Dictionary
#########################################



#########################################
# Dataset Class (No MUSCLE alignment)
#########################################
class DeepProtectNeoDataset(Dataset):
    def __init__(self, sequences, labels, max_tcr_len=20, max_pep_len=11, align=False, tokenizer=None, token_csv_path=None):
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.tokenizer = tokenizer if tokenizer is not None else TCRBertTokenizer()
        self.token_csv_path = token_csv_path
        self.data = self.__init_data(sequences, labels, align)
        """
        if token_csv_path is not None:
            self.save_tokenizations()

        if align:
            sequences.loc[:, 'TCR'] = get_numbering(sequences['TCR'])
        self.data = self.__init_data(sequences, labels, align)
        """

        if token_csv_path is not None:
            self.save_tokenizations()
        
        if align:
            import re
            valid_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
            sequences = sequences[sequences['TCR'].apply(lambda x: bool(valid_pattern.match(x)))]
            sequences.loc[:, 'TCR'] = get_numbering(sequences['TCR'])
       
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __init_data(self, sequences, labels, align):
        data_list = []
        arr = sequences.values
        labels = labels.values.astype('int')
        for idx, sequences in enumerate(arr):
            tcr, pep = sequences
            # Skip samples with empty TCR or peptide
            if len(tcr.strip()) == 0 or len(pep.strip()) == 0:
                continue
            tcr_tokens = torch.tensor(self.tokenizer.encode(tcr, self.max_tcr_len)).long()
            pep_tokens = torch.tensor(self.tokenizer.encode(pep, self.max_pep_len)).long()
            tcr_blos = encode_sequence_blosum(tcr, self.max_tcr_len)
            pep_blos = encode_sequence_blosum(pep, self.max_pep_len)
            tcr_phys = encode_sequence_physico(tcr, self.max_tcr_len)
            pep_phys = encode_sequence_physico(pep, self.max_pep_len)
            tcr_hand = torch.tensor(generate_peptide_embedding(tcr)).float()
            pep_hand = torch.tensor(generate_peptide_embedding(pep)).float()
            data_list.append((tcr_tokens, pep_tokens, tcr_blos, pep_blos, tcr_phys, pep_phys, tcr_hand, pep_hand, labels[idx]))
        return data_list
    def save_tokenizations(self):
        tcr_seqs, pep_seqs, tcr_tokens_list, pep_tokens_list = [], [], [], []
        for item in self.data:
            tcr_tok, pep_tok, _, _, _, _, _, _, _ = item
            tcr_seq = "".join([self.tokenizer.id2token[int(i)] for i in tcr_tok])
            pep_seq = "".join([self.tokenizer.id2token[int(i)] for i in pep_tok])
            tcr_seqs.append(tcr_seq)
            pep_seqs.append(pep_seq)
            tcr_tokens_list.append(tcr_tok.tolist())
            pep_tokens_list.append(pep_tok.tolist())
        df_tokens = pd.DataFrame({
            "TCR_sequence": tcr_seqs,
            "TCR_token_ids": tcr_tokens_list,
            "Peptide_sequence": pep_seqs,
            "Peptide_token_ids": pep_tokens_list
        })
        df_tokens.to_csv(self.token_csv_path, index=False)
        print(f"Tokenizations saved to {self.token_csv_path}")
class DeepProtectNeoDatasetTest(Dataset):
    def __init__(self, sequences, labels, max_tcr_len=20, max_pep_len=11, align=False, tokenizer=None, token_csv_path=None):
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.tokenizer = tokenizer if tokenizer is not None else TCRBertTokenizer()
        self.token_csv_path = token_csv_path
        self.data = self.__init_data(sequences, labels, align)
        """
        if token_csv_path is not None:
            self.save_tokenizations()

        if align:
            sequences.loc[:, 'TCR'] = get_numbering(sequences['TCR'])
        self.data = self.__init_data(sequences, labels, align)
        """

        if token_csv_path is not None:
            self.save_tokenizations()
        
        if align:
            import re
            valid_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
            sequences = sequences[sequences['TCR'].apply(lambda x: bool(valid_pattern.match(x)))]
            sequences.loc[:, 'TCR'] = get_numbering(sequences['TCR'])
       
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __init_data(self, sequences, labels, align):
        data_list = []
        arr = sequences.values
        labels = labels.values.astype('int')
        for idx, sequences in enumerate(arr):
            tcr, pep = sequences
            # Skip samples with empty TCR or peptide
            if len(tcr.strip()) == 0 or len(pep.strip()) == 0:
                continue
            tcr_tokens = torch.tensor(self.tokenizer.encode(tcr, self.max_tcr_len)).long()
            pep_tokens = torch.tensor(self.tokenizer.encode(pep, self.max_pep_len)).long()
            tcr_blos = encode_sequence_blosum(tcr, self.max_tcr_len)
            pep_blos = encode_sequence_blosum(pep, self.max_pep_len)
            tcr_phys = encode_sequence_physico(tcr, self.max_tcr_len)
            pep_phys = encode_sequence_physico(pep, self.max_pep_len)
            tcr_hand = torch.tensor(generate_peptide_embedding(tcr)).float()
            pep_hand = torch.tensor(generate_peptide_embedding(pep)).float()
            data_list.append((tcr, pep, tcr_tokens, pep_tokens, tcr_blos, pep_blos, tcr_phys, pep_phys, tcr_hand, pep_hand, labels[idx]))
        return data_list
    def save_tokenizations(self):
        tcr_seqs, pep_seqs, tcr_tokens_list, pep_tokens_list = [], [], [], []
        for item in self.data:
            tcr_tok, pep_tok, _, _, _, _, _, _, _ = item
            tcr_seq = "".join([self.tokenizer.id2token[int(i)] for i in tcr_tok])
            pep_seq = "".join([self.tokenizer.id2token[int(i)] for i in pep_tok])
            tcr_seqs.append(tcr_seq)
            pep_seqs.append(pep_seq)
            tcr_tokens_list.append(tcr_tok.tolist())
            pep_tokens_list.append(pep_tok.tolist())
        df_tokens = pd.DataFrame({
            "TCR_sequence": tcr_seqs,
            "TCR_token_ids": tcr_tokens_list,
            "Peptide_sequence": pep_seqs,
            "Peptide_token_ids": pep_tokens_list
        })
        df_tokens.to_csv(self.token_csv_path, index=False)
        print(f"Tokenizations saved to {self.token_csv_path}")