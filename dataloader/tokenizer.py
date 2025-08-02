from .utils import PAD, AA_VOCAB

#########################################
# Custom TCRBERT-Inspired Tokenizer
#########################################
class TCRBertTokenizer:
    def __init__(self, vocab=AA_VOCAB):
        self.vocab = vocab
        self.token2id = {token: idx for idx, token in enumerate(vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.pad_token = PAD
        self.pad_token_id = self.token2id[PAD]
    def tokenize(self, sequence):
        return list(sequence)
    def convert_tokens_to_ids(self, tokens):
        return [self.token2id.get(tok, self.token2id["X"]) for tok in tokens]
    def encode(self, sequence, max_len):
        tokens = self.tokenize(sequence.upper())
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_token] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return self.convert_tokens_to_ids(tokens)
    
    def save_tokenizations(self, sequences, output_csv):
        data = []
        for seq in sequences:
            tokens = self.tokenize(seq)
            token_ids = self.convert_tokens_to_ids(tokens)
            data.append({'sequence': seq, 'token_ids': token_ids})
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Tokenizations saved to {output_csv}")




