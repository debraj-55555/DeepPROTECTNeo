from .model import BiLSTMEncoderModule

from .model import PositionalEncodingModule, FeatureProjection, AttentionPooling, DeepCNNAttnProjector, GatedFusion, CrossAttentionBlock, ExtraSelfAttention, EnhancedRetNetBlock
import torch
import numpy as np
import torch.nn as nn
import random
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
PAD = "-"  # padding
MASK = "."  # mask token
UNK = "?"   # unknown token
SEP = "|"   # separator
CLS = "*"   # classification token
AA_VOCAB = STANDARD_AA + "X" + PAD + MASK + UNK + SEP + CLS
VOCAB_SIZE = len(AA_VOCAB)
#########################################
# Reproducibility
#########################################
def set_modelseed(modelseed=42):
    random.seed(modelseed)
    np.random.seed(modelseed)
    torch.manual_seed(modelseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(modelseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




#########################################
# Main Model with Attention Weight Support
#########################################
class DeepProtectNeo(nn.Module):
    def __init__(self, d_model, n_heads, phys_dim, skip_dim, sa_n_heads, cross_attn_heads,
                 modelseed=42, bilstm_layers=1, bilstm_dropout=0.0, max_tcr_len=20, max_pep_len=11,
                 use_ablation=True, use_cross_attn=True, learned_pos=False,
                 cnn_dim=128, cnn_blocks=16, cnn_ch=64, cnn_ks=5, cnn_attn_heads=16):
        super().__init__()
        
        set_modelseed(modelseed)
        self.use_ablation = use_ablation
        self.use_cross_attn = use_cross_attn

        # Token branch
        self.tcr_encoder = BiLSTMEncoderModule(VOCAB_SIZE, embed_dim=d_model, d_model=d_model,
                                              num_layers=bilstm_layers, dropout=bilstm_dropout)
        self.pep_encoder = BiLSTMEncoderModule(VOCAB_SIZE, embed_dim=d_model, d_model=d_model,
                                              num_layers=bilstm_layers, dropout=bilstm_dropout)
        self.tcr_pos_enc = PositionalEncodingModule(d_model, max_len=max_tcr_len,
                                                   dropout_rate=bilstm_dropout, learned=learned_pos)
        self.pep_pos_enc = PositionalEncodingModule(d_model, max_len=max_pep_len,
                                                   dropout_rate=bilstm_dropout, learned=learned_pos)

        if use_ablation:
            concat_dim = d_model + d_model + phys_dim
            self.tcr_blos_proj = FeatureProjection(24, d_model)
            self.pep_blos_proj = FeatureProjection(24, d_model)
            self.tcr_phys_proj = FeatureProjection(28, phys_dim)
            self.pep_phys_proj = FeatureProjection(28, phys_dim)
            self.attn_proj = nn.Linear(concat_dim, concat_dim)
            self.tcr_ablation_pool = AttentionPooling(concat_dim, max_tcr_len, concat_dim//16)
            self.pep_ablation_pool = AttentionPooling(concat_dim, max_pep_len, concat_dim//16)
        else:
            self.tcr_pool = AttentionPooling(d_model, max_tcr_len, d_model//24)
            self.pep_pool = AttentionPooling(d_model, max_pep_len, d_model//30)

        # Skip branch
        self.tcr_cnn = DeepCNNAttnProjector(input_dim=102, output_dim=cnn_dim, num_blocks=cnn_blocks,
                                           channels=cnn_ch, kernel_size=cnn_ks, attn_heads=cnn_attn_heads)
        self.pep_cnn = DeepCNNAttnProjector(input_dim=102, output_dim=cnn_dim, num_blocks=cnn_blocks,
                                           channels=cnn_ch, kernel_size=cnn_ks, attn_heads=cnn_attn_heads)
        
        self.tcr_skip = nn.Linear(cnn_dim, skip_dim)
        self.pep_skip = nn.Linear(cnn_dim, skip_dim)
        self.tcr_skip_extra1 = nn.Linear(cnn_ch, skip_dim)
        self.tcr_skip_extra2 = nn.Linear(cnn_ch, skip_dim)
        self.pep_skip_extra1 = nn.Linear(cnn_ch, skip_dim)
        self.pep_skip_extra2 = nn.Linear(cnn_ch, skip_dim)
        
        self.skip_gate_t = nn.Sequential(nn.Linear(skip_dim * 3, skip_dim), nn.ReLU())
        self.skip_gate_p = nn.Sequential(nn.Linear(skip_dim * 3, skip_dim), nn.ReLU())

        # Fusion and cross-attention
        self.gated_fusion = GatedFusion(d_model, skip_dim, d_model)
        
        if use_cross_attn:
            self.cross_attn = CrossAttentionBlock(d_model, cross_attn_heads)

        fused_dim = 2*d_model + (d_model if use_cross_attn else 0)
        self.extra_SA = ExtraSelfAttention(fused_dim, sa_n_heads)
        self.retnet = EnhancedRetNetBlock(fused_dim, num_layers=24)
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 2)
        )

    def forward(self, tcr_tokens, pep_tokens, tcr_blos, pep_blos, tcr_phys, pep_phys, 
                tcr_hand, pep_hand, return_attention=False):
        device = next(self.parameters()).device
        tcr_tokens = tcr_tokens.to(device)
        pep_tokens = pep_tokens.to(device)
        tcr_blos = tcr_blos.to(device)
        pep_blos = pep_blos.to(device)
        tcr_phys = tcr_phys.to(device)
        pep_phys = pep_phys.to(device)
        tcr_hand = tcr_hand.to(device)
        pep_hand = pep_hand.to(device)

        # Token branch feature extraction
        if self.use_ablation:
            t_lstm = self.tcr_pos_enc(self.tcr_encoder(tcr_tokens))
            p_lstm = self.pep_pos_enc(self.pep_encoder(pep_tokens))
            t_concat = torch.cat([
                t_lstm,
                self.tcr_blos_proj(tcr_blos).unsqueeze(1).expand_as(t_lstm),
                self.tcr_phys_proj(tcr_phys).unsqueeze(1).expand_as(t_lstm)
            ], dim=-1)
            p_concat = torch.cat([
                p_lstm,
                self.pep_blos_proj(pep_blos).unsqueeze(1).expand_as(p_lstm),
                self.pep_phys_proj(pep_phys).unsqueeze(1).expand_as(p_lstm)
            ], dim=-1)
            t_feat = self.tcr_ablation_pool(self.attn_proj(t_concat))
            p_feat = self.pep_ablation_pool(self.attn_proj(p_concat))
        else:
            t_lstm = self.tcr_pos_enc(self.tcr_encoder(tcr_tokens))
            p_lstm = self.pep_pos_enc(self.pep_encoder(pep_tokens))
            t_feat = self.tcr_pool(t_lstm)
            p_feat = self.pep_pool(p_lstm)

        # Skip branch feature extraction
        t_final, t_skips = self.tcr_cnn(tcr_hand)
        p_final, p_skips = self.pep_cnn(pep_hand)
        
        t_skip_main = self.tcr_skip(t_final)
        p_skip_main = self.pep_skip(p_final)
        
        t_skip1 = self.tcr_skip_extra1(t_skips[-3].mean(dim=2))
        t_skip2 = self.tcr_skip_extra2(t_skips[-2].mean(dim=2))
        p_skip1 = self.pep_skip_extra1(p_skips[-3].mean(dim=2))
        p_skip2 = self.pep_skip_extra2(p_skips[-2].mean(dim=2))
        
        t_skip_stack = torch.cat([t_skip_main, t_skip1, t_skip2], dim=1)
        t_skip_combined = self.skip_gate_t(t_skip_stack)
        p_skip_stack = torch.cat([p_skip_main, p_skip1, p_skip2], dim=1)
        p_skip_combined = self.skip_gate_p(p_skip_stack)

        # Gated fusion
        t_fused = self.gated_fusion(t_feat, t_skip_combined)
        p_fused = self.gated_fusion(p_feat, p_skip_combined)

        # Cross-attention with attention weight extraction
        if self.use_cross_attn:
            if return_attention:
                x_fused, attention_weights = self.cross_attn(t_fused, p_fused, return_attention=True)
                fusion = torch.cat([t_fused, p_fused, x_fused], dim=1)
            else:
                x_fused = self.cross_attn(t_fused, p_fused, return_attention=False)
                fusion = torch.cat([t_fused, p_fused, x_fused], dim=1)
                attention_weights = None
        else:
            fusion = torch.cat([t_fused, p_fused], dim=1)
            attention_weights = None

        # Final processing
        fusion = self.retnet(fusion) + self.extra_SA(fusion)
        output = self.classifier(fusion)
        
        if return_attention:
            return output, attention_weights
        else:
            return output