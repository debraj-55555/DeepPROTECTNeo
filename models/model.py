
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#########################################
# Model Components
#########################################
class PositionalEncodingModule(nn.Module):
    def __init__(self, d_model, max_len=512, dropout_rate=0.1, learned=False):
        super(PositionalEncodingModule, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.learned = learned
        if learned:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
    def forward(self, x):
        if self.learned:
            indices = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            pos_emb = self.pe(indices)
            x = x + pos_emb
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BiLSTMEncoderModule(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, num_layers=2, dropout=0.2):
        super(BiLSTMEncoderModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncodingModule(embed_dim, dropout_rate=dropout)
        hidden_dim = d_model // 2
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_enc(emb)
        lstm_out, _ = self.lstm(emb)
        return lstm_out

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, step_dim, context_dim):
        super(AttentionPooling, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feature_dim, context_dim))
        self.bias = nn.Parameter(torch.Tensor(step_dim, context_dim))
        self.context_vector = nn.Parameter(torch.Tensor(context_dim, 1))
        self.tanh = nn.Tanh()
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.context_vector, a=math.sqrt(5))
    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.weight) + self.bias)
        scores = torch.matmul(e, self.context_vector)
        attn_weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        return weighted_sum

class FeatureProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class DeepCNNProjector(nn.Module):
    def __init__(self, input_dim=102, output_dim=16, num_blocks=3, channels=32, kernel_size=3):
        super(DeepCNNProjector, self).__init__()
        layers = []
        in_channels = 1
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(ResBlock(in_channels, channels, kernel_size, stride))
            in_channels = channels
        layers.append(nn.Conv1d(channels, output_dim, kernel_size=1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.mean(dim=-1)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V, attn_mask=None):
        batch_size = Q.size(0)
        Q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        context, attn = ScaledDotProductAttention(self.d_k)(Q_s, K_s, V_s, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.fc(context)
        return output, attn

#########################################
# Rotary Positional Embedding for Cross Attention
#########################################
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin", sinusoid_inp.sin().unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        cos = self.cos[:, :, :x.size(2), :]
        sin = self.sin[:, :, :x.size(2), :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(RotaryMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.d_k)
    def forward(self, Q, K, V, attn_mask=None):
        batch_size = Q.size(0)
        Q_proj = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K_proj = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V_proj = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        Q_proj = self.rotary(Q_proj)
        K_proj = self.rotary(K_proj)
        scores = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V_proj)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.fc(context)
        return output, attn

class CrossAttentionBlock(nn.Module):
    def __init__(self, token_branch_dim, cross_n_heads):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = RotaryMultiHeadAttention(token_branch_dim, cross_n_heads)
    
    def forward(self, tcr_feat, pep_feat, return_attention=False):
        tcr_feat = tcr_feat.unsqueeze(1)
        pep_feat = pep_feat.unsqueeze(1)
        
        cross_tcr, cross_attn_tcr = self.cross_attn(tcr_feat, pep_feat, pep_feat)
        cross_pep, cross_attn_pep = self.cross_attn(pep_feat, tcr_feat, tcr_feat)
        
        cross_tcr = cross_tcr.squeeze(1)
        cross_pep = cross_pep.squeeze(1)
        cross_fused = (cross_tcr + cross_pep) / 2
        
        if return_attention:
            attention_weights = {
                'tcr_to_pep': cross_attn_tcr,
                'pep_to_tcr': cross_attn_pep
            }
            return cross_fused, attention_weights
        
        return cross_fused




#########################################
# Gated Fusion Module
#########################################
class GatedFusion(nn.Module):
    def __init__(self, x1_dim, x2_dim, fusion_dim):
        super(GatedFusion, self).__init__()
        self.proj_x1 = nn.Linear(x1_dim, fusion_dim)
        self.proj_x2 = nn.Linear(x2_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        x1_proj = self.proj_x1(x1)
        x2_proj = self.proj_x2(x2)
        combined = torch.cat([x1_proj, x2_proj], dim=1)
        gate_weight = self.gate(combined)
        fused = gate_weight * x1_proj + (1 - gate_weight) * x2_proj
        return fused



#########################################
# Additional Modules: Extra SA and Conv1d Block
#########################################
class ExtraSelfAttention(nn.Module):
    def __init__(self, input_dim, sa_n_heads):
        super(ExtraSelfAttention, self).__init__()
        self.SA_layer = MultiHeadAttention(input_dim, sa_n_heads)
    def forward(self, x):
        # x is (B, D); unsqueeze to (B, 1, D)
        x_unsq = x.unsqueeze(1)
        sa_out, _ = self.SA_layer(x_unsq, x_unsq, x_unsq, attn_mask=None)
        return sa_out.squeeze(1)

class Conv1dBlock(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(4, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.05)
        )
    def forward(self, x):
        # x is (B, F); reshape to (B, 1, F)
        x_unsq = x.unsqueeze(1)
        conv_out = self.conv1d(x_unsq)  # (B, d_model, F)
        # Global average pooling over the F dimension:
        pooled = conv_out.mean(dim=2)  # (B, d_model)
        return pooled
class ExtraSelfAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(input_dim, n_heads)
    def forward(self, x):
        out,_ = self.mha(x.unsqueeze(1),x.unsqueeze(1),x.unsqueeze(1))
        return out.squeeze(1)






class AttentionPool(nn.Module):
    def __init__(self, feat_dim, seq_len, ctx_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, ctx_dim))
        self.b = nn.Parameter(torch.Tensor(seq_len, ctx_dim))
        self.v = nn.Parameter(torch.Tensor(ctx_dim, 1))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, L, D)
        e = torch.tanh(x @ self.W + self.b)
        a = torch.softmax(e @ self.v, dim=1)  # (B, L,1)
        return torch.sum(x * a, dim=1)       # (B, D)

# ─── DeepCNN + Channel + Self‐Attention ───────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, C, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(C, C//reduction, bias=True),
            nn.ReLU(True),
            nn.Linear(C//reduction, C, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (B, C, L)
        y = self.avg(x).view(x.size(0), x.size(1))
        y = self.fc(y).view(x.size(0), x.size(1), 1)
        return x * y

class DeepCNNAttnProjector(nn.Module):
    def __init__(self, input_dim=102, output_dim=16, num_blocks=3, channels=32,
                 kernel_size=3, attn_heads=3, use_self_attn: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = 1
        for i in range(num_blocks):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, channels, kernel_size,
                          padding=(kernel_size-1)//2,
                          stride=2 if i==0 else 1),
                nn.BatchNorm1d(channels),
                nn.GELU()
            ))
            in_ch = channels
        self.final_conv = nn.Conv1d(channels, output_dim, 1)
        self.chan_attn = ChannelAttention(output_dim)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(output_dim, attn_heads, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,F)
        skips = []          # collect intermediate skip features
        for layer in self.layers:
            x = layer(x)
            skips.append(x)  # collect after each block
        x = self.final_conv(x)
        skips.append(x)      # final conv output
        x = self.chan_attn(x)
        
        if self.use_self_attn:
            # attend across the length dimension
            seq, _ = self.self_attn(x.transpose(1,2), x.transpose(1,2), x.transpose(1,2))
            seq = seq.transpose(1,2)     # (B,out,L)
            out = (x + seq).mean(dim=2)  # pooled output with attention
        else:
            # No self-attention, just use x directly
            out = x.mean(dim=2)  # pooled output without attention
            
        return out, skips  # (B,out)



# ─── Fusion & Classifier ────────────────────────────────────────────────────
class GatedFusion(nn.Module):
    def __init__(self, d1, d2, out):
        super().__init__()
        self.p1 = nn.Linear(d1, out)
        self.p2 = nn.Linear(d2, out)
        self.g  = nn.Sequential(nn.Linear(out*2, out), nn.Sigmoid())

    def forward(self, x1, x2):
        y1, y2 = self.p1(x1), self.p2(x2)
        gate   = self.g(torch.cat([y1,y2], dim=1))
        return gate*y1 + (1-gate)*y2

class ExtraSA(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.mha = RotaryMultiHeadAttention(d_model, heads)
    def forward(self, x):
        return self.mha(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1)).squeeze(1)

class EnhancedRetNet(nn.Module):
    def __init__(self, d_model, layers=12):
        super().__init__()
        self.layers = nn.ModuleList()
        self.scales = nn.ParameterList()
        for _ in range(layers):
            self.layers.append(
                nn.Sequential(nn.LayerNorm(d_model),
                              nn.Linear(d_model,d_model),
                              nn.ReLU()))
            self.scales.append(nn.Parameter(torch.zeros(1)))
    def forward(self, x):
        for l,s in zip(self.layers,self.scales):
            x = x + s*l(x)
        return x


#########################################
# Enhanced RetNet Block with ReZero-style scaling
#########################################
class EnhancedRetNetBlock(nn.Module):
    def __init__(self, d_model, num_layers=12):
        super(EnhancedRetNetBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_scales = nn.ParameterList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU()
            )
            self.layers.append(layer)
            self.layer_scales.append(nn.Parameter(torch.zeros(1)))
    def forward(self, x):
        for layer, scale in zip(self.layers, self.layer_scales):
            x = x + scale * layer(x)
        return x
