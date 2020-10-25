import torch
import torch.nn as nn
import torch.nn.functional as F

class Code2VecAttn(nn.Module):
    def __init__(self, c_dim):
        super(Code2VecAttn, self).__init__()
        self.c_dim = c_dim  # (batch, input_len, dim)
        self.attn_parameters = nn.Parameter(
            torch.randn(c_dim, 1),
            requires_grad=True)
    
    def forward(self, x, mask=None):
        attn_weights = torch.matmul(x, self.attn_parameters)
        if mask is not None:
            attn_weights = attn_weights.masked_fill_(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        results = torch.sum(x * attn_weights, dim=1)
        return results, attn_weights


class Code2VecEncoder(nn.Module):
    """
    Vanilla code2vec model
    """
    def __init__(self,
        token_vocab, path_vocab, target_vocab,
        token_dim, path_dim, target_dim, max_len, padding_idx,
        dropout
    ):
        assert token_dim == path_dim and path_dim == target_dim
        super(Code2VecEncoder, self).__init__()
        self.token_embeddings = nn.Embedding(token_vocab, token_dim, padding_idx=padding_idx)
        self.path_embeddings = nn.Embedding(path_vocab, path_dim, padding_idx=padding_idx)
        self.context_dropout = nn.Dropout(p=dropout)
        self.attn = Code2VecAttn(path_dim)
        self.linear = nn.Linear(2 * token_dim + path_dim, path_dim)
        self.tanh = nn.Tanh()
        self.target_embeddings = nn.Linear(path_dim, target_vocab)  # not actual embeddings but weights of linear
    
    def forward(self, x, mask=None):
        token_l, path, token_r = self.token_embeddings(x[:, :, 0]), self.path_embeddings(x[:, :, 1]), self.token_embeddings(x[:, :, 2])
        c = self.context_dropout(torch.cat([token_l, path, token_r], dim=-1))
        c_tlide = self.tanh(self.linear(c))
        v, attn_weights = self.attn(c_tlide, mask)
        return v
        # return F.log_softmax(self.target_embeddings(v.squeeze()), dim=1)


class PathContextEncoder(nn.Module):
    """
    Encoder that mimics code2vec to encode bags of path contexts.
    """
    def __init__(self,
        token_vocab, path_vocab, token_dim, path_dim, max_len, pad_idx, dropout):
        super(PathContextEncoder, self).__init__()
        assert token_dim == path_dim
        self.d_model = token_dim
        self.pad = pad_idx
        self.token_embeddings = nn.Embedding(token_vocab, token_dim, padding_idx=pad_idx)
        self.path_embeddings = nn.Embedding(path_vocab, path_dim, padding_idx=pad_idx)
        self.context_dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(2 * token_dim + path_dim, path_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.d_model, nhead=4, dim_feedforward=256, 
                dropout=0.1, activation='relu'),
            num_layers=3, norm=nn.LayerNorm(self.d_model)
        )

    def forward(self, x):
        """
        x: (batch_size, bag_size, 3)
        """
        mask = (x[:, :, 0] != self.pad)    # assume same padding on 3 dimensions
        token_l, path, token_r = self.token_embeddings(x[:, :, 0]), self.path_embeddings(x[:, :, 1]), self.token_embeddings(x[:, :, 2])
        c = self.context_dropout(torch.cat([token_l, path, token_r], dim=-1))
        c_tlide = self.tanh(self.linear(c))
        memory = self.encoder(c_tlide.transpose(1, 0), mask=None, src_key_padding_mask=mask)
        code = F.max_pool1d(memory.permute(1, 2, 0), memory.size(0)).squeeze(2)
        print(code.shape)
        assert 0
        return code


from .set import SAB, ISAB, PMA
import copy

class SetPathContextEncoder(nn.Module):
    def __init__(self,
        token_vocab, path_vocab, token_dim, path_dim, max_len, pad_idx, dropout,
        num_outputs, num_inds=32, num_heads=4, ln=False):
        super(SetPathContextEncoder, self).__init__()
        assert token_dim == path_dim
        self.d_model = token_dim
        self.pad = pad_idx
        self.token_embeddings = nn.Embedding(token_vocab, token_dim, padding_idx=pad_idx)
        self.path_embeddings = nn.Embedding(path_vocab, path_dim, padding_idx=pad_idx)
        self.context_dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(2 * token_dim + path_dim, path_dim)
        self.relu = nn.ReLU()
        self.encoder = nn.Sequential(
            # nn.ModuleList(
            #     [copy.deepcopy(
            #         ISAB(self.d_model, self.d_model, num_heads, num_inds, ln))
            #         for _ in range(num_layers)]),
            ISAB(self.d_model, self.d_model, num_heads, num_inds, ln),
            ISAB(self.d_model, self.d_model, num_heads, num_inds, ln),
            ISAB(self.d_model, self.d_model, num_heads, num_inds, ln),
            PMA(self.d_model, num_heads, num_outputs, ln)
        )

    def forward(self, x):
        mask = (x[:, :, 0] != self.pad)    # assume same padding on 3 dimensions
        token_l, path, token_r = self.token_embeddings(x[:, :, 0]), self.path_embeddings(x[:, :, 1]), self.token_embeddings(x[:, :, 2])
        c = self.context_dropout(torch.cat([token_l, path, token_r], dim=-1))
        c_tlide = self.relu(self.linear(c)) # replace tanh with relu ot prevent gradient vanishment
        code = self.encoder(c_tlide)
        return code.squeeze(1)  # necessary for loss computation