from turtle import forward
from uu import decode
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
    # matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # vector of shape (seq_len)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # apply sin to even and cos to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(1) #(1, seq_len, d_model)

    self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # additive
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask ==0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)  
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        x = self.w_o(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.multihead_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.multihead_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.masked_multihead_attention_block = self_attention_block
        self.cross_multihead_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.masked_multihead_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_multihead_attention_block(x, encoder_output, encoder_output, src_mask)) # query from decoder. key and value from encoder
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer, dropout:float) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int = 512, N:int = 6, h:int = 8, dropout:float = 0.1, d_ff:int = 2048) -> Transformer:
    # Embedding layer
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_multihead_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_multihead_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_masked_multihead_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_multihead_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_masked_multihead_attention_block, decoder_cross_multihead_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer, dropout)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer