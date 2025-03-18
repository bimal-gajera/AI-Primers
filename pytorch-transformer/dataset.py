import torch
import torch.nn as nn
from torch.utils.data import Dataset


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_language, tgt_language, seq_len) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_language]
        tgt_text = src_target_pair['translation'][self.tgt_language]
        
        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_target_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_num_padding_tokens = self.seq_len -  len(encoder_input_tokens) - 2 # 2 for SOS and EOS
        decoder_num_padding_tokens = self.seq_len -  len(decoder_target_tokens) - 1 # 1 for SOS

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add SOS and EOS to encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_target_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # decoder output, Add EOS
        label = torch.cat(
            [
                torch.tensor(decoder_target_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )
    
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

