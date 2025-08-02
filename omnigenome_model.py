# -*- coding: utf-8 -*-
"""
OmniGenome Seq2Seq policy with a Value head for RL (GRPO).
Provides:
  - BaseStructurePredictor: lightweight folding for rewards (ViennaRNA if available).
  - create_omnigenome_model(): factory returning BaseStructurePredictor.
  - OmniGenomeForSeq2SeqWithValueHead: tiny Transformer encoder-decoder + value head.
  - OmniGenomeSeq2SeqPolicy: RL-facing wrapper with set_target() and get_policy_output().
"""
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from OmniGenome.modeling_omnigenome import OmniGenomeForSeq2SeqWithValueHead
# ------------------------
# Lightweight base predictor for rewards (structure folding)
# ------------------------
try:
    import ViennaRNA  # optional
    _VRNA = True
except Exception:
    _VRNA = False

class BaseStructurePredictor:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_structure(self, sequence: str) -> str:
        if _VRNA:
            md = ViennaRNA.md()
            md.max_bp_span = 400
            fc = ViennaRNA.fold_compound(sequence, md)
            (ss, mfe) = fc.mfe()
            return ss
        return '.' * len(sequence)

    # stubs for compatibility with previous code
    def predict_sequence(self, structure: str, partial_sequence: str = "") -> str:
        return 'A' * len(structure)
    def predict_free_energy(self, sequence: str) -> float:
        return 0.0
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        return torch.tensor([0]*len(sequence))
    def encode_structure(self, structure: str) -> torch.Tensor:
        return torch.tensor([0]*len(structure))

def create_omnigenome_model(model_type: str = "mock", model_name: str = "", device: Optional[str] = None):
    return BaseStructurePredictor(device=device)

# ------------------------
# Vocab
# ------------------------
STRUCT_TOKS = {'(' : 0, ')' : 1, '.' : 2}
NUC_TOKS = {'A':0,'U':1,'G':2,'C':3}
INV_NUC = {v:k for k,v in NUC_TOKS.items()}
BOS_ID = 4
ENC_VOCAB = 3
DEC_VOCAB = 5  # A/U/G/C + BOS
#
# # ------------------------
# # Blocks
# # ------------------------
# class ValueHead(nn.Module):
#     def __init__(self, d_model: int):
#         super().__init__()
#         self.value = nn.Linear(d_model, 1)
#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         return self.value(h).squeeze(-1)
#
# class OmniGenomeForSeq2SeqWithValueHead(nn.Module):
#     def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_ff: int = 1024, max_len: int = 512):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#         self.enc_embed = nn.Embedding(ENC_VOCAB, d_model)
#         self.dec_embed = nn.Embedding(DEC_VOCAB, d_model)
#         self.pos_enc = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
#         self.pos_dec = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
#         enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
#         dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
#         self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
#         self.lm_head = nn.Linear(d_model, 4)
#         self.value_head = ValueHead(d_model)
#
#     def _add_pos(self, x: torch.Tensor, is_dec: bool) -> torch.Tensor:
#         pos = self.pos_dec if is_dec else self.pos_enc
#         return x + pos[:x.size(1), :]
#
#     def forward(self, enc_ids: torch.LongTensor, dec_ids: torch.LongTensor, enc_pad_mask: Optional[torch.BoolTensor]=None, dec_pad_mask: Optional[torch.BoolTensor]=None):
#         B, S = enc_ids.size()
#         B2, T = dec_ids.size()
#         assert B == B2, "Batch mismatch"
#         enc = self._add_pos(self.enc_embed(enc_ids), is_dec=False)
#         mem = self.encoder(enc, src_key_padding_mask=enc_pad_mask)
#         dec = self._add_pos(self.dec_embed(dec_ids), is_dec=True)
#         tgt_mask = torch.triu(torch.ones(T, T, device=dec.device), diagonal=1).bool()
#         out = self.decoder(dec, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=dec_pad_mask, memory_key_padding_mask=enc_pad_mask)
#         logits = self.lm_head(out)
#         values = self.value_head(out)
#         return logits, values

# ------------------------
# RL wrapper
# ------------------------
class OmniGenomeSeq2SeqPolicy(nn.Module):
    def __init__(self, base_predictor: Optional[BaseStructurePredictor] = None, device: Optional[str] = None):
        super().__init__()
        self.device_ = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        auto_config = AutoConfig.from_pretrained("OmniGenome", trust_remote_code=True) # load default config
        self.seq2seq = OmniGenomeForSeq2SeqWithValueHead(auto_config).to(self.device_)
        self.base_predictor = base_predictor or BaseStructurePredictor(device=device)
        self._cached_struct_ids = None
        self._target_len = None

    # compatibility: reward/env can call folding through the policy if needed
    def predict_structure(self, seq: str) -> str:
        return self.base_predictor.predict_structure(seq)
    def predict_sequence(self, structure: str, partial_sequence: str = "") -> str:
        return self.base_predictor.predict_sequence(structure, partial_sequence)
    def predict_free_energy(self, sequence: str) -> float:
        return self.base_predictor.predict_free_energy(sequence)

    def set_target(self, structure: str):
        ids = torch.tensor([STRUCT_TOKS.get(c, 2) for c in structure], dtype=torch.long, device=self.device_).view(1, -1)
        self._cached_struct_ids = ids
        self._target_len = int(ids.size(1))

    @property
    def policy_network(self):
        class _Dummy: input_dim = 1
        return _Dummy()

    # def get_policy_output(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Batched:
    #     state: (B, F) where F = 2*S + 1 for fixed target length S in current batch.
    #     Build decoder inputs per row based on pos_ratio* S, pad to T_max, compute logits/values,
    #     then select the last valid step for each row.
    #     Returns: (B,4) logits, (B,) values
    #     """
    #     if self._cached_struct_ids is None:
    #         raise RuntimeError("Call set_target(structure) before using the policy.")
    #     if state.dim() == 1:
    #         state = state.view(1, -1)
    #     B, F = state.size()
    #     S = int(self._target_len)
    #     expected_F = 2 * S + 1
    #     if F != expected_F:
    #         raise RuntimeError(f"State size mismatch: got F={F}, expected 2*S+1={expected_F} with S={S}")
    #
    #     # split
    #     seq_block = state[:, :S].long().to(self.device_)
    #     pos_ratio = state[:, -1].clamp(0, 1)
    #     Ls = torch.round(pos_ratio * S).long().tolist()
    #
    #     # decoder inputs
    #     dec_list = []
    #     T_list = []
    #     for b in range(B):
    #         L = int(Ls[b])
    #         if L <= 0:
    #             dec = torch.tensor([BOS_ID], dtype=torch.long, device=self.device_).unsqueeze(0)
    #         else:
    #             prev = seq_block[b, :L].view(1, -1)
    #             bos = torch.full((1, 1), BOS_ID, dtype=torch.long, device=self.device_)
    #             dec = torch.cat([bos, prev], dim=1)
    #         dec_list.append(dec);
    #         T_list.append(dec.size(1))
    #     T_max = max(T_list) if T_list else 1
    #     dec_ids = torch.full((B, T_max), BOS_ID, dtype=torch.long, device=self.device_)
    #     dec_pad = torch.ones(B, T_max, dtype=torch.bool, device=self.device_)
    #     for b, dec in enumerate(dec_list):
    #         t = dec.size(1)
    #         dec_ids[b, :t] = dec
    #         dec_pad[b, :t] = False
    #     enc_ids = self._cached_struct_ids.expand(B, S)
    #
    #     # 使用mixed precision和gradient checkpointing来减少显存
    #     with torch.amp.autocast(enabled=True, device_type=self.device_.type, dtype=torch.float16):
    #         # 使用return_dict=True获取结构化输出
    #         outputs = self.seq2seq(
    #             input_ids=enc_ids,
    #             attention_mask=None,
    #             decoder_input_ids=dec_ids,
    #             decoder_attention_mask=~dec_pad,  # 注意这里需要取反
    #             return_dict=True
    #         )
    #         logits = outputs.logits
    #         values = outputs.value
    #
    #     # 提取最后一步的输出
    #     last_logits = torch.zeros(B, logits.size(-1), device=logits.device, dtype=logits.dtype)
    #     last_value = torch.zeros(B, device=values.device, dtype=values.dtype)
    #     for b, t in enumerate(T_list):
    #         last_logits[b] = logits[b, t - 1, :]
    #         last_value[b] = values[b, t - 1]
    #
    #     # 立即清理中间张量
    #     del outputs, logits, values, enc_ids, dec_ids, dec_pad, dec_list, seq_block
    #
    #     return last_logits, last_value

    # In omnigenome_model.py
    def get_policy_output(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... (code to validate state size)

        # --- vectorized split & decoder build ---
        B, S = state.size(0), self._target_len
        seq_block = state[:, :S].long().to(self.device_)
        pos_ratio = state[:, -1].clamp(0, 1)
        Ls = torch.round(pos_ratio * S).long()  # (B,)

        # Decoder IDs: BOS + previous tokens up to length Ls[b]
        T_max = int(torch.clamp(Ls, min=0).max().item()) + 1  # +1 for BOS
        if T_max < 1: T_max = 1
        dec_ids = torch.full((B, T_max), BOS_ID, dtype=torch.long, device=self.device_)
        if T_max > 1:
            T_sub = min(T_max - 1, S)
            dec_ids[:, 1:1 + T_sub] = seq_block[:, :T_sub]
            # Zero-out positions beyond each L with BOS to be tidy (mask already handles it)
            ar = torch.arange(T_max - 1, device=self.device_).unsqueeze(0)  # (1,T-1)
            mask_over = ar >= Ls.clamp(min=0).unsqueeze(1)  # (B,T-1)
            dec_ids[:, 1:1 + T_sub] = torch.where(mask_over[:, :T_sub],
                                                  torch.full_like(dec_ids[:, 1:1 + T_sub], BOS_ID),
                                                  dec_ids[:, 1:1 + T_sub])
        # Attention masks
        enc_ids = self._cached_struct_ids.expand(B, S)
        enc_attn = torch.ones(B, S, dtype=torch.long, device=self.device_)  # no padding on encoder
        dec_attn = (torch.arange(T_max, device=self.device_).unsqueeze(0) <= Ls.clamp(min=0).unsqueeze(1)).long()

        with torch.amp.autocast(enabled=True, device_type=self.device_.type, dtype=torch.float16):
            outputs = self.seq2seq(
                input_ids=enc_ids,
                attention_mask=enc_attn,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_attn,
                return_dict=True
            )
            logits = outputs.logits  # (B,T_max,V)
            values = outputs.value  # (B,T_max)

        last_pos = Ls.clamp(min=0)  # (B,)
        arangeB = torch.arange(B, device=logits.device)
        last_logits = logits[arangeB, last_pos, :]
        last_value = values[arangeB, last_pos]

        del outputs, logits, values, enc_ids, dec_ids, enc_attn, dec_attn, seq_block
        return last_logits, last_value

    # torch.Module plumbing
    def parameters(self, *args, **kwargs):
        return self.seq2seq.parameters(*args, **kwargs)
    def state_dict(self, *args, **kwargs):
        return self.seq2seq.state_dict(*args, **kwargs)
    def load_state_dict(self, *args, **kwargs):
        return self.seq2seq.load_state_dict(*args, **kwargs)
    def to(self, *args, **kwargs):
        self.seq2seq.to(*args, **kwargs); return self
    def train(self, mode: bool = True):
        self.seq2seq.train(mode); return self
    def eval(self):
        self.seq2seq.eval(); return self
