
# -*- coding: utf-8 -*-
"""
Vectorized RNADesignEnvironment with safe padding for batch rollouts (single target per batch).

State encoding per sample (fixed for a given target of length S):
    [ seq_ids[0:S]  +  struct_ids[0:S]  +  pos_ratio ]
where:
    - seq_ids[i] in {0,1,2,3} for A,U,G,C; unused positions are 0 (ignored by policy via pos_ratio)
    - struct_ids[i] in {0,1,2} for '(', ')', '.' (constant for the batch)
    - pos_ratio = current_length / S  (scalar in [0,1])

The environment advances one token per step. An episode terminates when current_length == S.
Rewards:
    - by default, only terminal reward from RNARewardCalculator.final_reward(sequence, target)
    - optional shaping from RNARewardCalculator.step_reward(prefix, target) can be enabled.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
from transformers import AutoTokenizer

NUC = ['A','T','G','C']
NUC2ID = {c:i for i,c in enumerate(NUC)}
STRUCT2ID = {'(' : 0, ')' : 1, '.' : 2}

@dataclass
class EnvConfig:
    use_step_shaping: bool = False
    device: str = 'cpu'

class RNADesignEnvironment:
    def __init__(self, reward_calculator, config: EnvConfig = EnvConfig(), tokenizer=None):
        self.rc = reward_calculator
        self.cfg = config
        self.device = torch.device(self.cfg.device)
        # runtime buffers
        self.target: Optional[str] = None
        self.S: int = 0
        self.struct_ids: torch.LongTensor = None  # (S,)
        self.batch_size: int = 1
        self.seq_ids: torch.LongTensor = None     # (B,S) int64
        self.lengths: torch.LongTensor = None     # (B,) int64
        self.dones: torch.BoolTensor = None       # (B,)
        self.tokenizer = tokenizer

    # ---------- helpers ----------
    def _encode_structure(self, s: str) -> torch.LongTensor:
        return torch.tensor([STRUCT2ID.get(c, 2) for c in s], dtype=torch.long, device=self.device)

    def _make_state(self) -> torch.FloatTensor:
        """Return padded state batch: (B, 2*S + 1)"""
        B, S = self.batch_size, self.S
        seq = self.seq_ids.clone()  # (B,S)
        struct = self.struct_ids.view(1, S).expand(B, S)
        pos_ratio = (self.lengths.float() / max(S, 1)).unsqueeze(1)  # (B,1)
        state = torch.cat([seq.float(), struct.float(), pos_ratio], dim=1)  # (B, 2S+1)
        return state

    def is_done(self) -> bool:
        return bool(torch.all(self.dones).item()) if self.dones is not None else True

    # ---------- API ----------
    def reset(self, target: str, batch_size: int = 1) -> torch.FloatTensor:
        self.target = target
        self.S = len(target)
        self.struct_ids = self._encode_structure(target)
        self.batch_size = batch_size
        self.seq_ids = torch.zeros(batch_size, self.S, dtype=torch.long, device=self.device)
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        return self._make_state()

    # def step(self, actions: List[int]) -> Tuple[torch.FloatTensor, List[float], List[bool], List[Dict]]:
    #     """
    #     actions: list[int] of length B (ignored for done entries)
    #     Returns: (next_state(B,F), rewards(list), dones(list), infos(list))
    #     """
    #     B = self.batch_size
    #     assert len(actions) == B, f"actions must have length {B}"
    #     rewards = [0.0] * B
    #     infos: List[Dict] = [{} for _ in range(B)]
    #
    #     for i in range(B):
    #         if self.dones[i]:
    #             continue
    #         a = int(actions[i])  # 0..3
    #         L = int(self.lengths[i].item())
    #         if L < self.S:
    #             self.seq_ids[i, L] = a
    #             self.lengths[i] = L + 1
    #             if self.cfg.use_step_shaping:
    #                 prefix = ''.join(NUC[x] for x in self.seq_ids[i, :L+1].tolist())
    #                 rewards[i] += float(self.rc.step_reward(prefix, self.target))
    #         if self.lengths[i] >= self.S:
    #             full_seq = self.tokenizer.decode(self.seq_ids[i, :self.S].tolist(), skip_special_tokens=True).replace(' ', '')
    #             rewards[i] += float(self.rc.final_reward(full_seq, self.target))
    #             infos[i]['sequence'] = full_seq
    #             infos[i]['terminal'] = True
    #             self.dones[i] = True
    #
    #     next_state = self._make_state()
    #     return next_state, rewards, self.dones.tolist(), infos


    # In rna_environment.py
    def step(self, actions: List[int]) -> Tuple[torch.FloatTensor, List[float], List[bool], List[Dict]]:
        B = len(actions)

        assert len(actions) == B, f"actions must have length {B}"
        rewards = [0.0] * B
        infos: List[Dict] = [{} for _ in range(B)]

        # vectorized write of next token
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.long)
        not_done = ~self.dones
        can_place = not_done & (self.lengths < self.S)
        idx = torch.nonzero(can_place, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            pos = self.lengths[idx]
            self.seq_ids[idx, pos] = actions_t[idx]
            self.lengths[idx] = pos + 1
            if self.cfg.use_step_shaping:
                for i in idx.tolist():
                    L = int(self.lengths[i].item())
                    # prefix = ''.join(NUC[x] for x in self.seq_ids[i, :L].tolist())
                    prefix = self.tokenizer.decode(self.seq_ids[i, :L].tolist(), skip_special_tokens=True).replace(' ', '')
                    # calculate step reward
                    rewards[i] += float(self.rc.step_reward(prefix, self.target))

        # handle newly finished episodes
        done_now = (self.lengths >= self.S) & (~self.dones)
        fin_idx = torch.nonzero(done_now, as_tuple=False).squeeze(1)
        if fin_idx.numel() > 0:
            for i in fin_idx.tolist():
                # faster decode without tokenizer
                # full_seq = ''.join(NUC[x] for x in self.seq_ids[i, :self.S].tolist())
                full_seq = self.tokenizer.decode(self.seq_ids[i, :self.S].tolist(), skip_special_tokens=True).replace(' ', '')
                # calculate final reward
                rewards[i] += float(self.rc.final_reward(full_seq, self.target))
                infos[i]['sequence'] = full_seq
                infos[i]['terminal'] = True
            self.dones[fin_idx] = True

        next_state = self._make_state()
        return next_state, rewards, self.dones.tolist(), infos