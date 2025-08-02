import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# 假定已从 reward_calculator 或 config_utils 导入
# from reward_calculator import f1_pairs
def pairs_from_dotparen(s: str):
    stack, pairs = [], set()
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.add((j, i))
    return pairs

def f1_pairs(pred: str, gold: str) -> float:
    P, G = pairs_from_dotparen(pred), pairs_from_dotparen(gold)
    if not P and not G:
        return 1.0
    tp = len(P & G)
    prec = tp / max(1, len(P))
    rec = tp / max(1, len(G))
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def score_validity(sequence: str) -> float:
    """简单验证 RNA 结构的合法性（括号匹配）"""
    ca = sequence.count('A')
    cg = sequence.count('G')
    cc = sequence.count('C')
    at = sequence.count('T')
    return (ca + cg + cc + at)/(len(sequence)+1e-6)  # 简单的有效性评分：核苷酸比例

def gc_content_reward(seq: str) -> float:
    gc_ratio = (seq.count('G') + seq.count('C')) / max(1, len(seq))
    return 1.0 - abs(gc_ratio - 0.5) * 2.0  # 奖励平衡GC含量

def diversity_reward(seq: str) -> float:
    return len(set(seq)) / max(1, len(seq))  # 奖励核苷酸种类多样性


@dataclass
class RewardWeights:
    structure_similarity: float = 1.0
    perplexity_score: float = 0.0  # 新增：LM 打分的权重（>0 启用）
    thermodynamic_stability: float = 0.0,  # 热力学稳定性（>0 启用）
    base_pairing_accuracy: float = 0.0,  # 基础配对准确率（>0 启用）
    loop_structure_penalty: float = 0.0,  # 循环结构惩罚（>0 启用）
    gc_content_reward: float = 0.0,  # GC 含量奖励（>0 启用）
    diversity_bonus: float = 0.0,  # 序列多样性奖励（>0 启用）

class LMScorer:
    """自适应 LM 打分器：支持 MLM / Causal / Seq2Seq。
    - 对 MLM：使用 LOO pseudo-perplexity（标准做法）。
    - 对 Causal：对结构段计算条件 NLL。
    - 对 Seq2Seq：encoder=sequence，decoder=structure 计算 NLL。
    """

    def __init__(
        self,
        model_name: str = "yangheng/OmniGenome-186M",
        tokenizer_name: Optional[str] = "yangheng/OmniGenome",
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        model=None,
        tokenizer=None,
        batch_mask_size: int = 32,  # LOO 并行遮盖的批大小
        sep_fallback: str = "\n",   # 当 tokenizer 没有 eos/sep 时的分隔符
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_mask_size = batch_mask_size
        self.sep_fallback = sep_fallback

        # 加载或接收外部注入
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name or model_name, trust_remote_code=trust_remote_code
        )
        if model is not None:
            self.model = model
            self.config = self.model.config
        else:
            # 优先尝试 MLM
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                self.config = self.model.config
            except Exception:
                self.model = None

            # 如果不是 MLM，尝试 Causal
            if self.model is None:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                    self.config = self.model.config
                except Exception:
                    self.model = None

            # 再尝试 Seq2Seq
            if self.model is None:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                self.config = self.model.config

        self.model.to(self.device).eval()

        # 判定模型类型
        self.is_decoder = getattr(self.config, "is_decoder", False)
        self.is_encoder_decoder = getattr(self.config, "is_encoder_decoder", False)
        self.is_mlm = hasattr(self.model, "get_output_embeddings") and not self.is_encoder_decoder and not self.is_decoder
        # 以上判定不是万无一失，但在大多数 HF 模型上足够区分

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            # 强制设置 pad（避免批处理报错）
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or "</s>"
            self.pad_token_id = self.tokenizer.pad_token_id

    def _sep(self):
        return self.tokenizer.eos_token or self.tokenizer.sep_token or self.sep_fallback

    @torch.inference_mode()
    def score(self, sequence: str, structure: str) -> float:
        """返回“越大越好”的一致性分数（负 NLL）。
        - MLM: -average LOO loss
        - Causal: -NLL(structure | sequence+SEP)
        - Seq2Seq: -NLL(structure | sequence)
        """
        if self.is_encoder_decoder:
            return -self._nll_seq2seq(sequence, structure)
        elif self.is_decoder:
            return -self._nll_causal(sequence, structure)
        else:
            # 视为 MLM
            return -self._ppl_mlm_loo(sequence, structure)

    # ---------- MLM: LOO PPL* ----------
    def _ppl_mlm_loo(self, sequence: str, structure: str) -> float:
        sep = self._sep()
        ctx_ids = self.tokenizer(f"{sequence}{sep}", return_tensors="pt", add_special_tokens=True).to(self.device)
        str_ids = self.tokenizer(structure, return_tensors="pt", add_special_tokens=False).to(self.device)

        ctx_len = ctx_ids.input_ids.shape[1]
        T = str_ids.input_ids.shape[1]
        if T == 0:
            return 0.0

        # 基础拼接
        base = torch.cat([ctx_ids.input_ids, str_ids.input_ids], dim=1)  # (1, ctx+T)
        attn = torch.ones_like(base)

        total_loss, count = 0.0, 0
        # 分批遮盖，避免一次性 T 前向
        for start in range(0, T, self.batch_mask_size):
            end = min(T, start + self.batch_mask_size)
            bsz = end - start

            input_ids = base.repeat(bsz, 1)           # (bsz, ctx+T)
            labels = torch.full_like(input_ids, -100) # 只在被遮盖位计算 loss

            # 对每个样本遮盖一个不同位置
            for i, pos in enumerate(range(start, end)):
                global_pos = ctx_len + pos
                # 用 [MASK] token（若无，则退回 tokenizer.mask_token_id 或 pad）
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None:
                    # 如果没有 mask_token，退化为把该位覆盖为 pad；虽然不理想，但总比不遮好
                    mask_id = self.pad_token_id
                input_ids[i, global_pos] = mask_id
                labels[i, global_pos] = base[0, global_pos]

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                # 注意：labels 已经是 -100 的遮盖形式
                # 计算 LOO pseudo-perplexity
                # 注意：outputs.loss 已经是被标注位置的平均 loss
                input_ids = input_ids.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attn[:bsz], labels=labels)
            # outputs.loss：已是被标注位置的平均 loss
            total_loss += float(outputs.loss.item()) * bsz
            count += bsz

        avg_loss = total_loss / max(1, count)  # 平均 token-level loss
        # 返回 pseudo-perplexity（可选）： math.exp(avg_loss)
        return avg_loss  # 注意：score() 返回 -avg_loss

    # ---------- Causal: 结构段条件 NLL ----------
    def _nll_causal(self, sequence: str, structure: str) -> float:
        sep = self._sep()
        ctx = self.tokenizer(f"{sequence}{sep}", return_tensors="pt", add_special_tokens=True).to(self.device)
        tgt = self.tokenizer(structure, return_tensors="pt", add_special_tokens=False).to(self.device)

        # 拼接输入
        input_ids = torch.cat([ctx.input_ids, tgt.input_ids], dim=1)
        attn = torch.ones_like(input_ids)

        # 只在结构段计算 NLL（labels 为下一个 token 的预测）
        labels = input_ids.clone()
        labels[:, : ctx.input_ids.size(1)] = -100  # 忽略上下文
        # 经典因果 LM 目标：predict next token → 向左对齐
        labels = labels[:, 1:]
        input_ids = input_ids[:, :-1]
        attn = attn[:, :-1]

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            # 注意：labels 已经是 -100 的遮盖形式
            # 计算结构段的 NLL
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
        return float(outputs.loss.item())

    # ---------- Seq2Seq: 结构段 NLL ----------
    def _nll_seq2seq(self, sequence: str, structure: str) -> float:
        enc = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        dec = self.tokenizer(structure, return_tensors="pt", add_special_tokens=True).to(self.device)
        labels = dec.input_ids.clone()
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            # 注意：labels 已经是 -100 的遮盖形式
            # 计算结构段的 NLL
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec.input_ids, labels=labels)
        return float(outputs.loss.item())

class RNARewardCalculator:
    """把结构 F1 与 LM 一致性合并为奖励。
    - sim = F1(pred_struct, target)
    - lm_score = 根据配置从 (sequence, X) 得到的一致性分数（越大越好）
    - final_reward = w.structure_similarity * sim + w.perplexity_score * lm_score
    """

    def __init__(
        self,
        structure_predictor,
        weights: RewardWeights = RewardWeights(),
        lm_scorer: Optional[LMScorer] = None,
        device: Optional[str] = None,
        lm_mode: str = "target",  # "target" | "pred" | "delta"
    ):
        self.sp = structure_predictor
        self.w = weights
        self.device = device or getattr(self.sp, "device", None)
        self.lm_mode = lm_mode
        self.lm = lm_scorer

    @torch.inference_mode()
    def step_reward(self, prefix: str, target: str) -> float:
        """为部分序列提供 shaping 奖励"""
        valid_score = score_validity(prefix)

        partial_sim = 0.0
        if len(prefix) >= 5:
            pred_struct = self.sp.predict_structure(prefix)
            partial_target = target[:len(pred_struct)]
            partial_sim = f1_pairs(pred_struct, partial_target)

        reward = (
                0.2 * valid_score
                + 0.6 * partial_sim
                + 0.2 * gc_content_reward(prefix)
        )
        return reward

    @torch.inference_mode()
    def final_reward(self, sequence: str, target: str) -> float:
        # (1) 结构相似度
        pred_struct = self.sp.predict_structure(sequence)
        sim = f1_pairs(pred_struct, target)
        validity = score_validity(sequence)
        reward_struct = self.w.structure_similarity * float(sim)

        # (2) 语言模型一致性（可选）
        reward_lm = 0.0
        if getattr(self.w, "perplexity_score", 0.0) > 0.0:
            with torch.no_grad():
                if self.lm_mode == "target":
                    score = self.lm.score(sequence, target)          # 希望更像“目标结构”
                elif self.lm_mode == "pred":
                    score = self.lm.score(sequence, pred_struct)      # 度量“预测结构”的自洽
                else:  # "delta"
                    score = self.lm.score(sequence, target) - self.lm.score(sequence, pred_struct)
                reward_lm = self.w.perplexity_score * float(score)
        gc_score = gc_content_reward(sequence) * self.w.gc_content_reward
        diversity = diversity_reward(sequence) * self.w.diversity_bonus

        return reward_struct + reward_lm + validity + gc_score + diversity