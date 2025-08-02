"""
GRPO trainer with parallel rollouts per target (batch episodes with safe padding).
Enhanced with memory management to prevent GPU memory leaks.
"""
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
import gc

import tqdm
from torch import GradScaler


@dataclass
class GRPOConfig:
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    epochs: int = 4
    group_size: int = 8
    max_steps: int = 512
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    temperature: float = 1.0
    greedy_eval: bool = False

class Episode:
    __slots__ = ("target", "obs_list", "act_list", "logp_list", "reward")
    def __init__(self, target: str):
        self.target = target
        self.obs_list: List[torch.Tensor] = []
        self.act_list: List[int] = []
        self.logp_list: List[float] = []
        self.reward: float = 0.0

    def traj_logp(self) -> float:
        return float(sum(self.logp_list))

    def clear_tensors(self):
        """Clear stored tensors to free memory"""
        self.obs_list.clear()
        del self.obs_list
        self.obs_list = []

class GRPOTrainer:
    def __init__(self, model, env, config: GRPOConfig):
        self.model = model
        self.env = env
        self.config = config
        self.device = torch.device(config.device if isinstance(config.device, str) else config.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.logs = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))  # ✅ 添加混合精度 scaler

    def _clear_gpu_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def rollout_parallel(self, target: str, batch_size: int) -> List[Episode]:
        # initialize env and model context
        if hasattr(self.model, 'set_target'):
            self.model.set_target(target)
        state = self.env.reset(target, batch_size=batch_size)  # (B, F)
        B = state.size(0)
        S = (state.size(1) - 1) // 2
        episodes = [Episode(target) for _ in range(B)]
        steps = 0
        dones = [False]*B
        rewards_acc = [0.0]*B

        while steps < self.config.max_steps and not all(dones):
            with torch.no_grad():
                logits, _ = self.model.get_policy_output(state.to(self.device))
                if self.config.temperature != 1.0:
                    logits = logits / max(self.config.temperature, 1e-8)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()  # (B,)
                logps = dist.log_prob(actions)  # (B,)

                del logits, dist

            # record per-episode step
            for i in range(B):
                if dones[i]: continue
                # 只保存CPU张量以减少GPU显存占用
                episodes[i].obs_list.append(state[i].detach().cpu().clone())
                episodes[i].act_list.append(int(actions[i].item()))
                episodes[i].logp_list.append(float(logps[i].item()))

            # env step - 先保存actions列表再删除张量
            actions_list = [int(actions[i].item()) for i in range(B)]
            del actions, logps

            state, rewards, dones_list, infos = self.env.step(actions_list)

            for i in range(B):
                rewards_acc[i] += float(rewards[i])
                if dones_list[i] and not dones[i]:
                    episodes[i].reward = rewards_acc[i]
                    self.episode_rewards.append(rewards_acc[i])
                    self.episode_lengths.append(steps+1)
            dones = dones_list
            steps += 1

        # fill any unfinished episodes (safety)
        for i in range(B):
            if not episodes[i].reward:
                episodes[i].reward = rewards_acc[i]

        return episodes

    def collect_by_group(self, targets: List[str]) -> Dict[str, List[Episode]]:
        groups: Dict[str, List[Episode]] = defaultdict(list)
        for tgt in targets:
            eps = self.rollout_parallel(tgt, batch_size=self.config.group_size)
            groups[tgt].extend(eps)

        return groups

    def _recompute_traj_logp(self, eps: List[Episode]) -> torch.Tensor:
        """
        Vectorized recomputation of trajectory log-probabilities for a list of episodes.
        This version correctly retains computation graphs for backpropagation.
        """
        if not eps:
            return torch.empty(0, device=self.device)

        if hasattr(self.model, 'set_target'):
            self.model.set_target(eps[0].target)

        # Flatten all observations and actions from the episode list
        flat_obs, flat_act, seq_lens = [], [], []
        for ep in eps:
            flat_obs.extend(ep.obs_list)
            flat_act.extend(ep.act_list)
            seq_lens.append(len(ep.act_list))

        obs_tensor = torch.stack(flat_obs).to(self.device)
        act_tensor = torch.tensor(flat_act, dtype=torch.long, device=self.device)

        # Single batch forward pass; gradients must be enabled for backpropagation.
        # Do NOT use torch.inference_mode() or torch.no_grad() here.
        with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            logits, _ = self.model.get_policy_output(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(act_tensor)

        # Sum the log-probability pieces for each trajectory to get a (B,) vector
        pieces = log_probs.split_with_sizes(seq_lens)
        traj_sums = [p.sum() for p in pieces]
        out = torch.stack(traj_sums)

        # Robustness checks
        assert out.dim() == 1 and out.size(0) == len(eps), f"new_logps shape={tuple(out.shape)}, expected ({len(eps)},)"
        assert out.requires_grad, "new_logps lost grad; check autocast/inference contexts"

        return out

    # def _recompute_traj_logp(self, ep: Episode) -> torch.Tensor:
    #     """重新计算轨迹log概率，添加显存管理"""
    #     if hasattr(self.model, 'set_target'):
    #         self.model.set_target(ep.target)
    #
    #     tot = torch.tensor(0.0, device=self.device, requires_grad=True)
    #
    #     for obs_cpu, act in zip(ep.obs_list, ep.act_list):
    #         obs = obs_cpu.to(self.device).view(1, -1)
    #         with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
    #             logits, _ = self.model.get_policy_output(obs)
    #             dist = torch.distributions.Categorical(logits=logits)
    #             log_prob = dist.log_prob(torch.tensor([act], device=logits.device)).squeeze()
    #             tot = tot + log_prob
    #
    #         # 立即清理中间张量
    #         del obs, logits, dist, log_prob
    #
    #     return tot

    def update_policy(self, groups: Dict[str, List[Episode]]) -> Dict[str, Any]:
        logs = {"loss": 0.0, "kl": 0.0, "avg_reward": 0.0}
        total_eps = sum(len(v) for v in groups.values())
        if total_eps == 0:
            return logs
        total_reward = sum(ep.reward for eps in groups.values() for ep in eps)
        logs["avg_reward"] = total_reward / total_eps

        for epoch in tqdm.tqdm(range(self.config.epochs), desc=f"Updating policy with {len(groups)} groups"):
            self.model.train()
            epoch_loss = 0.0
            epoch_kl = 0.0
            n_groups = 0

            for tgt, eps in groups.items():
                rewards = torch.tensor([e.reward for e in eps], dtype=torch.float32, device=self.device)
                baseline = rewards.mean()
                adv = rewards - baseline
                if adv.std() > 1e-6:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                old_logps = torch.tensor([e.traj_logp() for e in eps], dtype=torch.float32, device=self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                    new_logps = self._recompute_traj_logp(eps)
                    ratio = torch.exp(new_logps - old_logps)
                    clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                    obj = -torch.mean(torch.min(ratio * adv, clipped * adv))
                    approx_kl = torch.clamp(torch.mean(old_logps - new_logps), min=0.0)
                    loss = obj + self.config.kl_coef * approx_kl

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()  # ✅ 缩放 loss 做 backward
                self.scaler.unscale_(self.optimizer)  # ✅ unscale 后才能 clip grad
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.scaler.step(self.optimizer)  # ✅ 使用 scaler.step
                self.scaler.update()  # ✅ 更新 scaler

                epoch_loss += float(loss.item())
                epoch_kl += float(approx_kl.item())
                n_groups += 1

                del rewards, baseline, adv, old_logps, new_logps, ratio, clipped, obj, approx_kl, loss

            logs["loss"] = epoch_loss / max(1, n_groups)
            logs["kl"] = epoch_kl / max(1, n_groups)

        for eps in groups.values():
            for ep in eps:
                ep.clear_tensors()

        self.logs.append(logs.copy())
        return logs


    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "logs": self.logs,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "config": self.config.__dict__,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.logs = ckpt.get("logs", [])
        self.episode_rewards = ckpt.get("episode_rewards", [])
        self.episode_lengths = ckpt.get("episode_lengths", [])

