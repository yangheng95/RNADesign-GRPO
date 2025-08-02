
"""
配置与通用工具（适配 Seq2Seq + GRPO、向量化环境）

变更要点：
- 兼容你现有工程结构与接口，但评估逻辑已适配新的向量化环境与 Seq2Seq 策略封装。
- evaluate_model：支持基于目标结构的评估（默认使用内置样例），自动调用 model.set_target(target)；
  对向量化环境（batch）与旧式单样本环境均可运行。
- 训练曲线绘制保留原字段名，新增对 'avg_reward' / 'kl' 等字段的兼容显示。
"""

import yaml
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



@dataclass
class ExperimentConfig:
    """实验配置（保留向后兼容字段，未用字段会被忽略）"""
    # 模型参数
    hidden_dim: int = 720                # 用于Seq2Seq的d_model
    max_sequence_length: int = 300

    # 模型选择
    model_name: str = "OmniGenomeS"

    # GRPO参数（核心项）
    learning_rate: float = 5e-5
    batch_size: int = 32
    max_episodes: int = 1000
    update_frequency: int = 10
    clip_grad_norm: float = 1.0
    use_step_shaping: bool = True

    # 兼容旧PPO字段（当前不使用，仅为不破坏外部引用）
    gamma: float = 0.99
    entropy_coeff: float = 0.0
    value_loss_coeff: float = 0.0

    # 奖励权重（保留接口，真实权重在 reward_calculator 中解释）
    perplexity_score_weight = 1.0
    structure_similarity_weight: float = 1
    thermodynamic_stability_weight: float = 0.0
    base_pairing_accuracy_weight: float = 0.0
    loop_structure_penalty_weight: float = 0.0
    gc_content_reward_weight: float = 0.0
    diversity_bonus_weight: float = 0.0

    # 训练设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_frequency: int = 100
    log_frequency: int = 10

    # 路径设置
    model_save_path: str = "models"
    log_save_path: str = "logs"
    result_save_path: str = "results"


def setup_logging(config: ExperimentConfig):
    """设置日志系统（文件+控制台，UTF-8 兼容 Windows 控制台）。"""
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    os.makedirs(config.log_save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_save_path, f"training_{timestamp}.log")
    # force=True 确保重新配置生效
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger("config_utils")


def set_random_seed(seed: int):
    """设置随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_config(config: ExperimentConfig, path: str):
    """保存配置到YAML。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)


def load_config(path: str) -> ExperimentConfig:
    """从YAML加载配置。"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)


def plot_training_curves(stats: Dict[str, list], save_path: str):
    """绘制训练曲线（兼容多种统计键）。"""
    # 自动检测需要绘制的子图
    keys = set(stats.keys())
    # 2x2布局：奖励/损失/熵/长度
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 奖励曲线（支持 episode_rewards 或 avg_reward）
    if 'episode_rewards' in keys or 'avg_reward' in keys:
        y = stats.get('episode_rewards', stats.get('avg_reward', []))
        axes[0, 0].plot(y)
        axes[0, 0].set_title('Rewards')
        axes[0, 0].set_xlabel('Step/Update')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

    # 损失曲线（policy_losses / loss / kl）
    if 'policy_losses' in keys or 'loss' in keys or 'kl' in keys:
        if 'policy_losses' in keys:
            axes[0, 1].plot(stats['policy_losses'], label='Policy Loss')
        if 'loss' in keys:
            axes[0, 1].plot(stats['loss'], label='Loss')
        if 'kl' in keys:
            axes[0, 1].plot(stats['kl'], label='KL')
        axes[0, 1].set_title('Loss / KL')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # 熵曲线（可选）
    if 'entropy' in keys:
        axes[1, 0].plot(stats['entropy'])
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True)

    # 序列长度或episode长度
    if 'episode_lengths' in keys:
        axes[1, 1].plot(stats['episode_lengths'])
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Length')
        axes[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    plt.close()


class EarlyStopping:
    """早停机制。"""
    def __init__(self, patience: int = 50, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def create_target_structures() -> list:
    """创建一些测试用的目标结构（与旧版保持一致）。"""
    structures = [
        # 简单发夹环
        "((((....))))",
        "((((.......))))",
        "((((((.......))))))",

        # 多个环
        "((((...))))..(((...)))",
        "(((...)))...(((...)))",

        # 复杂结构
        "((((..(((...)))..))))",
        "((..((...))..))..(((...)))",

        # 长结构
        "(((((...(((....)))...)))))...((((....))))",
    ]
    return structures


def validate_rna_structure(structure: str) -> bool:
    """验证RNA结构字符串的有效性：字符合法且括号匹配。"""
    stack = []
    valid_chars = set(['(', ')', '.'])
    if not all(c in valid_chars for c in structure):
        return False
    for ch in structure:
        if ch == '(':
            stack.append(ch)
        elif ch == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0
