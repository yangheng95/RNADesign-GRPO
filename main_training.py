"""
Main training (GRPO) wired to OmniGenomeSeq2SeqPolicy.
Enhanced with memory management to prevent GPU memory leaks.
"""
import os, json, argparse, random, gc
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer

from grpo_trainer import GRPOTrainer, GRPOConfig
from rna_environment import RNADesignEnvironment, EnvConfig
from reward_calculator import RNARewardCalculator, RewardWeights, LMScorer
from omnigenome_model import create_omnigenome_model, OmniGenomeSeq2SeqPolicy
from config_utils import (
    ExperimentConfig, setup_logging, set_random_seed, save_config, load_config,
    plot_training_curves, EarlyStopping, create_target_structures, validate_rna_structure
)

def create_reward_weights(exp_config: ExperimentConfig) -> RewardWeights:
    return RewardWeights(
        perplexity_score=exp_config.perplexity_score_weight,
        structure_similarity=exp_config.structure_similarity_weight,
        thermodynamic_stability=exp_config.thermodynamic_stability_weight,
        base_pairing_accuracy=exp_config.base_pairing_accuracy_weight,
        loop_structure_penalty=exp_config.loop_structure_penalty_weight,
        gc_content_reward=exp_config.gc_content_reward_weight,
        diversity_bonus=exp_config.diversity_bonus_weight
    )

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main(args):
    config = load_config(args.config) if getattr(args, "config", None) else ExperimentConfig()
    if getattr(args, 'device', None) and args.device != 'auto':
        config.device = args.device

    # 启用内存优化模式
    if getattr(args, 'memory_efficient', False):
        config.batch_size = min(config.batch_size, 8)  # 减小批大小
        config.update_frequency = max(config.update_frequency, 4)  # 减少更新频率

    scoring_model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
    lm_scorer = LMScorer(
        model_name=scoring_model_name,
        tokenizer_name=scoring_model_name,
        device=config.device,
        trust_remote_code=True,
    )
    set_random_seed(config.seed)
    logger = setup_logging(config)
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.result_save_path, exist_ok=True)
    save_config(config, os.path.join(config.result_save_path, "config.yaml"))

    # Base predictor for rewards/env
    base_predictor = create_omnigenome_model(model_type="mock", device=config.device)

    # Seq2Seq RL policy
    model = OmniGenomeSeq2SeqPolicy(
        base_predictor=base_predictor, device=config.device,
    )

    # env + reward
    reward_weights = create_reward_weights(config)
    reward_calc = RNARewardCalculator(base_predictor, reward_weights, lm_scorer=lm_scorer,)
    env = RNADesignEnvironment(
        reward_calculator=reward_calc,
        config=EnvConfig(device=config.device, use_step_shaping=config.use_step_shaping),
        tokenizer=tokenizer
    )

    # targets
    targets = create_target_structures()

    grpo_cfg = GRPOConfig(
        learning_rate=config.learning_rate,
        clip_eps=0.2,
        kl_coef=0.01,
        epochs=1000,
        group_size=max(1, config.batch_size // 4),
        max_steps=config.max_sequence_length,
        clip_grad_norm=config.clip_grad_norm,
        device=config.device,
        temperature=1.0,
    )
    trainer = GRPOTrainer(model, env, grpo_cfg)
    early_stopping = EarlyStopping(patience=50, min_delta=1e-3)

    logger.info("开始基于Seq2Seq+ValueHead的极简GRPO训练 for RNA设计")
    training_stats = {"avg_reward": [], "loss": [], "kl": []}
    best_avg_reward = -1e9

    try:
        total_updates = max(1, config.max_episodes // config.update_frequency)
        for upd in range(1, total_updates + 1):
            # 使用内存高效的上下文管理器
            k = min(len(targets), max(2, config.batch_size // 2))
            batch_targets = random.sample(targets, k=k)
            # 收集经验
            groups = trainer.collect_by_group(batch_targets)
            # 策略更新
            logs = trainer.update_policy(groups)
            # 清理groups数据
            del groups

            training_stats['avg_reward'].append(logs.get('avg_reward', 0.0))
            training_stats['loss'].append(logs.get('loss', 0.0))
            training_stats['kl'].append(logs.get('kl', 0.0))

            if upd % 5 == 0:
                logger.info(f"[Upd {upd}] avg_reward={logs['avg_reward']:.4f} loss={logs['loss']:.4f} kl={logs['kl']:.5f}")

            if upd % 10 == 0:
                # 评估时也要注意显存管理
                with torch.no_grad():
                    eval_stats = evaluate_model(model, env, num_episodes=5)
                    avg_eval_reward = eval_stats['avg_reward']

                if avg_eval_reward > best_avg_reward:
                    best_avg_reward = avg_eval_reward
                    best_path = os.path.join(config.model_save_path, "best_model.pt")
                    trainer.save(best_path)
                    logger.info(f"保存最佳模型: {best_path} (avg_eval_reward={avg_eval_reward:.3f})")
                if early_stopping(avg_eval_reward):
                    logger.info(f"早停在更新 {upd}")
                    break

            if upd % max(1, config.save_frequency // max(1, config.update_frequency)) == 0:
                ckpt_path = os.path.join(config.model_save_path, f"checkpoint_upd_{upd}.pt")
                trainer.save(ckpt_path)
                with open(os.path.join(config.result_save_path, "training_stats.json"), 'w') as f:
                    json.dump(training_stats, f, indent=2, default=str)
                plot_training_curves({
                    'episode_rewards': trainer.episode_rewards,
                    'episode_lengths': trainer.episode_lengths,
                    'policy_losses': training_stats['loss'],
                }, os.path.join(config.result_save_path, "training_curves.png"))

    except KeyboardInterrupt:
        logger.info("训练被中断")
    finally:
        final_path = os.path.join(config.model_save_path, "final_model.pt")
        trainer.save(final_path)
        results = {
            "training_stats": training_stats,
            "best_avg_reward": best_avg_reward,
            "config": config.__dict__,
        }
        with open(os.path.join(config.result_save_path, "final_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("训练结束")

@torch.inference_mode()
def evaluate_model(
    model,
    env,
    targets: Optional[List[str]] = None,
    num_episodes: int = 10,
    success_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    评估模型性能（贪心策略）。自动适配 vectorized 环境与 Seq2Seq policy。
    返回：avg_reward / avg_similarity / avg_length / success_rate。
    """
    if targets is None or len(targets) == 0:
        from config_utils import create_target_structures
        targets = create_target_structures()[:num_episodes]
    else:
        targets = targets[:num_episodes]

    stats = {
        "avg_reward": 0.0,
        "avg_structure_similarity": 0.0,
        "avg_length": 0.0,
        "success_rate": 0.0,
    }

    total_reward = 0.0
    total_similarity = 0.0
    total_length = 0
    success_count = 0

    model.eval()

    for target in targets:
        if hasattr(model, "set_target"):
            model.set_target(target)

        try:
            state = env.reset(target, batch_size=1)  # Vectorized env
            vectorized = True
        except TypeError:
            state = env.reset()  # Legacy single-sample env
            vectorized = False

        episode_reward = 0.0
        episode_len = 0
        done = False
        last_info = {}

        while not done:
            if state.dim() == 1:
                state = state.unsqueeze(0)

            with torch.amp.autocast(device_type=state.device.type, dtype=torch.float16):
                logits, _ = model.get_policy_output(state)

            action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

            if vectorized:
                state, rewards, dones, infos = env.step([action])
                reward = float(rewards[0])
                done = bool(dones[0])
                last_info = infos[0]
            else:
                state, reward, done, last_info = env.step(action)

            episode_reward += reward
            episode_len += 1

        total_reward += episode_reward
        total_length += episode_len

        # compute structure similarity if available
        sim = None
        try:
            if 'sequence' in last_info and hasattr(env, 'rc') and hasattr(env.rc, 'sp'):
                seq = last_info['sequence']
                pred_struct = env.rc.sp.predict_structure(seq)
                sim = f1_pairs(pred_struct, target)
            elif 'structure_similarity' in last_info:
                sim = float(last_info['structure_similarity'])
        except Exception:
            pass

        if sim is not None:
            total_similarity += sim
            if sim >= success_threshold:
                success_count += 1

    n = len(targets)
    stats['avg_reward'] = total_reward / max(1, n)
    stats['avg_structure_similarity'] = total_similarity / max(1, n)
    stats['avg_length'] = total_length / max(1, n)
    stats['success_rate'] = success_count / max(1, n)

    model.train()
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA序列设计 - Seq2Seq+GRPO 训练脚本")
    parser.add_argument("--config", type=str, help="配置文件路径", default=None)
    parser.add_argument("--device", type=str, choices=["cpu","cuda","auto"], default="auto")
    parser.add_argument("--memory-efficient", action="store_true", help="启用显存优化模式")

    args = parser.parse_args()
    main(args)
