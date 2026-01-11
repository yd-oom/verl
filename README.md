# 在Verl中实现Qwen3-VL的GAD

## 参考脚本
```bash
#!/bin/bash
# 简化版 Qwen3-VL GAD 训练脚本 (单节点FSDP)
# 适用于Qwen3-VL-8B及以下模型

set -x

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export NCCL_TIMEOUT=36000


MODEL_PATH=${MODEL_PATH:-"/mnt/exp/output/generator_warmup/v3-20251228-085621/checkpoint-1200"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"/mnt/exp/output/discriminator_warmup/v2-20251228-194939/checkpoint-600"}

TRAIN_FILES=${TRAIN_FILES:-"/mnt/exp/datasets/gad_single_image_train.parquet"}
VAL_FILES=${VAL_FILES:-"/mnt/exp/datasets/gad_single_image_test.parquet"}

# ============= 实验配置 =============
EXP_NAME=${EXP_NAME:-"qwen3_vl_gad_exp1"}
SAVE_DIR=${SAVE_DIR:-"/tmp/${EXP_NAME}"}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-"qwen3_vl_gad_exp"}

# ============= 训练超参数 =============
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-100}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}

# GAD关键参数: 每个prompt采样多个response
ROLLOUT_N=${ROLLOUT_N:-8}  

# 学习率设置
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-6}

# ============= 推理引擎配置 =============
GEN_TP=${GEN_TP:-2}  # Rollout推理的tensor并行度

# ============= 运行训练 =============
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_files=$VAL_FILES \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    critic.model.path=$REWARD_MODEL_PATH \
    reward_model.use_reward_loop=False \
    critic.model.use_remove_padding=True \
    critic.optim.lr=$CRITIC_LR \
    critic.ppo_max_token_len_per_gpu=12288 \
    critic.grad_clip=0.2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${SWANLAB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=4 \
    trainer.default_local_dir=$SAVE_DIR "$@"

echo "Training completed! Model saved to: ${SAVE_DIR}"

```