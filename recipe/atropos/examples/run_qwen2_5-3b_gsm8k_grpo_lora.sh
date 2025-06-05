set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# -- Atropos Config --
atropos_enable=True
atropos_host=http://localhost:8000
atropos_wandb_group=atropos_examples
atropos_wandb_project=atropos_examples
atropos_batch_size=2
atropos_max_token_len=2048
atropos_checkpoint_dir=null
atropos_save_checkpoint_interval=null
atropos_starting_step=0
atropos_num_steps=10

# -- VeRL Config --
# Algorithm
adv_estimator=grpo
use_kl_in_reward=False

# Rollout Ref Model
model_path=Qwen/Qwen2.5-3B-Instruct
use_shm=True
lora_rank=64
lora_alpha=32
use_remove_padding=True
enable_gradient_checkpointing=True

# Rollout Ref Actor
optim_lr=3e-6
ppo_mini_batch_size=256
ppo_micro_batch_size_per_gpu=40
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl
entropy_coeff=0
fsdp_config_param_offload=False
fsdp_config_optimizer_offload=False

# Rollout Ref Rollout
log_prob_micro_batch_size_per_gpu=40
tensor_model_parallel_size=2
rollout_name=sglang
gpu_memory_utilization=0.6
n=5
load_format=safetensors
layered_summon=True

# Rollout Ref Ref
log_prob_micro_batch_size_per_gpu=40
fsdp_config_param_offload=True

# Trainer
critic_warmup=0
# TIP: add wandb logger
logger=['console']
project_name='verl_examples'
experiment_name='qwen2.5_3b_grpo_lora'
n_gpus_per_node=8
nnodes=1
save_freq=20
test_freq=5
total_epochs=15


python3 -m recipe.atropos.main_atropos \
    atropos.enable=${atropos_enable} \
    atropos.atropos_host=${atropos_host} \
    atropos.wandb_group=${atropos_wandb_group} \
    atropos.wandb_project=${atropos_wandb_project} \
    atropos.batch_size=${atropos_batch_size} \
    atropos.max_token_len=${atropos_max_token_len} \
    atropos.checkpoint_dir=${atropos_checkpoint_dir} \
    atropos.save_checkpoint_interval=${atropos_save_checkpoint_interval} \
    atropos.starting_step=${atropos_starting_step} \
    atropos.num_steps=${atropos_num_steps} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_shm=${use_shm} \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.optim.lr=${optim_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${fsdp_config_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${fsdp_config_optimizer_offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.load_format=${load_format} \
    actor_rollout_ref.rollout.layered_summon=${layered_summon} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${fsdp_config_param_offload} \
    trainer.critic_warmup=${critic_warmup} \
    trainer.logger=${logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} $@