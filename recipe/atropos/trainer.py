# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from recipe.atropos.client import AtroposClient
from recipe.atropos.schemas import DeviceNameTypes
from recipe.atropos.server import AtroposInferenceServer
from verl import DataProto
from verl.single_controller.ray.base import RayWorkerGroup, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdaptiveKLController, agg_loss, get_kl_controller
from verl.trainer.ppo.metric_utils import (  # compute_timing_metrics, # TODO: add later
    compute_data_metrics,
    compute_throughout_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayClassWithInitArgs,
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward  # compute_reward_async # TODO: add later
from verl.utils.tracking import ValidationGenerationsLogger

if TYPE_CHECKING:
    # Reward Manager type check
    from verl.workers.reward_manager import (
        BatchRewardManager,
        DAPORewardManager,
        NaiveRewardManager,
        PrimeRewardManager,
    )


local_logger = logging.getLogger(__file__)
local_logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextmanager
def _timer_atropos(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayAtroposTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # GRPO doesn't need critic
    USE_CRITIC = False

    # Overrides RayPPOTrainer.__init__
    def __init__(
        self,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[
            Union["BatchRewardManager", "DAPORewardManager", "NaiveRewardManager", "PrimeRewardManager"]
        ] = None,
        device_name: DeviceNameTypes = DeviceNameTypes.CUDA,
    ):
        # -- Rewriting the init method of RayPPOTrainer --
        self.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.config = config

        self.reward_fn = reward_fn

        # -- Engine --
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.use_rm = Role.RewardModel in role_worker_mapping

        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.device_name = device_name.value
        self.validation_generations_logger = ValidationGenerationsLogger()

        # -- In-reward KL control --
        if config.algorithm.use_kl_in_reward:
            kl_type = config.algorithm.kl_ctrl.type
            assert kl_type == "adaptive", "Only adaptive KL controller is supported for Atropos"
            kl_ctrl_in_reward = get_kl_controller(kl_type)
            assert isinstance(kl_ctrl_in_reward, AdaptiveKLController), (
                "Only adaptive KL controller is supported for Atropos"
            )
            self.kl_ctrl_in_reward = kl_ctrl_in_reward

        # GRPO doesn't need it
        supported_adv_estimators = [
            AdvantageEstimator.GAE,
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]
        assert self.config.algorithm.adv_estimator in supported_adv_estimators, (
            f"Advantage estimator {self.config.algorithm.adv_estimator} not supported"
        )

        # -- Init Atropos Client --
        self.atropos_client = AtroposClient(verl_config=config, tokenizer=tokenizer)

        self._validate_config()

    # Overrides RayPPOTrainer._validate_config
    def _validate_config(self) -> None:
        config = self.config

        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # skip original mutual exclusivity, we'll handle it on config level

        # 2. Actor settings
        dynamic_batches = config.actor_rollout_ref.actor.use_dynamic_bsz
        assert dynamic_batches, "Dynamic batching is required for Atropos"

        actor_loss_agg_mode = config.actor_rollout_ref.actor.loss_agg_mode
        assert actor_loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {actor_loss_agg_mode}"

        actor_strategy = config.actor_rollout_ref.actor.strategy
        actor_ulysses_sequence_parallel_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        ref_ulysses_sequence_parallel_size = config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if actor_strategy == "fsdp" and (
            actor_ulysses_sequence_parallel_size > 1 or ref_ulysses_sequence_parallel_size > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        # Check if do_sample is enabled when using validation
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # Check for multi turn tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, (
                "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        # 3. Validate key values to launch Atropos
        atropos_enabled = config.atropos.enable
        # If enabled
        assert atropos_enabled, "Atropos must be enabled"
        # Required total_training_steps for fit()
        assert config.atropos.num_steps is not None, "num_steps must be set"
        assert config.atropos.num_steps > 0, "num_steps must be greater than 0"
        self.total_training_steps = config.atropos.num_steps

        local_logger.info("[validate_config] All configuration checks passed successfully!")

    def init_workers(self):
        """
        Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create reference policy
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # TODO: Check if Atropos needs this
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # Initialize WorkerGroup
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            class_dict = cast(dict[str, RayClassWithInitArgs], class_dict)
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,  # type: ignore
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = True
        self.async_rollout_manager = AtroposInferenceServer(
            config=self.config.actor_rollout_ref,
            worker_group=cast(RayWorkerGroup, self.actor_rollout_wg),
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.atropos_client.atropos_cfg.wandb_project,
            experiment_name=self.atropos_client.atropos_cfg.wandb_group,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.current_step = 0
        self.global_steps = 1  # we start from step 1

        self._load_checkpoint()

        progress_bar = tqdm(
            total=cast(int, self.total_training_steps),
            initial=self.current_step,
            desc="Training Progress",
        )
        if self.current_step == 0 and self.total_training_steps > 0:
            self.current_step = 1

        # --- Main Training Loop: Driven by current_step ---
        while self.current_step < self.total_training_steps:
            global_step_str = f"Global Step {self.global_steps}/{self.total_training_steps}"

            # --- Phase 1: Get Batch Data ---
            local_logger.info(f"{global_step_str}: Requesting batch data from Atropos API")
            try:
                batch_atropos_data = self.atropos_client.fetch_batch()
            except Exception as e:
                local_logger.error(f"Error fetching batch from Atropos API: {e}. Retrying...")
                progress_bar.update(0)
                return

            data_batches = batch_atropos_data.convert_to_data_proto(batch_size=5)
            num_of_batches = len(data_batches)

            for idx, batch in enumerate(data_batches):
                batch_str = f"Batch {idx + 1}/{num_of_batches}"
                local_logger.info(f"{global_step_str}: Processing {batch_str}")
                metrics = {}
                timing_raw: Dict[str, float] = {}

                is_last_step = self.current_step >= self.total_training_steps

                # --- Phase 2: PPO Core Logic ---
                with _timer_atropos("step", timing_raw):
                    # This should happen *before* worker calls that depend on specific data distribution.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 2.1 Old Log Prob
                    with _timer_atropos("old_log_prob", timing_raw):
                        old_log_prob_data = self.actor_rollout_wg.compute_log_prob(batch)
                        old_log_prob_data = cast(DataProto, old_log_prob_data)

                        entropys = old_log_prob_data.batch.get("entropys")
                        entropys = cast(Union[torch.Tensor, None], entropys)

                        if entropys is not None:
                            # matches shape of responses, masks only padding within response block
                            response_block_attn_mask = batch.batch["attention_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_block_attn_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics.update({"actor/entropy_loss": entropy_loss.detach().item()})
                            old_log_prob_data.batch.pop("entropys", None)
                        else:
                            local_logger.warning("Warning: No entropys found in batch. Skipping entropy loss.")

                        batch = batch.union(old_log_prob_data)

                    # 2.2 Ref Log Prob
                    has_reflogprob = batch.batch.get("ref_log_prob", None)
                    if self.use_reference_policy:
                        if has_reflogprob:
                            local_logger.info("Using ref_log_prob provided by Atropos API.")
                        else:
                            local_logger.info("No ref_log_prob found in batch. Computing ref_log_prob...")
                            with _timer_atropos("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob_data = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob_data = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                ref_log_prob_data = cast(DataProto, ref_log_prob_data)
                                batch = batch.union(ref_log_prob_data)

                    # 2.3 Advantage Calculation
                    has_advantages = batch.batch.get("advantages", None)

                    if not has_advantages:
                        local_logger.info("Advantages not from API, computing them...")
                        with _timer_atropos("reward", timing_raw):
                            local_logger.info("Computing reward...")

                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            reward_tensor = cast(DataProto, reward_tensor)
                            batch = batch.union(reward_tensor)

                            # TODO: Currently only sync supported, add async support later
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)  # type: ignore

                        with _timer_atropos("advantage", timing_raw):
                            # calculate token-level scores
                            reward_extra_infos_dict: dict[str, List[Any]] = {}
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards, apply_kl_penalty if available
                            multi_turn = batch.meta_info.get(
                                "multi_turn",
                                self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            )  # type: ignore
                            multi_turn = cast(bool, multi_turn)

                            if self.config.algorithm.use_kl_in_reward:
                                local_logger.info("Applying KL penalty to rewards.")
                                penalty = cast(str, self.config.algorithm.kl_penalty)

                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=penalty,
                                    multi_turn=multi_turn,
                                )
                                metrics.update(kl_metrics)
                            else:
                                local_logger.info("No KL penalty applied to rewards.")
                                assert "token_level_scores" in batch.batch, (
                                    "token_level_scores missing before assignment to token_level_rewards"
                                )
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=1,
                                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                                multi_turn=multi_turn,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            )
                            batch = cast(DataProto, batch)
                    else:
                        local_logger.info("Using advantages provided by Atropos API.")

                    # --- Phase 3: Actor Update ---
                    with _timer_atropos("update_actor", timing_raw):
                        assert "advantages" in batch.batch, "Actor update requires 'advantages'"
                        assert "old_log_probs" in batch.batch, "Actor update requires 'old_log_probs'"

                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output = cast(DataProto, actor_output)

                    # --- Save Checkpoint ---
                    save_freq = self.config.trainer.save_freq
                    is_save_time = self.global_steps % save_freq == 0
                    if save_freq > 0 and (is_last_step or is_save_time):
                        self._save_checkpoint()

                    # Update metrics
                    actor_metrics = actor_output.meta_info.get("metrics", {})
                    actor_metrics = cast(Dict[str, Any], actor_metrics)
                    metrics.update(reduce_metrics(actor_metrics))

                # --- Phase 4: Logging ---
                metrics.update(
                    {
                        "training/current_step": self.current_step,
                        "training/global_step": self.global_steps,
                    }
                )
                # Collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.USE_CRITIC))

                # TODO: Fix _timer names or rework the compute_timing_metrics function
                # metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.current_step)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    local_logger.info(f"\nTarget training steps ({self.total_training_steps}) reached.")
                    break

                # execute for each batch
                self.global_steps += 1

            # execute when batches processed
            self.current_step += 1

        progress_bar.close()
        local_logger.info(f"Training completed. Final global step: {self.current_step}")
