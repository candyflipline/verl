from typing import Any, Dict, cast

import ray
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.rollout.async_server import async_server_class


class AtroposInferenceServer:
    """
    Inference server for Atropos.

    Uses core logic from AsyncLLMServerManager, but without scheduler
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """
        Initialize AtroposInferenceServer.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
        """
        self.config = config
        self.worker_group = worker_group

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        workers_info = cast(Dict[int, str], workers_info)
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )

        rollout_db_size = self.config.rollout.db_size
        unready_dp_ranks = set(range(rollout_db_size))

        # Start all server instances
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(  # type: ignore
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(  # type: ignore
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }
            # TODO: double check the type of servers
            servers = cast(Dict[int, Any], servers)

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    address = cast(str, address)
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])
