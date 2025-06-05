import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, HttpUrl

from verl.protocol import DataProto

# --- Constants ---
PADDING_MULTIPLE = 64
LABEL_IGNORE_INDEX = -100
NORMALIZATION_EPSILON = 1e-8


# --- Enums ---
class DeviceNameTypes(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, Dict[str, Any]]
    reward: Optional[float]


class AtroposScoredData(BaseModel):
    # Required fields
    tokens: List[List[int]]
    masks: List[List[int]]
    scores: Union[List[float], np.ndarray[np.float32, Any]]

    # Optional fields
    advantages: Optional[List[List[float]]] = None
    ref_logprobs: Optional[List[List[float]]] = None
    messages: Optional[List[List[Message]]] = None
    overrides: Optional[List[Dict[str, Any]]] = None  # Per-item logging overrides
    group_overrides: Optional[Dict[str, Any]] = None  # Group logging overrides
    images: Optional[Any] = None  # Image data (if applicable)


class AtroposScoredDataProcessed(BaseModel):
    """
    Contains fields for training.
    """

    # Mandatory fields
    input_ids: torch.Tensor
    responses: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    def convert_to_data_proto(self) -> DataProto:
        return DataProto.from_single_dict(self.model_dump())


class AtroposRegisterPayload(BaseModel):
    """
    Pydantic model to validate the config values and pass the payload to the register endpoint.
    """

    wandb_group: str = Field(default="default", description="WandB group name")
    wandb_project: str = Field(default="default", description="WandB project name")
    batch_size: int = Field(..., gt=0, description="Batch size")
    max_token_len: int = Field(..., gt=0, description="Max token length expected in trajectories")
    checkpoint_dir: str = Field(..., description="Shared location for checkpoints")
    save_checkpoint_interval: int = Field(..., gt=0, description="Save checkpoint interval")
    starting_step: int = Field(..., gt=0, description="Starting step")
    num_steps: int = Field(..., gt=0, description="Total expected training steps")

    @classmethod
    def from_atropos_config(cls, atropos_cfg: "AtroposConfig") -> "AtroposRegisterPayload":
        return cls(
            wandb_group=atropos_cfg.wandb_group,
            wandb_project=atropos_cfg.wandb_project,
            batch_size=atropos_cfg.batch_size,
            max_token_len=atropos_cfg.max_token_len,
            checkpoint_dir=atropos_cfg.checkpoint_dir,
            save_checkpoint_interval=atropos_cfg.save_checkpoint_interval,
            starting_step=atropos_cfg.starting_step,
            num_steps=atropos_cfg.num_steps,
        )


class AtroposConfig(AtroposRegisterPayload):
    """
    Pydantic model to validate the config values and pass the payload to the register endpoint.
    """

    atropos_host: HttpUrl = Field(default="http://localhost:8000", description="Atropos host")


class AtroposRegisterResponse(BaseModel):
    """
    Expected response from the register endpoint.
    """

    uuid: str = Field(..., description="UUID of the registered trainer")


class AtroposScoredDataBatch(BaseModel):
    """
    This class handles:

    1. Score normalization (z-score)
    2. Efficient padding to GPU-friendly sizes (multiples of 64)
    3. Causal language modeling setup (input[:-1], target[1:])
    4. Proper masking for padded positions
    5. Optional advantage and reference logprob processing
    """

    batch: List[AtroposScoredData]

    def convert_to_data_proto(self, batch_size: int) -> List[DataProto]:
        """
        Pad data and calculate training inputs.

        Args:
            batch_size: Size of each batch

        Returns:
            List of DataProto objects containing all necessary training inputs
        """

        padded_data = self._pad_data_to_good_offset(batch_size)

        token_batches = padded_data["token_batches"]
        label_batches = padded_data["label_batches"]
        advantage_batches = padded_data["advantage_batches"]
        ref_logprobs_batches = padded_data["ref_logprobs_batches"]

        batched_inputs = self._calculate_attention_masks_and_position_ids(
            token_batches,
            label_batches,
        )

        # Create tensor batch
        data_proto_array = [batch.convert_to_data_proto() for batch in batched_inputs]  # type: ignore

        # Add optional values to the batch
        for idx, batch in enumerate(data_proto_array):
            if ref_logprobs_batches:
                batch.batch["ref_log_prob"] = ref_logprobs_batches[idx]
            if advantage_batches:
                batch.batch["advantages"] = advantage_batches[idx]

        return data_proto_array

    def _pad_data_to_good_offset(
        self,
        batch_size: int,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Pads token sequences to multiples of PADDING_MULTIPLE for GPU efficiency and prepares data for training.

        Args:
            batch_size: Size of each batch for processing

        Returns:
            Tuple of (token_batches, label_batches, advantage_batches, ref_logprobs_batches)
            Note: advantage_batches and ref_logprobs_batches may be None/empty
        """
        assert self.batch is not None, "Batch is None"
        assert len(self.batch) > 0, "Batch is empty"
        assert batch_size > 0, "Batch size is not positive"

        max_token_len = max([max([len(x) for x in item.tokens]) for item in self.batch])
        assert max_token_len > 0, "Max token length is not positive"

        # Pad for GPU efficiency
        if (max_token_len - 1) % PADDING_MULTIPLE != 0:
            padded_length = math.ceil((max_token_len - 1) / PADDING_MULTIPLE) * PADDING_MULTIPLE
            token_setup_len = padded_length + 1
        else:
            token_setup_len = max_token_len
            padded_length = max_token_len - 1

        # Initialize lists for batching
        input_ids: List[np.ndarray[np.int32, Any]] = []
        labels: List[np.ndarray[np.int32, Any]] = []

        # Optional values
        advantages: List[np.ndarray[np.float32, Any]] = []
        ref_logprobs: List[np.ndarray[np.float32, Any]] = []

        for item in self.batch:
            assert item.scores is not None, "scores must be provided"
            assert item.tokens is not None, "tokens must be provided"
            assert item.masks is not None, "masks must be provided"

            if item.ref_logprobs is not None:
                assert len(item.ref_logprobs) == len(item.tokens), "ref_logprobs length mismatch with tokens"
            if item.advantages is not None:
                assert len(item.advantages) == len(item.tokens), "advantages length mismatch with tokens"

            # Z-score normalization for scores
            scores = np.array(item.scores, dtype=np.float32)
            if len(scores) > 1:
                mean_score = scores.mean()
                std_score = scores.std()
                scores = (scores - mean_score) / max(std_score, NORMALIZATION_EPSILON)

            # Apply score overrides
            if item.overrides is not None:
                for i, override in enumerate(item.overrides):
                    if override.get("set_advantage_to_zero", False):
                        scores[i] = 0.0

            item.scores = scores

            # Process each sequence
            for i in range(len(item.tokens)):
                assert len(item.tokens[i]) == len(item.masks[i]), f"Token and mask length mismatch at index {i}"

                # Create labels with padding
                label_item = np.concatenate(
                    [
                        np.array(item.masks[i], dtype=np.int32),
                        np.full(
                            max(0, token_setup_len - len(item.tokens[i])),
                            LABEL_IGNORE_INDEX,
                            dtype=np.int32,
                        ),
                    ]
                )

                # Pad tokens with zeros
                item_tokens = np.array(item.tokens[i], dtype=np.int32)
                item_tokens = np.concatenate(
                    [
                        item_tokens,
                        np.zeros(max(0, token_setup_len - len(item_tokens)), dtype=np.int32),
                    ]
                )

                # Causal modeling: input[:-1], target[1:]
                input_ids.append(item_tokens[:-1])
                labels.append(label_item[1:])

                # Optional values
                if item.advantages is not None:
                    item_advantages = np.array(item.advantages[i], dtype=np.float32)
                    assert len(item_advantages) == len(item.tokens[i]), f"advantages length mismatch at index {i}"

                    # Pad advantages
                    item_advantages = np.concatenate(
                        [
                            item_advantages,
                            np.zeros(max(0, token_setup_len - len(item_advantages)), dtype=np.float32),
                        ]
                    )
                    advantages.append(item_advantages)

                if item.ref_logprobs is not None:
                    item_ref_logprobs = np.array(item.ref_logprobs[i], dtype=np.float32)
                    assert len(item_ref_logprobs) == len(item.tokens[i]), f"ref_logprobs length mismatch at index {i}"

                    # Pad ref_logprobs
                    item_ref_logprobs = np.concatenate(
                        [
                            item_ref_logprobs,
                            np.zeros(max(0, token_setup_len - len(item_ref_logprobs)), dtype=np.float32),
                        ]
                    )
                    ref_logprobs.append(item_ref_logprobs[1:])  # Align with labels

        # Create batches
        token_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []
        advantage_batches: List[torch.Tensor] = []
        ref_logprobs_batches: List[torch.Tensor] = []

        # Handle all batches including the last partial batch
        num_sequences = len(input_ids)
        num_batches = math.ceil(num_sequences / batch_size)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sequences)

            # Stack sequences for this batch
            batch_input_ids = np.stack(input_ids[start_idx:end_idx], axis=0)
            batch_labels = np.stack(labels[start_idx:end_idx], axis=0)

            # Convert to tensors
            token_batches.append(torch.tensor(batch_input_ids, dtype=torch.long))
            label_batches.append(torch.tensor(batch_labels, dtype=torch.long))

            # Optional values
            if advantages:
                batch_advantages = np.array(advantages[start_idx:end_idx], dtype=np.float32)
                advantage_batches.append(torch.tensor(batch_advantages, dtype=torch.float32).view(-1, 1))

            if ref_logprobs:
                batch_ref_logprobs = np.stack(ref_logprobs[start_idx:end_idx], axis=0)
                ref_logprobs_batches.append(torch.tensor(batch_ref_logprobs, dtype=torch.float32))

        return {
            "token_batches": token_batches,
            "label_batches": label_batches,
            "advantage_batches": advantage_batches,
            "ref_logprobs_batches": ref_logprobs_batches,
        }

    def _calculate_attention_masks_and_position_ids(
        self,
        token_batches: List[torch.Tensor],
        label_batches: List[torch.Tensor],
    ) -> List[AtroposScoredDataProcessed]:
        """
        Calculate attention masks and position IDs for each batch.

        Args:
            data: Tuple of (token_batches, label_batches, advantage_batches, ref_logprobs_batches)

        Returns:
            List of dictionaries containing processed batch data
        """
        batched_inputs: List[AtroposScoredDataProcessed] = []
        zip_data = zip(token_batches, label_batches)

        for input_ids, responses in zip_data:
            bsz, seq_len = input_ids.shape

            # Attention mask: 1 where we have real tokens (not padding), 0 for padding
            # We identify padding by checking where labels != LABEL_IGNORE_INDEX
            attention_mask = (responses != LABEL_IGNORE_INDEX).long()

            # Position IDs: sequential indices for each position in the sequence
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

            processed_data = AtroposScoredDataProcessed(
                input_ids=input_ids,
                responses=responses,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            batched_inputs.append(processed_data)

        return batched_inputs
