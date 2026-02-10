# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import get_model, get_model_loader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)

from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _get_model(self) -> nn.Module:
        # Draft models may be quantized or on different parallelism,
        # so we load them with a modified vllm config
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config: VllmConfig = create_vllm_config_for_draft_model(self.vllm_config)
        logger.info(
            "Starting to load draft model %s. TP=%d, rank=%d",
            temp_vllm_config.model_config.model,
            temp_vllm_config.parallel_config.tensor_parallel_size,
            temp_vllm_config.parallel_config.rank,
        )

        # Check if we need vocabulary remapping (pruned lm_head)
        needs_vocab_remapping = self.speculative_config.needs_draft_vocab_remapping()

        with set_model_tag("draft_model"):
            if needs_vocab_remapping:
                model = load_pruned_draft_model(
                    vllm_config=temp_vllm_config,
                    target_vocab_size=self.vllm_config.model_config.get_vocab_size(),
                    device=self.device,
                )
            else:
                model = get_model(
                    vllm_config=temp_vllm_config, prefix="draft_model"
                )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass

# TODO add prefix argument to the function to be consistent with get_model()
# TODO figure out how to register loaders for draft models
def load_pruned_draft_model(
    vllm_config: VllmConfig,
    target_vocab_size: int,
    device: torch.device,
) -> nn.Module:
    """Load a draft model with pruned vocabulary (smaller lm_head).

    This function creates a dynamic model class that:
    1. Has the correct lm_head size for the pruned vocabulary
    2. Includes compute_logits remapping to target vocabulary space

    Args:
        vllm_config: VllmConfig for the draft model
        target_vocab_size: Vocabulary size of the target model
        device: Device to load the d2t mapping to

    Returns:
        The loaded model with vocabulary remapping enabled
    """
    from vllm.model_executor.model_loader.utils import get_model_cls

    # Get the base model class (e.g., LlamaForCausalLM)
    base_model_cls = get_model_cls(vllm_config.model_config)

    # Get draft vocabulary info from config
    hf_config = vllm_config.model_config.hf_config
    draft_vocab_size = hf_config.draft_vocab_size
    hidden_size = hf_config.hidden_size

    # Create dynamic model class with pruned lm_head
    pruned_model_cls = create_pruned_draft_model_class(
        base_model_cls=base_model_cls,
        draft_vocab_size=draft_vocab_size,
        hidden_size=hidden_size,
        target_vocab_size=target_vocab_size,
    )

    # Load the model using the custom class
    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    device_config = vllm_config.device_config

    load_device = (
        device_config.device if load_config.device is None else load_config.device
    )
    target_device = torch.device(load_device)

    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                model_class=pruned_model_cls,
                prefix="draft_model",
            )

        loader = get_model_loader(load_config)
        loader.load_weights(model, model_config)
        process_weights_after_loading(model, model_config, target_device)

    logger.info(
        "Loaded pruned draft model with vocabulary remapping: "
        "draft_vocab_size=%d -> target_vocab_size=%d",
        draft_vocab_size,
        target_vocab_size,
    )

    return model.eval()


def create_pruned_draft_model_class(
    base_model_cls: type[nn.Module],
    draft_vocab_size: int,
    hidden_size: int,
    target_vocab_size: int,
) -> type[nn.Module]:
    """Create a dynamic model class with pruned lm_head and vocab remapping.

    This factory creates a subclass of the base model that:
    1. Replaces lm_head with correct size for pruned vocabulary
    2. Replaces logits_processor with correct vocabulary size
    3. Overrides compute_logits to remap to target vocabulary

    Args:
        base_model_cls: The base model class (e.g., LlamaForCausalLM)
        draft_vocab_size: Size of the pruned output vocabulary
        hidden_size: Model hidden size for lm_head
        target_vocab_size: Size of the target model's vocabulary

    Returns:
        A dynamically created model class with pruned vocabulary support
    """

    class PrunedDraftModel(base_model_cls):
        """Dynamic model class with pruned lm_head and vocabulary remapping."""

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            # Initialize the base model normally
            super().__init__(vllm_config=vllm_config, prefix=prefix)

            # Replace lm_head with correct pruned size
            # (matching Eagle3 pattern)
            # TODO is if get_pp_rank required here?
            # TODO handle quant_config?
            # TODO should we delete the old lm_head to free memory?
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )

            # Replace logits_processor with correct vocab size
            logit_scale = getattr(
                vllm_config.model_config.hf_config, "logit_scale", 1.0
            )
            self.logits_processor = LogitsProcessor(draft_vocab_size, scale=logit_scale)

            # Store the d2t mapping as a parameter (matching Eagle3)
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )

            # Store vocab sizes for compute_logits
            self._draft_vocab_size = draft_vocab_size
            self._target_vocab_size = target_vocab_size

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Compute logits and remap to target vocabulary space."""
            # Get logits in draft vocabulary space using base implementation
            logits = self.logits_processor(self.lm_head, hidden_states)

            # Compute target indices on the same device as logits
            # (same as Eagle3 pattern: offset-based mapping)
            base = torch.arange(self._draft_vocab_size, device=logits.device)
            targets = base + self.draft_id_to_target_id

            # Remap to target vocabulary space
            logits_remapped = logits.new_full(
                (logits.shape[0], self._target_vocab_size),
                float("-inf"),
            )
            logits_remapped[:, targets] = logits
            return logits_remapped

    # Give the class a meaningful name for debugging
    PrunedDraftModel.__name__ = f"Pruned{base_model_cls.__name__}"
    PrunedDraftModel.__qualname__ = f"Pruned{base_model_cls.__qualname__}"

    return PrunedDraftModel
