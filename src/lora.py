"""Assistant-only causal labels for an optional future LoRA experiment.

This utility masks every token before the assistant response. LoRA remains outside
the primary pipeline unless it improves on the frozen VLM under the same protocol.
"""

from __future__ import annotations

from collections.abc import Sequence


def assistant_only_labels(
    input_ids: Sequence[int],
    assistant_start_index: int,
    attention_mask: Sequence[int] | None = None,
    ignore_index: int = -100,
) -> list[int]:
    """Mask prompt and padded positions without confusing EOS with padding.

    A token-value-based padding rule is unsafe because many causal language
    models use the same ID for EOS and PAD. The attention mask is the only
    unambiguous way to preserve a real assistant EOS while masking padding.
    """
    if not 0 <= assistant_start_index <= len(input_ids):
        raise ValueError("assistant_start_index is outside input_ids")
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    if len(attention_mask) != len(input_ids):
        raise ValueError("attention_mask and input_ids must have equal length")
    if any(value not in {0, 1} for value in attention_mask):
        raise ValueError("attention_mask values must be binary")
    return [
        token if index >= assistant_start_index and bool(attention_mask[index]) else ignore_index
        for index, token in enumerate(input_ids)
    ]
