"""Plain causal SFT with the same per-token segment weights as Beacon."""
import torch
import torch.nn.functional as F
from transformers import Trainer

from ..loss_segments import LossSegmentConfig, segment_ids_to_weights


def segmented_causal_loss(logits, labels, segment_ids, loss_segments):
    """Shift targets and categories together; average by active token weight."""
    weights = segment_ids_to_weights(segment_ids[:, 1:], loss_segments)
    targets = labels[:, 1:]
    weights = weights * (targets != -100).to(weights.dtype)
    active = weights > 0
    # Keep a gradient path even when every segment has been disabled.
    if not active.any():
        return logits[..., :1].sum() * 0.0
    token_loss = F.cross_entropy(logits[:, :-1][active].float(), targets[active], reduction='none')
    return (token_loss * weights[active]).sum() / weights[active].sum()


class SegmentedSFTTrainer(Trainer):
    def __init__(self, *args, loss_segments=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_segments = LossSegmentConfig.from_dict(loss_segments)
        # Loss is already averaged within each microbatch, like ordinary Trainer CE.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        segments = inputs.pop('loss_segment_ids')
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = segmented_causal_loss(outputs.logits, labels, segments, self.loss_segments)
        return (loss, outputs) if return_outputs else loss
