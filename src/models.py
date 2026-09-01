"""Small, explicit CNN baselines for the visible-ingredient benchmark.

The benchmark distinguishes a frozen linear probe from a model with its final
block fine-tuned, allowing model complexity to be compared directly.
"""

from __future__ import annotations


def build_resnet18(num_labels: int, pretrained: bool = True, trainable_scope: str = "head"):
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if trainable_scope == "layer4_and_head":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif trainable_scope != "head":
        raise ValueError("trainable_scope must be 'head' or 'layer4_and_head'")
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def transfer_compatible_backbone(source_state: dict, target_model) -> dict[str, list[str]]:
    """Load only equal-shape tensors, allowing a new ontology-specific head."""
    target_state = target_model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    result = target_model.load_state_dict(compatible, strict=False)
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
