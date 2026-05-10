import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision


def get_cifar_resnet18():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def get_pruned_architecture(pruning_ratio=0.7):
    """
    Reconstruct the physically pruned architecture.
    Must match exactly what torch-pruning did during training.
    """
    model = get_cifar_resnet18()
    example_input = torch.randn(1, 3, 32, 32)

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_input,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=[model.fc],
    )
    pruner.step()

    return model
