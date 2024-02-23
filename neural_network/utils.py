import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class DistanceBasedLogisticLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(DistanceBasedLogisticLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.to(inputs.dtype).view(-1)
        p = self.distance_to_probability(inputs)
        return torch.nn.functional.binary_cross_entropy(input=p, target=targets, reduction=self.reduction)

    @staticmethod
    def distance_to_probability(distance: torch.Tensor) -> torch.Tensor:
        p = 1.3678793907165527 * torch.sigmoid(1.0 - distance)
        return p


class LayerGatingNetwork(torch.nn.Module):
    __constants__ = ['in_features']
    in_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.empty((1, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initial_layer_weights = np.array(
            [1.0 / (self.in_features - layer_idx) for layer_idx in range(self.in_features)],
            dtype=np.float32
        )
        initial_layer_weights /= np.sum(initial_layer_weights)
        initial_layer_weights_pt = torch.tensor(
            initial_layer_weights.reshape((1, self.in_features)),
            dtype=self.weight.dtype,
            device=self.weight.device
        )
        del initial_layer_weights
        self.weight = torch.nn.Parameter(initial_layer_weights_pt)
        del initial_layer_weights_pt

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, torch.softmax(self.weight, dim=-1))

    def extra_repr(self) -> str:
        return 'in_features={}'.format(self.in_features)


class HierarchyPrinterCallback(TrainerCallback):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        layer_importances = kwargs['model'].layer_importances
        n = min(len(layer_importances), self.num_layers)
        print(f'The top-{self.num_layers} most important layers are: {layer_importances[0:n]}')
