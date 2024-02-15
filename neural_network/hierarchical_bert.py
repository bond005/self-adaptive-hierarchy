from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_outputs import ModelOutput


@dataclass
class HierarchicalSequenceEmbedderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.FloatTensor = None
    layer_embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HierarchicalSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None
    layer_embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HierarchicalBertConfig(BertConfig):
    model_type = "hierarchical-bert"

    def __init__(self, label_smoothing: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing


class BertHierarchicalClassificationHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


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


class BertForHierarchicalEmbedding(BertPreTrainedModel, ABC):
    config_class = HierarchicalBertConfig

    def __init__(self, config: HierarchicalBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.temperature = config.temperature
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.layer_weights = LayerGatingNetwork(in_features=config.num_hidden_layers)

        self.init_weights()

    def init_weights(self):
        super().init_weights()
        with torch.no_grad():
            self.layer_weights.reset_parameters()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            input_ids_2: Optional[torch.LongTensor] = None,
            attention_mask_2: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HierarchicalSequenceEmbedderOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=False
        )
        cls_hidden_states = torch.stack(
            tensors=outputs[2][-self.config.num_hidden_layers:],
            dim=1
        )[:, :, 0, :]
        cls_emb = self.layer_weights(cls_hidden_states.permute(0, 2, 1))[:, :, 0]

        loss = None
        if labels is not None:
            cls_emb_ = cls_emb.view(-1, self.config.hidden_size)
            emb_norm = torch.linalg.norm(cls_emb_, dim=-1, keepdim=True) + 1e-9
            if (input_ids_2 is None) or (attention_mask_2 is None):
                err_msg = 'The second texts (their input IDs and attention masks) in the pairs are not specified!'
                raise ValueError(err_msg)
            outputs_2 = self.bert(
                input_ids_2,
                attention_mask=attention_mask_2,
                output_hidden_states=True,
                return_dict=False
            )
            cls_hidden_states_2 = torch.stack(
                tensors=outputs_2[2][-self.config.num_hidden_layers:],
                dim=1
            )[:, :, 0, :]
            cls_emb_2 = self.layer_weights(cls_hidden_states_2.permute(0, 2, 1))[:, :, 0]
            cls_emb_2_ = cls_emb_2.view(-1, self.config.hidden_size)
            cls_emb_2_norm = torch.linalg.norm(cls_emb_2_, dim=-1, keepdim=True) + 1e-9
            distances = torch.norm(cls_emb_ / emb_norm - cls_emb_2_ / cls_emb_2_norm, 2, dim=-1)
            loss_fct = DistanceBasedLogisticLoss()
            loss = loss_fct(distances, labels.view(-1))

        if not return_dict:
            output = (cls_emb, cls_hidden_states) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return HierarchicalSequenceEmbedderOutput(
            loss=loss,
            embeddings=cls_emb,
            layer_embeddings=cls_hidden_states,
            hidden_states=outputs[2],
            attentions=outputs[3] if output_attentions else None,
        )

    @property
    def layer_importances(self) -> List[Tuple[int, float]]:
        with torch.no_grad():
            importances = torch.softmax(self.layer_weights.weight, dim=-1).detach().cpu().numpy().flatten()
        indices_and_importances = []
        for layer_idx in range(importances.shape[0]):
            indices_and_importances.append((layer_idx + 1, float(importances[layer_idx])))
        indices_and_importances.sort(key=lambda it: (-it[1], it[0]))
        return indices_and_importances


class BertForHierarchicalSequenceClassification(BertForHierarchicalEmbedding, ABC):
    def __init__(self, config: HierarchicalBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.label_smoothing = config.label_smoothing
        self.config = config

        self.classifier = BertHierarchicalClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            right_input_ids: Optional[torch.LongTensor] = None,
            right_attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HierarchicalSequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.label_smoothing is None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs
            return ((loss,) + output) if loss is not None else output

        return HierarchicalSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            embeddings=outputs.embeddings,
            layer_embeddings=outputs.layer_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


AutoConfig.register("hierarchical-bert", HierarchicalBertConfig)
AutoModelForSequenceClassification.register(
    HierarchicalBertConfig,
    BertForHierarchicalSequenceClassification
)
