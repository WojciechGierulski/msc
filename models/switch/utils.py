from transformers import TimesformerForVideoClassification, TimesformerConfig, VivitConfig, VivitForVideoClassification
from models.switch.switch_ff import SwitchFeedForward
import torch
from labml_nn.transformers.feed_forward import FeedForward
from transformers.activations import FastGELUActivation, GELUActivation


def timesformer_to_switch(
    timesformer: TimesformerForVideoClassification,
    layers_for_switch: list[int],
    n_experts,
    config: TimesformerConfig,
    switch_top_k,
):
    # first - replace layers
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            l.intermediate = SwitchFeedForward(
                is_scale_prob=False,
                n_experts=n_experts,
                expert=FeedForward(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    dropout=0,
                    activation=GELUActivation(),
                ),
                d_model=config.hidden_size,
                top_k=switch_top_k,
            )
            l.output = torch.nn.Identity()
    return timesformer


def collect_loss(timesformer: TimesformerForVideoClassification, layers_for_switch: list[int]):
    first_term = True
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            f = l.intermediate.route_frac
            p = l.intermediate.route_prob
            if first_term:
                balance_loss = (f * p).sum()
                first_term = False
            else:
                balance_loss += (f * p).sum()
    return balance_loss


def clear_memory(timesformer: TimesformerForVideoClassification, layers_for_switch: list[int]):
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            del l.intermediate.route_prob
            del l.intermediate.route_frac


def set_experts_param(model, switch_layers, param, value):
    for i, layer in enumerate(model.timesformer.encoder.layer):
        if i in switch_layers:
            setattr(model.timesformer.encoder.layer[i].intermediate, param, value)
    return model



def set_grad_except_router(timesformer: TimesformerForVideoClassification, switch_layers, n_layers, val):
    timesformer.timesformer.embeddings.patch_embeddings.projection.weight.requires_grad = val
    timesformer.timesformer.embeddings.patch_embeddings.projection.bias.requires_grad = val
    timesformer.timesformer.embeddings.position_embeddings.requires_grad = val
    timesformer.timesformer.embeddings.time_embeddings.requires_grad = val
    for i in range(n_layers):
        timesformer.timesformer.encoder.layer[i].attention.attention.qkv.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].attention.attention.qkv.bias.requires_grad = val
        timesformer.timesformer.encoder.layer[i].attention.output.dense.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].attention.output.dense.bias.requires_grad = val

        if i in switch_layers:
            for expert in timesformer.timesformer.encoder.layer[i].intermediate.experts:
                expert.layer1.weight.requires_grad = val
                expert.layer1.bias.requires_grad = val
                expert.layer2.weight.requires_grad = val
                expert.layer2.bias.requires_grad = val
        else:
            timesformer.timesformer.encoder.layer[i].intermediate.dense.weight.requires_grad = val
            timesformer.timesformer.encoder.layer[i].intermediate.dense.bias.requires_grad = val
            timesformer.timesformer.encoder.layer[i].output.dense.weight.requires_grad = val
            timesformer.timesformer.encoder.layer[i].output.dense.bias.requires_grad = val

        timesformer.timesformer.encoder.layer[i].layernorm_before.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].layernorm_before.bias.requires_grad = val

        timesformer.timesformer.encoder.layer[i].layernorm_after.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].layernorm_after.bias.requires_grad = val

        timesformer.timesformer.encoder.layer[i].temporal_layernorm.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_layernorm.bias.requires_grad = val

        timesformer.timesformer.encoder.layer[i].temporal_attention.attention.qkv.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_attention.attention.qkv.bias.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_attention.output.dense.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_attention.output.dense.bias.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_dense.weight.requires_grad = val
        timesformer.timesformer.encoder.layer[i].temporal_dense.bias.requires_grad = val

    timesformer.timesformer.layernorm.weight.requires_grad = val
    timesformer.timesformer.layernorm.bias.requires_grad = val

    timesformer.classifier.weight.requires_grad = val
    timesformer.classifier.bias.requires_grad = val

    return timesformer