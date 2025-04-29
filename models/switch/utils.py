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


def vivit_to_switch(vivit: VivitForVideoClassification, layers_for_switch: list[int], n_experts, config: VivitConfig):
    # first - replace layers
    for i, l in enumerate(vivit.vivit.encoder.layer):
        if i in layers_for_switch:
            l.intermediate = SwitchFeedForward(
                is_scale_prob=True,
                n_experts=n_experts,
                expert=FeedForward(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    dropout=0,
                    activation=FastGELUActivation(),
                ),
                d_model=config.hidden_size,
                top_k=config["switch_top_k"],
            )
            l.output = torch.nn.Identity()
    return vivit


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


def set_activate_n_expert(model: TimesformerForVideoClassification, activate_n_expert, switch_layers):
    for i, layer in enumerate(model.timesformer.encoder.layer):
        if i in switch_layers:
            model.timesformer.encoder.layer[i].intermediate.activate_only_n_expert = activate_n_expert
    return model


def set_save_token_routes(model, config, save_token_routes):
    for i, layer in enumerate(model.timesformer.encoder.layer):
        if i in config["switch_layers"]:
            model.timesformer.encoder.layer[i].intermediate.save_token_routes = save_token_routes
    return model
