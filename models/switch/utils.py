from transformers import TimesformerForVideoClassification, TimesformerConfig, VivitConfig, VivitForVideoClassification
from models.switch.switch_ff import SwitchFeedForward
import torch
from labml_nn.transformers.feed_forward import FeedForward
from transformers.activations import FastGELUActivation, GELUActivation


def timesformer_to_switch(
    timesformer: TimesformerForVideoClassification, layers_for_switch: list[int], n_experts, config: TimesformerConfig
):
    # first - replace layers
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            l.intermediate = SwitchFeedForward(
                is_scale_prob=True,
                n_experts=n_experts,
                expert=FeedForward(
                    d_model=config.hidden_size,
                    d_ff=config.intermediate_size,
                    dropout=0,
                    activation=GELUActivation(),
                ),
                d_model=config.hidden_size,
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
            )
            l.output = torch.nn.Identity()
    return vivit


def collect_route_stats(timesformer: TimesformerForVideoClassification, layers_for_switch: list[int]):
    counts = []
    route_prob = []
    route_prob_max = []
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            f = l.intermediate.counts
            p = l.intermediate.prob_sum
            p_max = l.intermediate.prob_max
            counts.append(f)
            route_prob.append(p)
            route_prob_max.append(p_max)
    return torch.stack(counts), torch.stack(route_prob), torch.stack(route_prob_max)


def clear_memory(timesformer: TimesformerForVideoClassification, layers_for_switch: list[int]):
    for i, l in enumerate(timesformer.timesformer.encoder.layer):
        if i in layers_for_switch:
            del l.intermediate.counts
            del l.intermediate.prob_sum
            del l.intermediate.prob_max
