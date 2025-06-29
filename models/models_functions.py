from transformers import TimesformerForVideoClassification, TimesformerConfig, VivitForVideoClassification, VivitConfig
from weight_inits import init_timesformer
from models.switch.utils import timesformer_to_switch
import copy


def check_model_name(model):
    if model not in ["timesformer", "vivit", "mvitv2"]:
        raise ValueError(f"Unknown model: {model}")


def get_model(model: str, num_labels, vit_config, config):
    run_config = config
    if config["init_kinetics"] and (config["vit_base_size"] != "base" or config["cut_layers"]):
        raise ValueError(
            "You cannot use `init_kinetics` with `cut_layers` or `vit_base_size` set not to 'base'. "
        )
    if config.get("cut_layers"):
        vit_config = copy.deepcopy(vit_config)
        vit_config["num_hidden_layers"] = vit_config["num_hidden_layers"] - config["cut_layers"]
    check_model_name(model)
    if model == "timesformer":
        if config["vit_base_size"] == "base" and config["init_kinetics"]:
            config = TimesformerConfig(num_frames=config["frames"], **vit_config, num_labels=num_labels)
            model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400", num_labels=run_config["num_classes"], ignore_mismatched_sizes=True)
        else:
            config = TimesformerConfig(num_frames=config["frames"], **vit_config, num_labels=num_labels)
            model = TimesformerForVideoClassification(config)
    return model, config


def init_model(model_name: str, model, vit, n_layers, config):
    check_model_name(model_name)
    if model_name == "timesformer":
        model = init_timesformer(model, vit, n_layers, config["switch"], config["switch_layers"])
    return model


def model_to_switch(model_name: str, model, config, model_config, keep_weights=False):
    check_model_name(model_name)
    if model_name == "timesformer":
        model = timesformer_to_switch(
            model, config["switch_layers"], config["n_experts"], model_config, config["switch_top_k"], keep_weights=keep_weights
        )
    return model
