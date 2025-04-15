from transformers import TimesformerForVideoClassification, TimesformerConfig, VivitForVideoClassification, VivitConfig
from weight_inits import init_timesformer, init_vivit
from models.switch.utils import timesformer_to_switch, vivit_to_switch


def check_model_name(model):
    if model not in ["timesformer", "vivit", "mvitv2"]:
        raise ValueError(f"Unknown model: {model}")


def get_model(model: str, num_labels, vit_config, config):
    check_model_name(model)
    if model == "timesformer":
        config = TimesformerConfig(num_frames=config["frames"], **vit_config, num_labels=num_labels)
        model = TimesformerForVideoClassification(config)
    if model == "vivit":
        config = VivitConfig(num_frames=config["num_frames"], **vit_config, num_labels=num_labels)
        model = VivitForVideoClassification(config)
    return model, config


def init_model(model_name: str, model, vit, n_layers, config):
    check_model_name(model_name)
    if model_name == "timesformer":
        model = init_timesformer(model, vit, n_layers, config["switch"], config["switch_layers"])
    if model_name == "vivit":
        model = init_vivit(model, vit, n_layers, config["switch"], config["switch_layers"])
    return model


def model_to_switch(model_name: str, model, config, model_config):
    check_model_name(model_name)
    if model_name == "timesformer":
        model = timesformer_to_switch(model, config["switch_layers"], config["n_experts"], model_config)
    if model_name == "vivit":
        model = vivit_to_switch(model, config["switch_layers"], config["n_experts"], model_config)
