VIT_BASE_CONFIG = {
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 768 * 4,
    "num_hidden_layers": 12,
}

VIT_SMALL_CONFIG = {
    "hidden_size": 384,
    "num_attention_heads": 6,
    "intermediate_size": 384 * 4,
    "num_hidden_layers": 12,
}

VIT_TINY_CONFIG = {
    "hidden_size": 192,
    "num_attention_heads": 3,
    "intermediate_size": 192 * 4,
    "num_hidden_layers": 12,
}


def get_vit_config(vit_size):
    if vit_size not in ["base", "small", "tiny"]:
        raise ValueError("Invalid Vit size")
    if vit_size == "small":
        return VIT_SMALL_CONFIG
    elif vit_size == "tiny":
        return VIT_TINY_CONFIG
    elif vit_size == "base":
        return VIT_BASE_CONFIG
