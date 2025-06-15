import torch


def init_timesformer(timesformer, vit, n_layers, switch, switch_layers):
    with torch.no_grad():
        timesformer.timesformer.embeddings.patch_embeddings.projection.load_state_dict(
            vit.vit.embeddings.patch_embeddings.projection.state_dict()
        )
        timesformer.timesformer.embeddings.position_embeddings.copy_(vit.vit.embeddings.position_embeddings)
        #torch.nn.init.uniform(timesformer.timesformer.embeddings.time_embeddings)
        for i in range(n_layers):
            vit_qkv_weight = torch.cat(
                [
                    vit.vit.encoder.layer[i].attention.attention.query.weight,
                    vit.vit.encoder.layer[i].attention.attention.key.weight,
                    vit.vit.encoder.layer[i].attention.attention.value.weight,
                ]
            )
            vit_qkv_bias = torch.cat(
                [
                    vit.vit.encoder.layer[i].attention.attention.query.bias,
                    vit.vit.encoder.layer[i].attention.attention.key.bias,
                    vit.vit.encoder.layer[i].attention.attention.value.bias,
                ]
            )

            timesformer.timesformer.encoder.layer[i].attention.attention.qkv.weight.copy_(vit_qkv_weight)
            timesformer.timesformer.encoder.layer[i].attention.attention.qkv.bias.copy_(vit_qkv_bias)
            timesformer.timesformer.encoder.layer[i].attention.output.dense.load_state_dict(
                vit.vit.encoder.layer[i].attention.output.dense.state_dict()
            )

            if switch and i in switch_layers:
                for expert in timesformer.timesformer.encoder.layer[i].intermediate.experts:
                    expert.layer1.load_state_dict(vit.vit.encoder.layer[i].intermediate.dense.state_dict())
                    expert.layer2.load_state_dict(vit.vit.encoder.layer[i].output.dense.state_dict())
            else:
                timesformer.timesformer.encoder.layer[i].intermediate.dense.load_state_dict(
                    vit.vit.encoder.layer[i].intermediate.dense.state_dict()
                )
                timesformer.timesformer.encoder.layer[i].output.dense.load_state_dict(
                    vit.vit.encoder.layer[i].output.dense.state_dict()
                )

            timesformer.timesformer.encoder.layer[i].layernorm_before.load_state_dict(
                vit.vit.encoder.layer[i].layernorm_before.state_dict()
            )
            timesformer.timesformer.encoder.layer[i].layernorm_after.load_state_dict(
                vit.vit.encoder.layer[i].layernorm_after.state_dict()
            )
            timesformer.timesformer.encoder.layer[i].temporal_layernorm.load_state_dict(
                vit.vit.encoder.layer[i].layernorm_before.state_dict()
            )
            timesformer.timesformer.encoder.layer[i].temporal_attention.attention.qkv.weight.copy_(vit_qkv_weight)
            timesformer.timesformer.encoder.layer[i].temporal_attention.attention.qkv.bias.copy_(vit_qkv_bias)
            timesformer.timesformer.encoder.layer[i].temporal_attention.output.dense.load_state_dict(
                vit.vit.encoder.layer[i].attention.output.dense.state_dict()
            )
            # timesformer.timesformer.encoder.layer[i].temporal_dense.load_state_dict(
            #     vit.vit.encoder.layer[i].attention.output.dense.state_dict()
            # )
        timesformer.timesformer.layernorm.load_state_dict(vit.vit.layernorm.state_dict())
    return timesformer
