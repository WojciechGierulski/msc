import torch
from torch import nn
from torch.nn import functional as F

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list


class SwitchFeedForward(Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(
        self,
        *,
        is_scale_prob: bool,
        n_experts: int,
        expert: FeedForward,
        d_model: int,
        top_k: int = 2,
        activate_only_n_expert=None,
        save_token_routes=False
    ):
        """
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer
        * `d_model` is the number of features in a token embedding
        """
        super().__init__()

        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.top_k = top_k

        # make copies of the FFNs
        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.route_frac = None
        self.route_prob = None

        self.activate_only_n_expert = activate_only_n_expert

        self.save_token_routes = save_token_routes
        self.token_routes = None

        self.labels = None
        self.route_tokens_to_labels = False
        self.experts_classes = None

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        if self.top_k == 1:
            batch_size, seq_len, d_model = x.shape
            x = x.view(-1, d_model)

            route_prob = self.softmax(self.switch(x))

            route_prob_max, routes = torch.max(route_prob, dim=-1)
            if self.save_token_routes:
                self.token_routes = routes

            if self.activate_only_n_expert is not None:
                # only for debug when testing
                routes = torch.zeros((routes.shape))
                routes[:] = self.activate_only_n_expert
            if self.route_tokens_to_labels:
                class_to_expert_idx_map = {}
                for expert_idx, classes_for_expert in enumerate(self.experts_classes):
                    for cls_label in classes_for_expert:
                        class_to_expert_idx_map[cls_label] = expert_idx
                routes = torch.empty(batch_size * seq_len, device=x.device)
                for i in range(batch_size):
                    item_class = self.labels[i]
                    assigned_expert_idx = class_to_expert_idx_map.get(item_class.item())
                    start_token_idx = i * seq_len
                    end_token_idx = (i + 1) * seq_len
                    routes[start_token_idx:end_token_idx] = assigned_expert_idx

            indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

            final_output = x.new_zeros(x.shape)

            counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

            # Get outputs of the expert FFNs
            expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

            for i in range(self.n_experts):
                final_output[indexes_list[i], :] = expert_output[i]

            if self.is_scale_prob:
                final_output = final_output * route_prob_max.view(-1, 1)
            else:
                final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

            final_output = final_output.view(batch_size, seq_len, d_model)
            total = counts.sum(dim=-1, keepdims=True)
            route_frac = counts / total
            route_prob = route_prob / total
            route_prob = route_prob.sum(0)
            self.route_frac = route_frac
            self.route_prob = route_prob
            return final_output
        else:
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])  # Flatten to [N, input_dim] where N = batch*seq_len

            # Get gating weights and expert indices
            # gate_weights: [N, num_experts], top_k_indices: [N, k]

            logits = self.switch(x)
            raw_weights = F.softmax(logits, dim=-1)

            # Select top-k experts
            # top_k_logits shape: [batch_size * seq_len, k]
            # top_k_indices shape: [batch_size * seq_len, k]
            top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=-1)

            # Apply softmax to top-k logits for weights
            # top_k_weights shape: [batch_size * seq_len, k]
            # top_k_weights = F.softmax(top_k_logits, dim=-1)

            # Create a sparse weight matrix for combining outputs
            # full_weights shape: [batch_size * seq_len, num_experts]
            gate_weights = torch.zeros_like(logits)
            gate_weights.scatter_(1, top_k_indices, top_k_weights)

            # Initialize final output tensor
            final_output = torch.zeros(x.shape[0], original_shape[-1], device=x.device, dtype=x.dtype)

            # Get indices for batch processing
            # flat_top_k_indices: [N * k]
            flat_top_k_indices = top_k_indices.view(-1)

            # Map tokens to their assigned experts
            # Create a flat tensor of inputs for batching across experts
            # flat_x: [N * k, input_dim]
            flat_x = x.repeat_interleave(self.top_k, dim=0)

            # Dispatch tokens to experts and compute outputs
            expert_outputs = []
            for i in range(self.n_experts):
                # Find indices of tokens assigned to expert i
                # idx: [num_tokens_for_expert_i]
                idx = torch.where(flat_top_k_indices == i)[0]

                if idx.numel() > 0:
                    # Process tokens assigned to this expert
                    expert_input = flat_x[idx]
                    expert_output = self.experts[i](expert_input)

                    # Store output and original indices
                    expert_outputs.append((idx, expert_output))

            # Combine expert outputs using gating weights
            # We need to map the results back to the original token positions
            flat_gate_weights = gate_weights.view(-1, 1)  # [N * num_experts, 1]

            for idx, output in expert_outputs:
                # Find the corresponding weights for these outputs
                # Need original token indices and expert indices
                original_indices = idx // self.top_k  # Get original token index (0 to N-1)
                expert_indices = flat_top_k_indices[idx]  # Which expert (0 to num_experts-1)

                # Gather the weights using original and expert indices
                weights = gate_weights[original_indices, expert_indices].unsqueeze(1)

                # Weight the expert output
                weighted_output = output * weights

                # Add to the final output tensor at the correct positions
                # Use index_add_ for scatter-add operation
                final_output.index_add_(0, original_indices, weighted_output)

            # Reshape back to original shape [batch_size, seq_len, output_dim]
            final_output = final_output.view(original_shape[0], original_shape[1], original_shape[-1])

            # Calculate fraction of tokens routed to each expert (f_i)
            one_hot_indices = F.one_hot(top_k_indices, num_classes=self.n_experts)
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=1)
            f_i = torch.mean(one_hot_indices.float(), dim=0)

            # Calculate average routing probability per expert (P_i)
            # This is the mean of the gate weights for each expert
            mean_prob_per_expert = torch.mean(raw_weights, dim=0)  # Shape [num_experts]
            P_i = mean_prob_per_expert

            self.route_frac = f_i
            self.route_prob = P_i
            return final_output
