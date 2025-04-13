import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list


class SwitchFeedForward(Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *, is_scale_prob: bool, n_experts: int, expert: FeedForward, d_model: int):
        """
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        """
        super().__init__()

        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts

        # make copies of the FFNs
        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

        self.counts = None
        self.prob_sum = None
        self.prob_max = None

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        # x = x.swapaxes(0, 1)
        batch_size, seq_len, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[batch_size, seq_len, d_model]`
        final_output = final_output.view(batch_size, seq_len, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        self.counts = counts
        self.prob_sum = route_prob.sum(0)
        self.prob_max = route_prob_max
        return final_output
