from  model_classes import VectorizedMoE
import torch.nn as nn
import math
import copy


class MH_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size # size of token
        self.num_heads = config.num_MH_MOE_heads
        self.sub_seq_len = config.seq_len * config.num_MH_MOE_heads
        self.head_dim = int(config.hidden_size / config.num_MH_MOE_heads) # size of subtoken

        # Initialization and names as in the Multi-Head Mixture-of-Experts paper:
        self.multi_head_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.merge_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.multi_head_layer.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.merge_layer.weight)
        nn.init.constant_(self.merge_layer.bias, 0.0)

        # Create a new config for nested moe:
        nested_moe_config = copy.deepcopy(config)
        nested_moe_config.hidden_size = int(self.head_dim)
        nested_moe_config.seq_len = int(self.sub_seq_len)

        self.nested_moe = VectorizedMoE(nested_moe_config)

    def forward(self, x):
        """Computes just MH-MoE pass (without skip connection)"""
        config = self.config
        x = self.multi_head_layer(x) # first projection - it "merges" attention heads outputs
        x = x.reshape(config.batch_size, self.sub_seq_len, self.head_dim).contiguous()
        x = self.nested_moe(x)
        x = x.reshape(config.batch_size, config.seq_len, config.num_MH_MOE_heads,
                      self.head_dim).reshape(config.batch_size, config.seq_len, config.hidden_size).contiguous()
        x = self.merge_layer(x) # second projection - it "merges" MoE heads outputs
        return x