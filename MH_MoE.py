from  model_classes import *
import torch
from transformers import PretrainedConfig
import torch.nn as nn
import math
import copy


class MH_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_MH_MOE_heads
        self.no_of_MOE_input_tokens = config.seq_len * config.num_MH_MOE_heads
        self.head_dim = int(config.hidden_size / config.num_MH_MOE_heads)

        self.multi_head_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.merge_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Initialization
        nn.init.xavier_uniform_(self.multi_head_layer.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.merge_layer.weight)
        nn.init.constant_(self.merge_layer.bias, 0.0)

        #creating a new config for nested moe
        nested_moe_config = copy.deepcopy(config)
        nested_moe_config.hidden_size = int(self.head_dim)
        nested_moe_config.seq_len = int(self.no_of_MOE_input_tokens)

        self.nested_moe = VectorizedMoE(nested_moe_config)

    def forward(self, x):
        config = self.config
        x = self.multi_head_layer(x)
        x = x.reshape(config.batch_size, self.no_of_MOE_input_tokens, self.head_dim).contiguous()
        x = self.nested_moe(x)
        x = x.reshape(config.batch_size, config.seq_len, config.num_MH_MOE_heads,
                       self.head_dim).reshape(config.batch_size, config.seq_len, config.hidden_size).contiguous()
        x = self.merge_layer(x)
        return x
    
    def test_if_without_moe_it_works_well(self, x): #ta funkcja jest tu tylko po to, że sprawdziłem czy te reshapy działają dobrze
        config = self.config
        input = x

        # x = self.multi_head_layer(x)
        x = x.reshape(config.batch_size, self.no_of_MOE_input_tokens, self.head_dim).contiguous()
        # x = self.nested_moe(x)
        x = x.reshape(config.batch_size, config.seq_len, config.num_MH_MOE_heads,
                       self.head_dim).reshape(config.batch_size, config.seq_len, config.hidden_size).contiguous()
        # x = self.merge_layer(x)

        print('It should return the same tensor, does it?', torch.equal(input, x), torch.max(abs((x - input).detach())))
        