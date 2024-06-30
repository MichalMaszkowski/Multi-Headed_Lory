from model_classes import *
import torch
from transformers import PretrainedConfig
import torch.nn as nn
import math
import copy

# input shape: [batch size, no segments, num_heads, head_dim]
# for every head, the router schould return weights for each expert, so:
# output shape: [bs, no seq, num heads, num experts]
class Router_mh_lori(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size / config.num_MH_MOE_heads)
        self.num_experts = config.num_experts
        self.device = config.device
        self.expert_embeddings = nn.Parameter(torch.randn(self.hidden_size, self.num_experts))
        torch.nn.init.kaiming_uniform_(self.expert_embeddings, nonlinearity='linear')

    def forward(self, x):
        dot = torch.einsum("bshd,de->bshe", x, self.expert_embeddings)
        res = torch.nn.functional.softmax(dot, dim=-1)
        return res
    
class MH_Lori(nn.Module):
    def __init__(self, config):
        super(MH_Lori, self).__init__()
        self.config = config

        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_size
        self.seq_len = config.seq_len
        self.num_heads = config.num_MH_MOE_heads
        self.head_dim = int(config.hidden_size / config.num_MH_MOE_heads)
        self.no_segments = config.no_lori_segments
        self.segment_len = int(self.seq_len / self.no_segments)
        self.device = config.device
        self.treat_mh_lori_as_regular_lori = config.treat_mh_lori_as_regular_lori
        

        self.router = Router_mh_lori(config).to(self.device)

        if self.treat_mh_lori_as_regular_lori == False:
            self.multi_head_layer = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
            self.merge_layer = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
            # Initialization
            nn.init.xavier_uniform_(self.multi_head_layer.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.merge_layer.weight)
            nn.init.constant_(self.merge_layer.bias, 0.0)

        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size

        self.first_linear = nn.Parameter(torch.randn((self.num_experts, self.intermediate_size, self.head_dim)))
        torch.nn.init.kaiming_uniform_(self.first_linear, nonlinearity='linear')
        self.second_linear = nn.Parameter(torch.randn((self.num_experts, self.head_dim, self.intermediate_size)))
        torch.nn.init.kaiming_uniform_(self.second_linear, nonlinearity='linear')

        self.to(self.device)


    def forward(self, x):
        #x.shape = [batch size, seq len, hidden dim]
        if self.treat_mh_lori_as_regular_lori == False:
            x = self.multi_head_layer(x) 
        #x.shape = [batch size, seq len, hidden dim]
        x = x.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).contiguous()
        #Dividing into lori segments
        x = x.reshape(self.batch_size, self.no_segments, self.segment_len, self.num_heads, self.head_dim).contiguous()
        average_segment_embedding = torch.mean(x, dim = 2).to(self.device)
        # print(f'avarage segment embeding shape = {average_segment_embedding.shape} [batch size, no segments, num_heads, head_dim] (to jest input routera)') #do tego miejsca dziala tak samo jak miriam
        # average_segment_embedding.size = [batch size, no segments, num_heads, head_dim]
        expert_weights = self.router(average_segment_embedding).to(self.device) #to jest inne niz analogiczny moment u  miriam #routery robią to samo, ale tylko kiedy bs 
        #jest większy od 1. U miriam do routera zawsze wchodzi tensor o bs 1? Jednak routery robią to samo nawet dla bs 1. Ale expert weights są różne?????
        # expert_weights shape = [bs, no seq, num heads, num experts]
        expert_weights_log = expert_weights

        #Tu jest dobrze

        # print(f'expert_weights shape = {expert_weights.shape} [bs, no seq, num heads, num experts]')
        # print(expert_weights)
        # calculating merged experts
        expert_weights = torch.transpose(expert_weights, 0, 3) # [num experts, no seq, num heads, bs]
        expert_weights = expert_weights.reshape(self.num_experts, 1, 1, self.no_segments, self.num_heads, self.batch_size) 

        # tu jest tak samo

        merged_experts_1 = self.first_linear.reshape(self.num_experts, self.intermediate_size, self.head_dim, 1, 1, 1)
        merged_experts_1 = (merged_experts_1 * expert_weights).sum(dim = 0) #[intermidiate, head_dim, no_seg, num heads, bs]
        # print('merged expert shape: [intermidiate, head_dim, no_seg, num heads, bs] ', merged_experts_1.shape)
        merged_experts_1 = torch.permute(merged_experts_1, (4, 2, 3, 0, 1))
        merged_experts_1_log = merged_experts_1
        # print('merged expert shape: [self.batch_size, self.no_segments, self.num_heads, self.intermediate_size, self.head_dim] ', merged_experts_1.shape)

        #Do tego miejsca jest dobrze

        # merged_experts_1 = merged_experts_1.reshape(self.batch_size, self.no_segments, self.num_heads, self.intermediate_size, self.head_dim)
        # print(f'merged expert 1 parameter count: {torch.numel(merged_experts_1):,}')
        merged_experts_1 = merged_experts_1[:, :-1, :, :, :] #we discard the last segment as expert created for it is never used


        merged_experts_2 = self.second_linear.reshape(self.num_experts, self.head_dim, self.intermediate_size, 1, 1, 1)
        merged_experts_2 = (merged_experts_2 * expert_weights).sum(dim = 0) #[head dim, intermidiete, n seg, num heads, bs]
        merged_experts_2 = torch.permute(merged_experts_2, (4, 2, 3, 0, 1))
        merged_experts_2_log = merged_experts_2
        # merged_experts_2 = merged_experts_2.reshape(self.batch_size, self.no_segments, self.num_heads, self.head_dim, self.intermediate_size)
        merged_experts_2 = merged_experts_2[:, :-1, :, :, :]
        
        # process x by experts
        x = x.reshape(self.batch_size, self.no_segments, self.num_heads, self.segment_len, self.head_dim).contiguous()
        x_causal = x[:, 1:, :, :, :]
        # process segments s>1 throuth which gradient flows
        result = torch.einsum("bnhld,bnhid->bnhli", x_causal, merged_experts_1)
        result = nn.functional.relu(result, inplace=False)
        result = torch.einsum("bnhli,bnhdi->bnhld", result, merged_experts_2)
        # process segment s=1 without gradient

        with torch.no_grad():
            segment_1 = x[:, 0, :, :, :]
            expert_segment_1 = merged_experts_1[:, 0, :, :, :]
            expert_segment_2 = merged_experts_2[:, 0, :, :, :]

            result_segment_1 = torch.einsum("bhld,bhid->bhli", segment_1, expert_segment_1)
            result_segment_1 = nn.functional.relu(result_segment_1, inplace=False)
            result_segment_1 = torch.einsum("bhli,bhdi->bhld", result_segment_1, expert_segment_2)

            result_segment_1 = result_segment_1.unsqueeze(1)

        # concatenate processed segments
        result = torch.cat((result_segment_1, result), dim = 1)

        # reshape back into orginal shape. Now, result.shape = [(self.batch_size, self.no_segments, self.num_heads, self.segment_len, self.head_dim)]
        # result = torch.transpose(result, 2, 3) We belive that this line schould be there but we are not shure yet
        result = result.reshape(self.batch_size, self.no_segments * self.segment_len, self.num_heads, self.head_dim)
        result = result.reshape(self.batch_size, self.no_segments * self.segment_len, self.hidden_dim)

        if self.treat_mh_lori_as_regular_lori == False:
            result = self.merge_layer(result)

        return result  #, average_segment_embedding, expert_weights, merged_experts_1_log, merged_experts_2_log

    
    def test_if_reshaping_works(self, x):
        input = x
        x = x.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim).contiguous()
        x = x.reshape(self.batch_size, self.no_segments, self.segment_len, self.num_heads, self.head_dim).contiguous()
        x = x.reshape(self.batch_size, self.no_segments, self.num_heads, self.segment_len, self.head_dim).contiguous()
        result = x
        result = result.reshape(self.batch_size, self.no_segments * self.segment_len, self.num_heads, self.head_dim)
        result = result.reshape(self.batch_size, self.no_segments * self.segment_len, self.hidden_dim)
        print(torch.equal(result, input))

