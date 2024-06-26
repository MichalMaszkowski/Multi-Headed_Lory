import torch
import torch.nn as nn
from typing import List, Tuple, Any, Optional
import math
from pytorch_lightning import LightningModule


AttentionT = torch.tensor  # torch tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
HiddenT = torch.tensor
TokensT = torch.tensor # [BATCH, SEQ_LEN]
ModelLT = torch.tensor # [BATCH, SEQ_LEN, VOCAB_SIZE]

class AttentionCreateQKV(torch.nn.Module):
    """
    Given a tensor of shape [BATCH, SEQ_LEN, HIDDEN_DIM]
    uses linear projections to create three tensors
    Query, Key and Value.
    Each of the created tensors has shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM].
    Where HEAD_DIM = HIDDEN_DIM // NUM_HEADS
    """

    def __init__(self, hidden_dim, num_heads) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        self.key_transform = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = False)
        self.query_transform = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = False)
        self.value_transform = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = False)

    def forward(self, x: HiddenT) -> Tuple[AttentionT, AttentionT, AttentionT]:
        assert len(x.shape) == 3  # torch tensor of shape [BATCH, SEQ_LEN, HIDDEN_DIM]

        result = []
        shape = x.shape

        Q = self.query_transform(x)
        result.append(torch.reshape(Q, (shape[0], shape[1], self.num_heads, self.head_dim)))

        K = self.key_transform(x)
        result.append(torch.reshape(K, (shape[0], shape[1], self.num_heads, self.head_dim)))

        V = self.value_transform(x)
        result.append(torch.reshape(V, (shape[0], shape[1], self.num_heads, self.head_dim)))

        assert len(result) == 3  # queries, keys, values
        for r in result:
            assert len(r.shape) == 4  # [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
            assert r.shape[-2:] == (self.num_heads, self.head_dim)
            assert r.shape[:-2] == x.shape[:2]

        return result
    
class RoPEPosEncoding(torch.nn.Module):
    """
    Given a tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
    applies Rotary Positional Encoding.
    offset allows to apply rotary to sequnce part by part by telling how much tokens preecede the input in the sequence.
    """

    def __init__(self, head_dim, number) -> None:
        super().__init__()

        assert head_dim % 2 == 0
        self.hidden_dim = head_dim
        self.number = number
        self.theta = (1. / (self.number ** (torch.arange(0, head_dim, 2).float() / head_dim))) #now of length head_dim//2  #with .repeat_interleave(2) would double


    def forward(self, x: AttentionT, offset: int = 0):
        assert (
            len(x.shape) == 4
        )  # torch tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
        assert offset >= 0

        (batch, seq_len, num_heads, head_dim) = x.shape
        pos_idx = offset + torch.arange(seq_len).float().to(x.device) #position index (j)
        self.theta = self.theta.to(x.device)
        angle_ji = torch.outer(pos_idx, self.theta) #outer product of position index j and θi #shape==(seq_len, head_dim//2)
        angle_ji = torch.reshape(input = angle_ji, shape = (1, seq_len, 1, head_dim//2))

        #both of shape (1, seq_len, 1, head_dim//2)
        cos = torch.cos(angle_ji).requires_grad_(requires_grad=False)
        sin = torch.sin(angle_ji)

        x_paired = torch.reshape(input = x, shape = (batch, seq_len, num_heads, head_dim // 2, 2))

        result = torch.zeros_like(x_paired)
        result[:, :, :, :, 0] = x_paired[:, :, :, :, 0] * cos - x_paired[:, :, :, :, 1] * sin
        result[:, :, :, :, 1] = x_paired[:, :, :, :, 0] * sin + x_paired[:, :, :, :, 1] * cos

        result = torch.reshape(input = result, shape = x.shape)

        assert result.shape == x.shape

        return result

number = 10000 #used in calculating theta

ACacheT = Tuple[
    torch.tensor, torch.tensor
]  # key, value, both of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]


class Attention(torch.nn.Module):
    """
    Implements multi-head attention layer.
    Inputs tensor x of shape [BATCH, SEQ_LEN, hidden_dim].
    Uses head_proj to create three tensors q, k, v - each of shape
    [BATCH, SEQ_LEN, num_heads, head_dim].
    Then applies RoPE to q and k.
    Then calculates attention within each head, concatenates the results
    and linearly projects them to a tensor of shape [BATCH, SEQ_LEN, hidden_dim].

    Cache is a tuple of keys (kc) and values (vc) calculated in previous calls.
    For training the cache will be empty (tensors kc and vc should have shape [BATCH, 0, num_heads, hidden_dim]),
    For efficient generation, the cache will contain keys (kc), values (vc) of already read/generated tokens
    (this allows the generation of one additional token without recomputing the keys and values for all preceding tokens).
    After RoPE application to k, kc and vc are prepended to k and v respectively.

    The model outputs the linearly projected output of attention along with a cache extended with new keys and values.
    """
    def __init__(
        self, hidden_dim: int, num_heads: int, head_proj=AttentionCreateQKV
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.projector = head_proj(self.hidden_dim, self.num_heads)
        self.encoder = RoPEPosEncoding(self.head_dim, number)
        self.linear = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)

    def get_empty_cache(self, batch_size: int, device) -> ACacheT:
        return torch.empty(
            batch_size, 0, self.num_heads, self.head_dim, device=device
        ), torch.empty(batch_size, 0, self.num_heads, self.head_dim, device=device)

    def forward(self, x: HiddenT) -> Tuple:
        assert len(x.shape) == 3  # torch tensor of shape [BATCH, SEQ_LEN, HIDDEN_DIM]

        (batch, seq_len, _) = x.shape

        qkv = self.projector(x)
        (Q, K, V) = (qkv[0], qkv[1], qkv[2])

        Q = self.encoder(Q) #shape [BATCH, SEQ_LEN, num_heads, head_dim]
        Q = torch.transpose(input = Q, dim0 = 1, dim1 = 2) #shape [BATCH, num_heads, SEQ_LEN, head_dim]
        K = self.encoder(K) #shape [BATCH, SEQ_LEN, num_heads, head_dim]
        K = torch.transpose(input = K, dim0 = 1, dim1 = 2) #shape [BATCH, num_heads, SEQ_LEN, head_dim]

        #A=QK^T
        A = torch.matmul(input = Q, other = torch.transpose(input = K, dim0 = 2, dim1 = 3))
        #shape of A is [BATCH, num_heads, SEQ_LEN, SEQ_LEN]
        A = torch.div(input = A, other = torch.sqrt(torch.tensor([self.head_dim], device = x.device)))

        #mask = torch.tensor([float('-inf')]) * torch.triu(torch.ones_like(A), diagonal = 1)
        mask = torch.triu(torch.ones_like(A), diagonal = 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        A = A + mask
        A = torch.nn.functional.softmax(input = A, dim = 3)

        O = torch.einsum("bhij,bjhd->bihd", A, V)
        O = torch.reshape(input = O, shape = x.shape)

        attention = self.linear(O) #as asked here: "The output:    concatenate outputs of the heads and project them linearly to have the hidden_dim dimension"

        ############# Uwaga, tu jest ta warstwa liniowa i ja nie jestem pewien czy ona powinna tu być? Czemu ona służy?

        attention_weights = A

        return attention, attention_weights

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_dim, eps=1e-05) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones((hidden_dim)))
        self.beta = nn.Parameter(torch.zeros((hidden_dim)))

    def forward(self, x: HiddenT) -> HiddenT:
        assert len(x.shape) == 3  # torch tensor of shape [BATCH, SEQ_LEN, HIDDEN_DIM]

        mean = torch.mean(input = x, dim = -1, keepdim = True)
        mean_x2 = torch.mean(input = (x ** 2), dim = -1, keepdim = True)
        var = mean_x2 - (mean ** 2)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        result = x_norm * self.gamma + self.beta

        assert x.shape == result.shape
        return result
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, config) -> None:
        """
        forward_layer_class - an nn module for MoE or Lori or MH moe or MH Lori
        config - 
        num_heads - num attention heads
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.layer_norm1 = LayerNorm(self.hidden_dim)
        self.layer_norm2 = LayerNorm(self.hidden_dim)
        self.forward_layer = config.forward_layer_class(config) #an nn module
        self.attention = Attention(self.hidden_dim, self.num_heads)

    def forward(self, x: HiddenT) -> HiddenT:
        #sub_with_skip(x) = x + sublayer(norm(x))
        a = self.attention(self.layer_norm1(x))
        result = x + a[0]
        moe_result, load_balancing_loss = self.forward_layer(self.layer_norm2(result))
        result = result + moe_result

        assert x.shape == result.shape

        return result, load_balancing_loss
    
# TokensT = torch.tensor # [BATCH, SEQ_LEN]
# ModelLT = torch.tensor # [BATCH, SEQ_LEN, VOCAB_SIZE]
#implemented a training step using pytorch lightning

class Transformer(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.hidden_dim = config.hidden_size
        self.forward_layer = config.forward_layer_class
        self.num_heads = config.num_attention_heads

        self.loss_fn = config.loss_fn
        self.load_balancing_coefficient = config.load_balancing_coefficient
        self.py_lightning_loging = config.py_lightning_loging
        self.lr = config.lr
        self.betas = config.betas

        #for loging purposes
        self.no_of_skiped_baches = 0
        self.no_of_total_batches = 0
        self.val_losses_list=[]
        self.train_losses_list=[]

        self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_dim)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(config = config) for _ in range(self.n_layers)
        ])

        self.final_proj = torch.nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x: TokensT) -> ModelLT:
        assert len(x.shape) == 2 # [BATCH, SEQ_LEN]

        x = self.embedding(x)

        auxiliary_loss = 0

        for l in self.layers:
            x, load_balancing_loss = l(x)
            auxiliary_loss += load_balancing_loss

        x = self.final_proj(x)
        return x, auxiliary_loss
    
    def training_step(self, batch, batch_idx):
        self.train()
        inputs = batch[:, :-1]
        self.no_of_total_batches += 1
        if inputs.shape != (self.config.batch_size, self.config.seq_len):
            # print('ERROR: Input has inproper shape')
            self.no_of_skiped_baches += 1
            return None #we skip this training step
        labels = batch[:, 1: ]
        outputs, auxiliary_loss = self.forward(inputs)
        outputs = torch.transpose(outputs, 1, 2)
        loss = self.loss_fn(outputs, labels) 
        
        # print(f'   TRRAINING: Batch {batch_idx}, loss {loss}')
        log_entry = {
            'batch_idx': batch_idx,
            'train_loss': loss.item(),  # Assuming loss is a tensor; convert to float if necessary
            # Add more logs as needed
        }
        self.train_losses_list.append(log_entry)
        if self.py_lightning_loging == True:
            self.py_lightning_loging('Training Loss', loss, on_step=True, on_epoch=True)
        
        loss += (self.load_balancing_coefficient * auxiliary_loss) # for experts' load balancing
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        inputs = batch[:, :-1]
        self.no_of_total_batches += 1
        if inputs.shape != (self.config.batch_size, self.config.seq_len):
            print('ERROR: Input has inproper shape')
            self.no_of_skiped_baches += 1
            return None #we skip this training step
        labels = batch[:, 1: ]
        with torch.no_grad():
            outputs, _ = self.forward(inputs) # auxiliary_loss is not necessary here
        outputs = torch.transpose(outputs, 1, 2)
        loss = self.loss_fn(outputs, labels) 
        
        # print(f'   VALIDATION: Batch {batch_idx}, loss {loss}')
        # Log entry for this validation step
        log_entry = {
            'batch_idx': batch_idx,
            'val_loss': loss.item(),  # Assuming loss is a tensor; convert to float if necessary
            # Add more logs as needed
        }
        self.val_losses_list.append(log_entry)
        if self.py_lightning_loging == True:
            self.py_lightning_loging('Validation Loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.lr, betas = self.betas)

# Input: [batch_size, seq_len, hidden_size] - input embeddings
# Output: [batch_size, seq_len, num_experts] - expert routing weights
class Router(nn.Module):
    """Corresponds to gating function in the Multi-Head Mixture-of-Experts paper"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        self.expert_embeddings = nn.Parameter(torch.randn(self.num_experts, self.hidden_size)).to(config.device)
        torch.nn.init.kaiming_uniform_(self.expert_embeddings, nonlinearity='linear')

    def forward(self, x): # as in the Multi-Head Mixture-of-Experts paper
        dot = torch.einsum("bsh,eh->bse", x, self.expert_embeddings)
        soft_dot = torch.nn.functional.softmax(dot, dim=-1)
        return soft_dot

# Input: [batch_size, seq_len, hidden_size] - input embeddings
# Output: [batch_size, seq_len, hidden_size] - output embeddings
class VectorizedMoE(nn.Module):
    """version which takes first not random tokens up to expert_capacity"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.capacity_factor
        self.intermediate_size = config.intermediate_size

        # You can change experts representation if you want
        self.first_linear = nn.Parameter(torch.randn(self.num_experts, self.intermediate_size, self.hidden_size).to(config.device))
        torch.nn.init.kaiming_uniform_(self.first_linear, nonlinearity='linear')
        self.second_linear = nn.Parameter(torch.randn(self.num_experts, self.hidden_size, self.intermediate_size).to(config.device))
        torch.nn.init.kaiming_uniform_(self.second_linear, nonlinearity='linear')

        self.router = Router(config)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        #assert hidden_size == self.hidden_size

        expert_capacity = math.ceil(batch_size * seq_len / self.num_experts * self.capacity_factor)

        # Load balancing loss as in section 3.2 in the Multi-Head Mixture-of-Experts paper:
        load_balancing_loss = self.num_experts / (batch_size * seq_len) # first factor in the equation (10)

        soft_dot = self.router(x) #[batch_size, seq_len, num_experts]
        top_k_weights = torch.topk(soft_dot, k=self.num_experts_per_token)
        weights = torch.zeros_like(soft_dot).scatter_(dim=-1, index=top_k_weights.indices, src=top_k_weights.values)
        
        experts_where_ones = torch.where((weights <= 0), 0, 1) #ceiling of weights
        experts_where_ones = torch.reshape(experts_where_ones, shape=(-1, self.num_experts)) #[num_of_tokens, num_experts]
        expert_frequency = experts_where_ones.sum(dim=0, keepdim=True) / experts_where_ones.shape[0] #[1, num_experts]
        expert_frequency = expert_frequency.unsqueeze(dim=0)
        load_balancing_loss *= (expert_frequency * soft_dot).sum()

        capacity_aware_ones = torch.where((torch.cumsum(experts_where_ones, dim= 0) <= expert_capacity), input = experts_where_ones, other = 0)

        capacity_aware_weights = weights.reshape(shape=(-1, self.num_experts)) * capacity_aware_ones
        which = torch.topk(capacity_aware_weights, k=expert_capacity, dim = 0)
        indices = which.indices.transpose(1,0)
        index = indices.reshape((-1))

        tokens_for_experts = torch.index_select(input=x.reshape((-1, hidden_size)), dim=0, index=index) #[capacity*num_experts, hidden_size]
        tokens_for_experts  = tokens_for_experts.reshape((self.num_experts, expert_capacity, hidden_size))
        #now I have the proper input to the "experts", which I should process by first layer parameters

        intermediate_result = torch.einsum("ech,eih->eci", tokens_for_experts, self.first_linear)
        intermediate_result = torch.nn.functional.relu(intermediate_result)
        result = torch.einsum("eci,ehi->ech", intermediate_result, self.second_linear)
        #now tokens are processed by the "experts", I need to multiply by the weights and add them up

        w = which.values.transpose(1,0).unsqueeze(-1)

        result = result * w

        final_result = torch.zeros_like(x).reshape((-1, hidden_size)).index_add_(dim = 0, index=index, source = result.reshape((-1, hidden_size)))

        return final_result.reshape(x.shape), load_balancing_loss
    
class FeedForward(torch.nn.Module): #służy mi do tego by sprawdzić czy jak się podmieni vectorised moe w configu na cos innego to dziala
    """
    Inputs a tensor of shape [BATCH, SEQ_LEN, HIDDEN_DIM]
    and processes it as follows:
    * project linearly from hidden_dim to inner_dim
    * apply activation function (GELU)
    * project linearly from inner_dim to hidden_dim
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.inner_dim = config.intermediate_size
        self.first_linear = nn.Linear(in_features = self.hidden_dim, out_features = self.inner_dim, bias = True) #in the paper there is a bias term here
        self.activation = nn.GELU()
        self.second_linear = nn.Linear(in_features = self.inner_dim, out_features = self.hidden_dim, bias = True) #in the paper there is a bias term here

    def forward(self, x: HiddenT) -> HiddenT:
        # [BATCH, SEQ_LEN, HIDDEN_DIM]
        assert len(x.shape) == 3

        ### YOUR CODE STARTS ###
        result = self.second_linear(self.activation(self.first_linear(x)))
        ###  YOUR CODE ENDS  ###

        # [BATCH, SEQ_LEN, HIDDEN_DIM]
        assert len(result.shape) == 3
        return result