import torch
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import LoRAAttnProcessor

def reference_head_to_batch_dim(tensor, num_heads, out_dim=3):
    # reference implementation from huggingface. function is called head_to_batch_dim there
    head_size = num_heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)

    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

    return tensor

def move_attn_head_to_front(attn_tensor, num_heads, out_dim=3):
    # given attn_tensor of shape batch x num_tokens x d_attn, split d_attn into num_heads x d_head
    # and move num_heads to front
    # if out_dim is 3, merge batch and num_heads together -- makes it easier to do torch.bmm bc the "batch" is functionally the batch x num_heads and then you can just do matmul using n_tokens x d_head
    assert attn_tensor.ndim == 3, "attn tensor should be 3d: batch x num_tokens x d_attn"
    batch, num_tokens, d_attn = attn_tensor.shape
    assert d_attn % num_heads == 0, "d_attn should be divisible by num_heads"
    d_head = d_attn // num_heads
    attn_tensor = attn_tensor.view(batch, num_tokens, num_heads, d_head)
    attn_tensor = attn_tensor.transpose(1, 2)
    if out_dim == 3:
        attn_tensor = attn_tensor.reshape(batch * num_heads, num_tokens, d_head)
    return attn_tensor

# write some test asserts to show these are the same, out_dim should be 4 for the default since we're not merging batch and head together
# asserts- start by makign a dummy tensor
# batch = 4
# num_tokens = 5
# d_attn = 6
# num_heads = 2
# dummy_tensor = torch.randn(batch, num_tokens, d_attn)
# assert torch.allclose(reference_head_to_batch_dim(dummy_tensor, num_heads, out_dim=4), move_attn_head_to_front(dummy_tensor, num_heads, out_dim=4))
# assert torch.allclose(reference_head_to_batch_dim(dummy_tensor, num_heads, out_dim=3), move_attn_head_to_front(dummy_tensor, num_heads, out_dim=3))

class MyLoRALinearLayer(nn.Module):
    def __init__(self, input_size, output_size, lora_inner_rank):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lora_inner_rank = lora_inner_rank
        # two layers, from hidden to lora inner and then back
        # self.in = nn.
        self.in_layer = nn.Linear(input_size, lora_inner_rank, bias=False)
        self.out_layer = nn.Linear(lora_inner_rank, output_size, bias=False)
        #      We use a random Gaussian initialization for A and
        # zero for B, so delta_W = BA is zero at the beginning of training
        nn.init.normal_(self.in_layer.weight, mean=0.0, std=1.0 / lora_inner_rank)
        nn.init.zeros_(self.out_layer.weight)

    def forward(self, x):
        return self.out_layer(self.in_layer(x))


class MyLoRAAttnProcessor(nn.Module):
    """my personally coded lora attn processor,

    closely following https://github.com/huggingface/diffusers/blob/74d902eb59f873b6156621220937f8e2521dfdc0/src/diffusers/models/attention_processor.py#L509

    but coded by hand so I can learn more stuff"""

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()
        self.hidden_size = hidden_size  # d_attn, dimension of vector that a q/k/v gets projected to during attention.
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        # hidden states corresponds to that which generates q. in case of pixel it's an entry from the latent representation.
        # encoder_hidden_states is from cross attention. None if it's self attention and we just use hidden states for both
        # shapes: hidden is
        # hint:  in original code the thing looks like this, as a hint for dimensions
        # self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        # self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        # self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        # self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)
        self.to_q_lora = MyLoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = MyLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = MyLoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = MyLoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        # hidden states: shape (batch_size, channel, height, width) # each entry is like a pixel from latent space
        # encoder_hidden_states: shape (batch_size, num_tokens, d_token) # each entry is like a token embedded from text prompt, if it exists
        residual = hidden_states
        assert attn.spatial_norm is None, "spatial norm not supported yet"
        assert hidden_states.ndim == 4, "hidden states should be 4d: batch x channel x height x width"
        # reshape + transpose hidden states b c h w -> b (hw) c
        input_ndim = hidden_states.ndim
        batch, channel, height, width = hidden_states.shape
        d_query = channel
        assert d_query == self.hidden_size, "seems to be assumed by huggingface code: https://github.com/huggingface/diffusers/pull/4287#issuecomment-1687276034"

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        text_batch, num_tokens, d_token = encoder_hidden_states.shape
        assert batch == text_batch, "batch size of hidden states and encoder hidden states should be the same"

        attention_mask = attn.prepare_attention_mask(attention_mask, height * width, batch) # height * width = num hidden states

        if attn.group_norm is not None:
            print(f"attn group norm is not None")
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        print(f"hidden states shape is {hidden_states.shape} with original shape {batch, channel, height, width}")
        print(f" attn.to_q is {attn.to_q}")
        print(f"attn.to_q(hidden_states) shape is {attn.to_q(hidden_states).shape} while self.to_q_lora(hidden_states) shape is {self.to_q_lora(hidden_states).shape}")
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = move_attn_head_to_front(query, attn.heads)
        # query = attn.head_to_batch_dim(query)
        print(f"attn has num heads {attn.heads}, query shape is {query.shape}")

        # now figure out if we're doing self attention or cross attention with encoder hidden states
        if encoder_hidden_states is None:
            # self attention
            key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states)
            value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states)
        else:
            if attn.norm_cross:
                # there are multiple possible norms that the encoder could use; here we just pick it with attn.norm_encoder_hidden_states
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            # cross attention
            key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        key = move_attn_head_to_front(key, attn.heads)
        value = move_attn_head_to_front(value, attn.heads)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)
        # all tensors q, k, v are now of shape (batch x num_heads) x num_tokens x d_head

        # todo: what is qkt doing
        qkT = torch.bmm(query, key.transpose(1, 2)) # Q K^T where T is transpose
        # qkT is now of shape (batch x num_heads) x num_tokens x num_tokens
        # skip attn_mask which was originally there, whatever

        # not quite sure what this is but copying from huggingface for now??
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # now hidden_states is of shape (batch x num_heads) x num_tokens x d_head
        # do the feedforward part back out to hidden_size
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout - this part i just copy pasted because, fair enough
        hidden_states = attn.to_out[1](hidden_states)

        # finally finish up by reshaping back to original shape
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

# now test against from diffusers.models.attention_processor import LoRAAttnProcessor
d_query = 4 # d_pixel in latent
cross_attention_dim = 4 # d_token
d_attn = 4
n_heads = 1
d_head = d_attn // n_heads # must be divisible

rank = 4
my_lora_attn_processor = MyLoRAAttnProcessor(d_attn, cross_attention_dim, rank)
lora_attn_processor = LoRAAttnProcessor(d_attn, cross_attention_dim, rank)
# now test using some imagined inputs with a reasonable shape
batch = 2
channel = d_query # d_pixel ie num channels in latent image
height = 2
width = 2
num_tokens = 3
hidden_states = torch.randn(batch, channel, height, width)
hidden_states_clone = hidden_states.clone()
encoder_hidden_states = torch.randn(batch, num_tokens, cross_attention_dim)
encoder_hidden_states_clone = encoder_hidden_states.clone()

# now run both

# Attention init some params:
# self,
# query_dim: int,
# cross_attention_dim: Optional[int] = None,
# heads: int = 8,
# dim_head: int = 64,
# dropout: float = 0.0,
# bias = False,
# upcast_attention: bool = False,
# upcast_softmax: bool = False,
# cross_attention_norm: Optional[str] = None,
attn = Attention(d_query, cross_attention_dim=cross_attention_dim, heads=n_heads, dim_head=d_head, dropout=0.0)
attn.requires_grad_(False)

# hardcode the weights a little bit to test things
torch.manual_seed(42)
for layer_tuple in ((lora_attn_processor.to_q_lora, my_lora_attn_processor.to_q_lora),
              (lora_attn_processor.to_k_lora, my_lora_attn_processor.to_k_lora),
              (lora_attn_processor.to_v_lora, my_lora_attn_processor.to_v_lora),):
    original_lora, my_lora = layer_tuple
    # arbitrary random normal init
    original_lora.down.weight = nn.Parameter(torch.randn(original_lora.down.weight.shape))
    my_lora.in_layer.weight = nn.Parameter(original_lora.down.weight)
    original_lora.up.weight = nn.Parameter(torch.randn(original_lora.up.weight.shape))
    my_lora.out_layer.weight = nn.Parameter(original_lora.up.weight)


result = lora_attn_processor(attn=attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
my_result = my_lora_attn_processor(attn=attn, hidden_states=hidden_states_clone, encoder_hidden_states=encoder_hidden_states_clone)

# now check that they're the same
assert torch.allclose(my_result, result)
