import time
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

flash_gat = load(name="flash_gat", sources=["ops/flash_gat_single.cu"],
                 extra_cuda_cflags=['-O2', '--ptxas-options=-v'], verbose=True)

m = torch.jit.load("gat_test.pt")
inputs = [param for name, param in m.named_parameters()]


from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from mypyg.utils import softmax

self_center_embed = SingleInputEmbedding(in_channel=2, out_channel=64)
self_nbr_embed = MultipleInputEmbedding(in_channels=[2, 2], out_channel=64)
self_lin_q = nn.Linear(64, 64)
self_lin_k = nn.Linear(64, 64)
self_lin_v = nn.Linear(64, 64)
self_lin_self = nn.Linear(64, 64)
self_lin_ih = nn.Linear(64, 64)
self_lin_hh = nn.Linear(64, 64)
self_out_proj = nn.Linear(64, 64)
self_norm1 = nn.LayerNorm(64)
self_norm2 = nn.LayerNorm(64)
self_mlp = nn.Sequential(
    nn.Linear(64, 64 * 4),
    nn.ReLU(inplace=True),
    nn.Linear(64 * 4, 64))
torch.set_grad_enabled(False)
torch.set_printoptions(precision=7)
x, out_gt, rotate_mat, \
    self_center_embed.embed[0].weight.data, self_center_embed.embed[0].bias.data, \
    self_center_embed.embed[1].weight.data, self_center_embed.embed[1].bias.data, \
    self_center_embed.embed[3].weight.data, self_center_embed.embed[3].bias.data, \
    self_center_embed.embed[4].weight.data, self_center_embed.embed[4].bias.data, \
    self_center_embed.embed[6].weight.data, self_center_embed.embed[6].bias.data, \
    self_center_embed.embed[7].weight.data, self_center_embed.embed[7].bias.data, \
    bos_mask, self_bos_token, \
    fixed_edge_index, edge_index, \
    self_norm1.weight.data, self_norm1.bias.data, \
    edge_attr, \
    self_nbr_embed.module_list[0][0].weight.data, self_nbr_embed.module_list[0][0].bias.data, \
    self_nbr_embed.module_list[0][1].weight.data, self_nbr_embed.module_list[0][1].bias.data, \
    self_nbr_embed.module_list[0][3].weight.data, self_nbr_embed.module_list[0][3].bias.data, \
    self_nbr_embed.module_list[1][0].weight.data, self_nbr_embed.module_list[1][0].bias.data, \
    self_nbr_embed.module_list[1][1].weight.data, self_nbr_embed.module_list[1][1].bias.data, \
    self_nbr_embed.module_list[1][3].weight.data, self_nbr_embed.module_list[1][3].bias.data, \
    self_nbr_embed.aggr_embed[0].weight.data, self_nbr_embed.aggr_embed[0].bias.data, \
    self_nbr_embed.aggr_embed[2].weight.data, self_nbr_embed.aggr_embed[2].bias.data, \
    self_nbr_embed.aggr_embed[3].weight.data, self_nbr_embed.aggr_embed[3].bias.data, \
    self_lin_q.weight.data, self_lin_q.bias.data, \
    self_lin_k.weight.data, self_lin_k.bias.data, \
    self_lin_v.weight.data, self_lin_v.bias.data, \
    self_lin_ih.weight.data, self_lin_ih.bias.data, \
    self_lin_hh.weight.data, self_lin_hh.bias.data, \
    self_lin_self.weight.data, self_lin_self.bias.data, \
    self_out_proj.weight.data, self_out_proj.bias.data, \
    self_norm2.weight.data, self_norm2.bias.data, \
    self_mlp[0].weight.data, self_mlp[0].bias.data, \
    self_mlp[2].weight.data, self_mlp[2].bias.data = inputs


def infer_pt():
    center_embed = self_center_embed(
        torch.matmul(x.view(20, x.shape[0] // 20, -1).unsqueeze(-2),
                        rotate_mat.expand(20, rotate_mat.shape[0], 2, 2)).squeeze(-2))
    center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                self_bos_token.unsqueeze(-2),
                                center_embed).reshape(x.shape[0], -1)
    center_embed1 = center_embed
    center_embed = self_norm1(center_embed)
    center_embed_i = center_embed.index_select(0, edge_index[0])
    x_j = x.index_select(0, edge_index[1])

    index = edge_index[0]
    dim_size = size_i = center_embed.size(0)

    # message
    center_rotate_mat = rotate_mat.repeat(20, 1, 1)[edge_index[0]]
    nbr_embed = self_nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
    query = self_lin_q(center_embed_i).view(-1, 8, 64 // 8)
    key = self_lin_k(nbr_embed).view(-1, 8, 64 // 8)
    value = self_lin_v(nbr_embed).view(-1, 8, 64 // 8)
    scale = (64 // 8) ** 0.5
    alpha = (query * key).sum(dim=-1) / scale
    alpha = softmax(alpha, index, size_i)
    out = value * alpha.unsqueeze(-1)

    # aggregate
    boradcast_size = [1] * out.dim()
    boradcast_size[0] = -1
    index = index.view(boradcast_size).expand_as(out)

    size = list(out.size())
    size[0] = dim_size
    out = out.new_zeros(size).scatter_add_(0, index, out)

    # update
    out = out.view(-1, 64)
    gate = torch.sigmoid(self_lin_ih(out) + self_lin_hh(center_embed))
    out = out + gate * (self_lin_self(center_embed) - out)

    center_embed = self_out_proj(out)
    center_embed = center_embed1 + center_embed
    center_embed = center_embed + self_mlp(self_norm2(center_embed))
    torch.cuda.synchronize()
    return center_embed


debug_gt = x
print(debug_gt.shape)
debug_out = torch.zeros_like(debug_gt)
out = torch.zeros_like(out_gt)
inputs = [x, debug_out, out] + inputs[2:]
for _ in range(20):
    start_time = time.time()
    flash_gat.flash_gat(*inputs)
    # out = infer_pt()
    print("Time elapsed: ", (time.time() - start_time) * 1000, "ms")
print(torch.max((out - out_gt).abs(), dim=0))
# print("\n ===== Debug output ===== ")
# print(torch.max((debug_out - debug_gt).abs(), dim=0))
# print(debug_out[338045:338052])
# print(debug_gt[338045:338052])
