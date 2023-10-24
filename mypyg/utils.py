from typing import Tuple, Optional
from torch import Tensor
from torch.utils.cpp_extension import load


def subgraph(
    subset: Tensor,
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    node_mask = subset

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)


def scatter(src: Tensor, index: Tensor, dim: int,
            dim_size: int, reduce: str = 'sum') -> Tensor:
    dim = src.dim() + dim if dim < 0 else dim

    size = list(src.size())
    size[dim] = dim_size

    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    # For "min" and "max" reduction, we prefer `scatter_reduce_` on CPU or
    # in case the input does not require gradients:
    if reduce == 'min' or reduce == 'max':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_reduce_(
            dim, index, src, reduce=f'a{reduce}', include_self=False)

    raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")


gat_cu = load(name="gat", sources=["ops/gat.cu"])

def softmax(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    dim: int = 0,
) -> Tensor:
    N = num_nodes
    src_max = gat_cu.scatter_max(src.detach(), index, dim, N)
    # src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
    out = src - src_max.index_select(dim, index)
    out = out.exp()
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
    out_sum = out_sum.index_select(dim, index)
    return out / out_sum
