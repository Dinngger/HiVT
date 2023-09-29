from inspect import Parameter
from typing import (
    Any,
    List,
    Set,
    Optional,
)

import torch
from torch import Tensor
from mypyg.utils import scatter

from mypyg.typing import Adj, Size, SparseTensor
from mypyg.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    ptr2index
)


class SumAggregation(torch.nn.Module):
    r"""An aggregation operator that sums up features across a set of elements

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
    def __call__(self, x: Tensor, index: Optional[Tensor] = None,
                 ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                 dim: int = -2, **kwargs) -> Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of "
                             f"source tensor with {x.dim()} dimensions")

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got "
                                 f"'{dim_size}' but expected "
                                 f"'{ptr.numel() - 1}')")

        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        try:
            return super().__call__(x, index, ptr, dim_size, dim, **kwargs)
        except (IndexError, RuntimeError) as e:
            if index is not None:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got "
                                     f"'{dim_size}' but expected "
                                     f">= '{int(index.max()) + 1}')")
            raise e
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
    def reduce(self, x: Tensor, index: Optional[Tensor] = None,
               ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
               dim: int = -2, reduce: str = 'sum') -> Tensor:
        assert ptr is None
        assert index is not None
        return scatter(x, index, dim, dim_size, reduce)


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (str or [str] or Aggregation, optional): The aggregation scheme
            to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective aggregation function in case it gets automatically
            resolved. (default: :obj:`None`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
        decomposed_layers (int, optional): The number of feature decomposition
            layers, as introduced in the `"Optimizing Memory Efficiency of
            Graph Neural Networks on Edge Computing Platforms"
            <https://arxiv.org/abs/2104.03058>`_ paper.
            Feature decomposition reduces the peak memory usage by slicing
            the feature dimensions into separated feature decomposition layers
            during GNN aggregation.
            This method can accelerate GNN execution on CPU-based platforms
            (*e.g.*, 2-3x speedup on the
            :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
            models such as :class:`~torch_geometric.nn.models.GCN`,
            :class:`~torch_geometric.nn.models.GraphSAGE`,
            :class:`~torch_geometric.nn.models.GIN`, etc.
            However, this method is not applicable to all GNN operators
            available, in particular for operators in which message computation
            can not easily be decomposed, *e.g.* in attention-based GNNs.
            The selection of the optimal value of :obj:`decomposed_layers`
            depends both on the specific graph dataset and available hardware
            resources.
            A value of :obj:`2` is suitable in most cases.
            Although the peak memory usage is directly associated with the
            granularity of feature decomposition, the same is not necessarily
            true for execution speedups. (default: :obj:`1`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(
        self,
        aggr: str = "add",
        *,
        node_dim: int = -2,
        decomposed_layers: int = 1,
        **kwargs,
    ):
        super().__init__()

        assert aggr == "add"
        self.aggr = str(aggr)
        self.aggr_module = SumAggregation()

        self.node_dim = node_dim
        assert decomposed_layers == 1

        # Support for "fused" message passing.
        self.fuse = False

        # Support for explainability.
        self._explain = False
        self._edge_mask = None
        self._loop_mask = None
        self._apply_sigmoid = True

    def forward(self, *args, **kwargs) -> Any:
        r"""Runs the forward pass of the module."""
        pass

    def _check_input(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if is_sparse(edge_index):
            the_size[0] = edge_index.size(1)
            the_size[1] = edge_index.size(0)
            return the_size
        elif isinstance(edge_index, Tensor):
            int_dtypes = (torch.uint8, torch.int8, torch.int32, torch.int64)

            if edge_index.dtype not in int_dtypes:
                raise ValueError(f"Expected 'edge_index' to be of integer "
                                 f"type (got '{edge_index.dtype}')")
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.dim()} dimensions)")
            if edge_index.size(0) != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.size(0)}')")
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports integer tensors of '
             'shape `[2, num_messages]`, `torch_sparse.SparseTensor` or '
             '`torch.sparse.Tensor` for argument `edge_index`.'))

    def _set_size(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def _lift(self, src, edge_index, dim):
        if is_torch_sparse_tensor(edge_index):
            assert dim == 0 or dim == 1
            if edge_index.layout == torch.sparse_coo:
                index = edge_index._indices()[1 - dim]
            elif edge_index.layout == torch.sparse_csr:
                if dim == 0:
                    index = edge_index.col_indices()
                else:
                    index = ptr2index(edge_index.crow_indices())
            elif edge_index.layout == torch.sparse_csc:
                if dim == 0:
                    index = ptr2index(edge_index.ccol_indices())
                else:
                    index = edge_index.row_indices()
            else:
                raise ValueError(f"Unsupported sparse tensor layout "
                                 f"(got '{edge_index.layout}')")
            return src.index_select(self.node_dim, index)

        elif isinstance(edge_index, Tensor):
            try:
                index = edge_index[dim]
                return src.index_select(self.node_dim, index)
            except (IndexError, RuntimeError) as e:
                if index.min() < 0 or index.max() >= src.size(self.node_dim):
                    raise IndexError(
                        f"Encountered an index error. Please ensure that all "
                        f"indices in 'edge_index' point to valid indices in "
                        f"the interval [0, {src.size(self.node_dim) - 1}] "
                        f"(got interval "
                        f"[{int(index.min())}, {int(index.max())}])")
                else:
                    raise e

                if index.numel() > 0 and index.min() < 0:
                    raise ValueError(
                        f"Found negative indices in 'edge_index' (got "
                        f"{index.min().item()}). Please ensure that all "
                        f"indices in 'edge_index' point to valid indices "
                        f"in the interval [0, {src.size(self.node_dim)}) in "
                        f"your node feature matrix and try again.")

                if (index.numel() > 0
                        and index.max() >= src.size(self.node_dim)):
                    raise ValueError(
                        f"Found indices in 'edge_index' that are larger "
                        f"than {src.size(self.node_dim) - 1} (got "
                        f"{index.max().item()}). Please ensure that all "
                        f"indices in 'edge_index' point to valid indices "
                        f"in the interval [0, {src.size(self.node_dim)}) in "
                        f"your node feature matrix and try again.")

                raise e

        elif isinstance(edge_index, SparseTensor):
            if dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
            elif dim == 1:
                row = edge_index.storage.row()
                return src.index_select(self.node_dim, row)

        raise ValueError(
            ('`MessagePassing.propagate` only supports integer tensors of '
             'shape `[2, num_messages]`, `torch_sparse.SparseTensor` '
             'or `torch.sparse.Tensor` for argument `edge_index`.'))

    def _collect(self, data, edge_index, size, dim):
        if isinstance(data, (tuple, list)):
            assert len(data) == 2
            if isinstance(data[1 - dim], Tensor):
                self._set_size(size, 1 - dim, data[1 - dim])
            data = data[dim]

        if isinstance(data, Tensor):
            self._set_size(size, dim, data)
            data = self._lift(data, edge_index, dim)
        return data

        out['adj_t'] = None
        out['edge_index'] = edge_index
        out['edge_index_i'] = edge_index[i]
        out['edge_index_j'] = edge_index[j]
        out['ptr'] = None

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)
