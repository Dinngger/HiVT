from typing import (
    Any,
    Tuple,
    List,
    Set,
    Optional,
)

import torch
from torch import Tensor
from mypyg.utils import scatter


class SumAggregation(torch.nn.Module):
    r"""An aggregation operator that sums up features across a set of elements

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """
    def forward(self, x: Tensor, index: Tensor,
                dim_size: int, dim: int = -2) -> Tensor:
        return self.reduce(x, index, dim_size, dim, reduce='sum')

    def reduce(self, x: Tensor, index: Tensor,
               dim_size: int, dim: int = -2, reduce: str = 'sum') -> Tensor:
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

    def _set_size(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def _lift(self, src, edge_index, dim: int):
        index = edge_index[dim]
        return src.index_select(self.node_dim, index)

    def _collect_tuple(self, data: Tuple[Tensor, Tensor], edge_index, size: List[Optional[int]], dim: int):
        assert len(data) == 2
        if isinstance(data[1 - dim], Tensor):
            self._set_size(size, 1 - dim, data[1 - dim])
        data = data[dim]
        return self._collect(data, edge_index, size, dim)

    def _collect(self, data: Tensor, edge_index, size: List[Optional[int]], dim: int):
        if isinstance(data, Tensor):
            self._set_size(size, dim, data)
            data = self._lift(data, edge_index, dim)
        return data

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: int) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return self.aggr_module(inputs, index, dim_size=dim_size,
                                dim=self.node_dim)
