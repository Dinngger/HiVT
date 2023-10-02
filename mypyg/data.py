import copy
import errno
import os
import os.path as osp
import re
import warnings
import inspect
from typing import (
    Any,
    Literal,
    Callable,
    List,
    Dict,
    Optional,
    Tuple,
    Union,
    Mapping,
    Sequence
)

import numpy as np
import torch
from torch import Tensor


class Data:
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        normal (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    """
    def __init__(self, num_nodes: Optional[int], kwargs: Dict[str, Tensor]):
        self.datas: Dict[str, Tensor] = kwargs
        self.__num_nodes__: Optional[int] = num_nodes
        self.__num_graphs__: int = 1

    def __getitem__(self, key: str):
        r"""Gets the data of the attribute :obj:`key`."""
        return self.datas[key]

    def __setitem__(self, key: str, value: Optional[Tensor]):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        if value is not None:
            self.datas[key] = value

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        return self.datas.keys()

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __cat_dim__(self, key: str, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # By default, concatenate sparse matrices diagonally.
        # Concatenate `*index*` and `*face*` attributes in the last dimension.
        if key.find('index') != -1:
            return -1
        return 0

    def __inc__(self, key: str, value) -> int:
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        num_nodes = self.num_nodes
        assert num_nodes is not None
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [num_nodes]])
        if key.find('index') != -1:
            return num_nodes
        else:
            return 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if self.__num_nodes__ is not None:
            return self.__num_nodes__
        for key in ['x', 'pos', 'normal']:
            if key in self.keys:
                item = self[key]
                return item.size(self.__cat_dim__(key, item))
        return None

    def contiguous(self):
        for key in sorted(self.keys):
            if key in self.keys:
                self[key] = self[key].contiguous()
        return self

    def to(self, device: torch.device):
        for key in sorted(self.keys):
            if key in self.keys:
                self[key] = self[key].to(device=device)
        return self

    def cpu(self):
        for key in sorted(self.keys):
            if key in self.keys:
                self[key] = self[key].cpu()
        return self

    def pin_memory(self):
        for key in sorted(self.keys):
            if key in self.keys:
                self[key] = self[key].pin_memory()
        return self


IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def combine_batch_data(data_list: List[Data]):
    keys: List[str] = list(data_list[0].keys)

    batch = Data(None, {})
    batch.__num_graphs__ = len(data_list)

    temp_dict: Dict[str, List[Tensor]] = {}
    for key in keys:
        temp_dict[key] = []

    slices = {key: [0] for key in keys}
    cumsum = {key: [0] for key in keys}
    cat_dims: Dict[str, int] = {}
    num_nodes_list: List[int] = []
    for data in data_list:
        for key in keys:
            item = data[key]

            # Increase values by `cumsum` value.
            cum = cumsum[key][-1]
            if isinstance(item, Tensor) and item.dtype != torch.bool:
                if not isinstance(cum, int) or cum != 0:
                    item = item + cum

            # Gather the size of the `cat` dimension.
            size = 1
            cat_dim = data.__cat_dim__(key, data[key])
            cat_dims[key] = cat_dim

            # Add a batch dimension to items whose `cat_dim` is `None`:
            if item.size():
                size = item.size(cat_dim)

            temp_dict[key].append(item)  # Append item to the attribute list.

            slices[key].append(size + slices[key][-1])
            inc = data.__inc__(key, item)
            cumsum[key].append(inc + cumsum[key][-1])

        data_num_nodes = data.__num_nodes__
        if data_num_nodes is not None:
            num_nodes_list.append(data_num_nodes)

    # batch.__slices__ = slices
    # batch.__cumsum__ = cumsum
    # batch.__cat_dims__ = cat_dims
    # batch.__num_nodes_list__ = num_nodes_list

    ref_data = data_list[0]
    for key in keys:
        items = temp_dict[key]
        item = items[0]
        cat_dim = ref_data.__cat_dim__(key, item)
        if item.size():
            batch[key] = torch.cat(items, cat_dim)
        else:
            batch[key] = torch.tensor(items)

    return batch.contiguous()


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def _repr(obj: Any) -> str:
    if obj is None: 
        return 'None' 
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', str(obj)) 


def makedirs(path: str):
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class Dataset(torch.utils.data.Dataset):
    r"""Dataset base class for creating graph datasets.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html>`__ for the accompanying tutorial.

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__()

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._indices: Optional[Sequence] = None

        if 'process' in self.__class__.__dict__.keys():
            self._process()

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        print('Done!')

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a PyTorch :obj:`LongTensor` or a :obj:`BoolTensor`, or a numpy
        :obj:`np.array`, will return a subset of the dataset at the specified
        indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)


def collate(batch):
    from utils import create_data
    batch = [create_data(**kargs) for kargs in batch]
    return combine_batch_data(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        exclude_keys (list or tuple, optional): Will exclude each key in the
            list. (default: :obj:`[]`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=collate, **kwargs)
