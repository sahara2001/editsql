import math

import torch
from torch import Tensor
from torch.nn import Parameter as Param, init
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

#(only work for this project!)
class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, input_dim, num_timesteps, num_edge_types, aggr='add', bias=True, dropout=0):
        super(GatedGraphConv, self).__init__(aggr)

        self._input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.num_edge_types = num_edge_types

        self.weight = Param(Tensor(num_timesteps, num_edge_types, input_dim, input_dim))
        self.bias = Param(Tensor(num_timesteps, num_edge_types, input_dim))
        self.rnn = torch.nn.GRUCell(input_dim, input_dim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for t in range(self.num_timesteps):
            for e in range(self.num_edge_types):
                torch.nn.init.xavier_uniform_(self.weight[t, e])
        init.uniform_(self.bias, -0.01, 0.01)
        self.rnn.reset_parameters()

    def forward(self, x, edge_indices):
        """"""
        if len(edge_indices) != self.num_edge_types:
            raise ValueError(f'GatedGraphConv constructed with {self.num_edge_types} edge types, '
                             f'but {len(edge_indices)} were passed')

        h = x

        if h.size(1) > self._input_dim:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self._input_dim:
            zero = h.new_zeros(h.size(0), self._input_dim - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for t in range(self.num_timesteps):
            new_h = []
            # print(t)
            for e in range(self.num_edge_types):
                if len(edge_indices[e]) == 0:
                    continue
                m = self.dropout(torch.matmul(h, self.weight[t, e]) + self.bias[t, e])
                # print(edge_indices[e].size())
                # if edge_indices[e].size(0) == 1: # transpose when only one edge encountered (should be (2,1) but (1,2) )
                new_h.append(self.propagate(edge_indices[e].transpose(0,1), size=(x.size(0), x.size(0)), x=m)) # transpose to make edges [2,n] for propagation (only for this project!)
                # else:
                #     new_h.append(self.propagate(edge_indices[e], size=(x.size(0), x.size(0)), x=m))
            m_sum = torch.sum(torch.stack(new_h), dim=0)
            h = self.rnn(m_sum, h)

        return h

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self._input_dim, self.num_timesteps)


if __name__ == '__main__':
    gcn = GatedGraphConv(input_dim=10, num_timesteps=3, num_edge_types=3)
    data = Data(torch.zeros((5, 10)), edge_index=[
        torch.tensor([[0,2],[2,3]]),
        torch.tensor([[1,3],[0,1]]),
        torch.tensor([[1,4],[2,3]]),
    ])
    
    edge1 = [torch.tensor([[ 0,  2],[ 0,  3],[ 0,  4],[ 0,  5],[ 0,  6],[ 0,  7],[ 0,  8],[ 9, 10],[ 9, 11],[ 9, 14], [ 9, 15]]),
            torch.tensor([[ 0,  1],[ 9, 12],[ 9, 13],[ 9, 16]]), 
            torch.tensor([[9, 1]])]
            
    output = gcn(data.x, data.edge_index)
    print(torch.tensor([[9, 1],[0,11]]),torch.tensor([[9, 1]]))
    print(output)
    gcn = GatedGraphConv(input_dim=300, num_timesteps=3, num_edge_types=3)
    emb = torch.zeros((20,300))
    output = gcn(emb,edge1)
    print(output)