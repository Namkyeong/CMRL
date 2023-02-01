import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GATConv, GINConv, GINEConv

class GCN(nn.Module):
    
    def __init__(self,
                node_input_dim = 52,
                node_hidden_dim = 52,
                out_dim = 52,
                num_step_message_passing = 3
                ):

        super().__init__()
        
        layers = []

        layers.append((GCNConv(node_input_dim, node_hidden_dim), 'x, edge_index -> x'))        
        layers.append(nn.ReLU())
        
        for i in range(1, num_step_message_passing):
            layers.append((GCNConv(node_hidden_dim, node_hidden_dim), 'x, edge_index -> x'))           
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(node_hidden_dim, out_dim))
    
        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):

        if len(data.edge_index) == 0:
            data.edge_index = torch.tensor([[0, 0]]).T.cuda()

        return self.model(data.x, data.edge_index)


class GAT(nn.Module):
    
    def __init__(self,
                node_input_dim = 52,
                node_hidden_dim = 52,
                out_dim = 52,
                num_step_message_passing = 3
                ):

        super().__init__()
        
        layers = []

        layers.append((GATConv(node_input_dim, node_hidden_dim), 'x, edge_index -> x'))        
        layers.append(nn.ReLU())
        
        for i in range(1, num_step_message_passing):
            layers.append((GATConv(node_hidden_dim, node_hidden_dim), 'x, edge_index -> x'))           
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(node_hidden_dim, out_dim))
    
        self.model = Sequential('x, edge_index', layers)

    def forward(self, data):

        if len(data.edge_index) == 0:
            data.edge_index = torch.tensor([[0, 0]]).T.cuda()

        return self.model(data.x, data.edge_index)


class GIN(nn.Module):
    
    def __init__(self,
                node_input_dim = 52,
                node_hidden_dim = 52,
                out_dim = 52,
                num_step_message_passing = 3,
                dropout = 0.0,
                ):

        super(GIN, self).__init__()
        
        self.num_layers = num_step_message_passing
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for _ in range(num_step_message_passing):
            mlp = nn.Sequential(
                nn.Linear(node_hidden_dim, 2 * node_hidden_dim),
                # BatchNorm(2 * node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * node_hidden_dim, node_hidden_dim)
            )
            self.layers.append(GINConv(mlp, train_eps=True))
            # layers.append(BatchNorm(node_hidden_dim))

        self.lin = nn.Linear(node_hidden_dim * (num_step_message_passing + 1), out_dim)

    def forward(self, x, edge_index):
        
        h_list = [x]

        for layer in range(self.num_layers):
            h = F.relu(self.layers[layer](h_list[layer], edge_index))
            h = F.dropout(h, p = self.dropout, training = self.training)
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        
        output = self.lin(node_representation)

        return output 


class GINE(nn.Module):
    
    def __init__(self,
                node_input_dim = 52,
                node_hidden_dim = 52,
                out_dim = 52,
                num_step_message_passing = 3,
                dropout = 0.0,
                ):

        super(GINE, self).__init__()
        
        self.num_layers = num_step_message_passing
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for _ in range(num_step_message_passing):
            mlp = nn.Sequential(
                nn.Linear(node_hidden_dim, 2 * node_hidden_dim),
                BatchNorm(2 * node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * node_hidden_dim, node_hidden_dim)
            )
            self.layers.append((GINEConv(mlp, train_eps=True)))
        
        self.lin = nn.Linear(node_hidden_dim * (num_step_message_passing + 1), out_dim)

    def forward(self, x, edge_index, edge_attr):

        h_list = [x]

        for layer in range(self.num_layers):
            h = F.relu(self.layers[layer](h_list[layer], edge_index, edge_attr))
            h = F.dropout(h, p = self.dropout, training = self.training)
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        
        output = self.lin(node_representation)

        return output 


class GNN(torch.nn.Module):
    """
    
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr = "add"))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation