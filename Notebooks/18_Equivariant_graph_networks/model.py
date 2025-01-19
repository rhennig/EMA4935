#############################################################################################################################

##       #### ########  ########     ###    ########  #### ########  ######
##        ##  ##     ## ##     ##   ## ##   ##     ##  ##  ##       ##    ##
##        ##  ##     ## ##     ##  ##   ##  ##     ##  ##  ##       ##
##        ##  ########  ########  ##     ## ########   ##  ######    ######
##        ##  ##     ## ##   ##   ######### ##   ##    ##  ##             ##
##        ##  ##     ## ##    ##  ##     ## ##    ##   ##  ##       ##    ##
######## #### ########  ##     ## ##     ## ##     ## #### ########  ######

#############################################################################################################################

import torch
import torch_geometric

#############################################################################################################################

 ######  ##          ###     ######   ######  ########  ######
##    ## ##         ## ##   ##    ## ##    ## ##       ##    ##
##       ##        ##   ##  ##       ##       ##       ##
##       ##       ##     ##  ######   ######  ######    ######
##       ##       #########       ##       ## ##             ##
##    ## ##       ##     ## ##    ## ##    ## ##       ##    ##
 ######  ######## ##     ##  ######   ######  ########  ######

#############################################################################################################################
### Equivariant Graph Convolutional Neural Network

class EquivariantGraphNetwork(torch.nn.Module):
#
### Initialization
#
    def __init__(self, nodes=2, edge_nf=2, node_nf=2, hidden_nf=4, activation=torch.nn.SiLU(), aggregation='sum'):
        super(EquivariantGraphNetwork, self).__init__()
        
        self.aggregation = aggregation
        self.hidden_nf   = hidden_nf
        
        self.EdgeMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf + hidden_nf + edge_nf + 1, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation
        )

        self.NodeMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf + hidden_nf + node_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, hidden_nf)
        )
        
        self.CoordMultiLayerPerceptron = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1),
        )

        self.EdgeAttention = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1),
            torch.nn.Sigmoid()
        )
        
        self.Embedding = torch.nn.Sequential(
            torch.nn.Linear(nodes, hidden_nf),
            activation
        )
        self.Encoding = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, hidden_nf)
        )

        self.Decoding = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1)
        )

        self.apply(initialize)
#
### Perform operation using the average or sum
#
    def SegmentOperation(self, data, index, segments, option='sum'):
        
        index  = index.unsqueeze(-1).expand(-1, data.size(1))
        
        result_shape = (segments, data.size(1))
        
        result = data.new_full(result_shape, 0)
        result.scatter_add_(0, index, data)
        
        if option == 'mean':
            count = data.new_full(result_shape, 0)
            count.scatter_add_(0, index, torch.ones_like(data))

        if option == 'mean': result = result/count.clamp(min=1.0)
        
        return result
#
### Operations on the edges
#
    def EdgeOperation(self, source, target, radial, edge_attr):
        
        out = torch.cat([source, target, radial], dim=1) if edge_attr is None else torch.cat([source, target, radial, edge_attr], dim=1)
            
        out = self.EdgeMultiLayerPerceptron(out)
        
        out = out*self.EdgeAttention(out)
            
        return out
#
### Operations on the nodes
#
    def NodeOperation(self, nodes, edges, edge_attr, node_attr):
        
        row, col = edges
        agg      = self.SegmentOperation(edge_attr, row, segments=nodes.size(0), option='sum')
        
        agg = torch.cat([nodes, agg], dim=1) if node_attr is None else torch.cat([nodes, agg, node_attr], dim=1)
            
        out = self.NodeMultiLayerPerceptron(agg)

        out = nodes + out
            
        return out
#
### Aggregation of features
#
    def CoordinateOperation(self, coord, edges, rij, edge_feat):
        
        row, col = edges
        
        trans    = rij*self.CoordMultiLayerPerceptron(edge_feat)
        
        agg      = self.SegmentOperation(trans, row, segments=coord.size(0), option=self.aggregation)
                    
        return coord + agg
#
### Coordinates transformation
#
    def RadialOperation(self, edges, coord, batch):

        
        row, col = edges
        idx      = batch[row]
        rij      = coord[row] - coord[col]
        radial   = torch.sum(rij*rij, 1).unsqueeze(1)

        return radial, rij
#
### Forward pass
#
    def forward(self, nodes, coord, edges, edge_attr=None, node_attr=None, batch=None, size=1):
        
        row, col  = edges
                
        nodes     = self.Embedding(nodes)
        
        r, rij    = self.RadialOperation(edges, coord, batch)

        edge_feat = self.EdgeOperation(nodes[row], nodes[col], r, edge_attr)
        
        coord     = self.CoordinateOperation(coord, edges, rij, edge_feat)
        
        nodes     = self.NodeOperation(nodes, edges, edge_feat, node_attr)
        
        nodes     = self.Encoding(nodes)
        
        nodes     = nodes.view(-1, nodes.size(0), self.hidden_nf)
        
        nodes     = torch_geometric.nn.global_add_pool(nodes, batch, size=size)

        nodes     = nodes.squeeze(0)

        out       = self.Decoding(nodes)
        
        return out

#############################################################################################################################
### Initilization of layers

def initialize(layer):

    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight.data, gain=1e-2)

        if layer.bias is not None: torch.nn.init.constant_(layer.bias.data, 1.0)
        
####################################################################################################

##     ##  #######  ########  ##     ## ##       ########  ######  
###   ### ##     ## ##     ## ##     ## ##       ##       ##    ## 
#### #### ##     ## ##     ## ##     ## ##       ##       ##       
## ### ## ##     ## ##     ## ##     ## ##       ######    ######  
##     ## ##     ## ##     ## ##     ## ##       ##             ## 
##     ## ##     ## ##     ## ##     ## ##       ##       ##    ## 
##     ##  #######  ########   #######  ######## ########  ######

####################################################################################################

def passdata(data=None, network=None, criterion=None, optimizer=None, train=False, batch=None, size=1):

    if train:
        optimizer.zero_grad()
        
        output = network(data.node_feat, data.coord, data.edge_index, data.edge_attr, data.node_attr, batch=batch, size=size)
        loss   = criterion(output, data.y)
        
        loss.backward()
        optimizer.step()
        
        return output, loss.item()

    with torch.no_grad():
        output = network(data.node_feat, data.coord, data.edge_index, data.edge_attr, data.node_attr, batch=batch, size=size)
        loss   = criterion(output, data.y)

    return output, loss.item()

####################################################################################################

def batches(dataset, network=None, criterion=None, optimizer=None, train=False):

    batch_loss = []

    for batch in dataset:

        _, loss = passdata(data=batch, network=network, criterion=criterion, optimizer=optimizer, train=train, batch=batch.batch, size=batch.batch_size)

        batch_loss.append(loss)

    mean_loss = sum(batch_loss)/len(batch_loss)

    return mean_loss

####################################################################################################

