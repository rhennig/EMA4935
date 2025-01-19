####################################################################################################

##       #### ########  ########     ###    ########  #### ########  ######  
##        ##  ##     ## ##     ##   ## ##   ##     ##  ##  ##       ##    ## 
##        ##  ##     ## ##     ##  ##   ##  ##     ##  ##  ##       ##       
##        ##  ########  ########  ##     ## ########   ##  ######    ######  
##        ##  ##     ## ##   ##   ######### ##   ##    ##  ##             ## 
##        ##  ##     ## ##    ##  ##     ## ##    ##   ##  ##       ##    ## 
######## #### ########  ##     ## ##     ## ##     ## #### ########  ######

####################################################################################################

import torch
import torch_geometric

###################################################################################################

 ######  ##          ###     ######   ######  ########  ######  
##    ## ##         ## ##   ##    ## ##    ## ##       ##    ## 
##       ##        ##   ##  ##       ##       ##       ##       
##       ##       ##     ##  ######   ######  ######    ######  
##       ##       #########       ##       ## ##             ## 
##    ## ##       ##     ## ##    ## ##    ## ##       ##    ## 
 ######  ######## ##     ##  ######   ######  ########  ######

####################################################################################################

class GraphAttentionNetwork(torch.nn.Module):
#
### Initialization
#
    def __init__(self, input_nf=2, hidden_nf=4, output_nf=1, attention_nf=1, reduce='cat', slope=0.2, drop=0.0, activation=torch.nn.ReLU()):
        super(GraphAttentionNetwork, self).__init__()
        
        self.reduce     = reduce
        self.n_heads    = attention_nf
        self.hidden_mlp = hidden_nf
                
        if reduce == 'cat': assert output_nf % attention_nf == 0
        
        self.hidden_nf  = output_nf // attention_nf if reduce == 'cat' else output_nf
            
        self.embedding  = torch.nn.Linear(input_nf, self.hidden_nf * attention_nf, bias=False)
        
        self.attention  = torch.nn.Linear(2*self.hidden_nf, 1, bias=False)
        
        self.activation = torch.nn.LeakyReLU(negative_slope=slope)
        
        self.softmax    = torch.nn.Softmax(dim=1)
        
        self.dropout    = torch.nn.Dropout(p=drop)
        
        self.encode     = torch.nn.Sequential(
                          torch.nn.Linear(output_nf, hidden_nf),
                          activation,
                          torch.nn.Linear(hidden_nf, hidden_nf)
                          )
        
        self.decode     = torch.nn.Sequential(
                          torch.nn.Linear(hidden_nf, hidden_nf),
                          activation,
                          torch.nn.Linear(hidden_nf, 1)
                          )
        
        self.apply(initialize)
#
### Forward pass
#
    def forward(self, nodes, adjacency, batch=None, size=1):
        
        adjacency = adjacency.unsqueeze(2)
        
        n_nodes   = nodes.shape[0]
        
        g         = self.embedding(nodes)
        g         = g.view(n_nodes, self.n_heads, self.hidden_nf)
        
        g_repeat  = g.repeat(n_nodes, 1, 1)
        
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        
        g_concat  = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        
        g_concat  = g_concat.view(n_nodes, n_nodes, self.n_heads, 2*self.hidden_nf)
        
        e         = self.activation(self.attention(g_concat))
        e         = e.squeeze(-1)
        
        # Safety checks before proceeding
        
        assert adjacency.shape[0] == 1 or adjacency.shape[0] == n_nodes
        assert adjacency.shape[1] == 1 or adjacency.shape[1] == n_nodes
        assert adjacency.shape[2] == 1 or adjacency.shape[2] == self.n_heads
        
        e         = e.masked_fill( adjacency == 0, float('-inf') )
        
        a         = self.softmax(e)
        a         = self.dropout(a)
        
        attn_rslt = torch.einsum('ijh,jhf->ihf', a, g)
        
        if self.reduce == 'cat':
            nodes = attn_rslt.reshape(n_nodes, self.n_heads*self.hidden_nf)
            
        else:
            nodes = attn_rslt.mean(dim=1)
            
        # Graph decoding
            
        nodes = self.encode(nodes)
        
        nodes = nodes.view(-1, nodes.size(0), self.hidden_mlp)
        
        nodes = torch_geometric.nn.global_add_pool(nodes, batch, size=size)

        nodes = nodes.squeeze(0)
        
        output = self.decode(nodes)
            
        return output
    
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
        
        output = network(data.node_feat, data.adjacency, batch=batch, size=size)
        loss   = criterion(output, data.y)
        
        loss.backward()
        optimizer.step()
        
        return output, loss.item()

    with torch.no_grad():
        output = network(data.node_feat, data.adjacency, batch=batch, size=size)
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

