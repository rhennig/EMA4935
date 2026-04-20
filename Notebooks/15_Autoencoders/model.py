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

###################################################################################################

 ######  ##          ###     ######   ######  ########  ######  
##    ## ##         ## ##   ##    ## ##    ## ##       ##    ## 
##       ##        ##   ##  ##       ##       ##       ##       
##       ##       ##     ##  ######   ######  ######    ######  
##       ##       #########       ##       ## ##             ## 
##    ## ##       ##     ## ##    ## ##    ## ##       ##    ## 
 ######  ######## ##     ##  ######   ######  ########  ######

####################################################################################################

class AutoEncoder(torch.nn.Module):
#
### Initialization
#
    def __init__(self, input_nf):
        super(AutoEncoder, self).__init__()

        self.input_nf = input_nf

        self.fc01 = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_nf*input_nf, out_features=108, bias=True),
            torch.nn.ReLU()
        )

        self.fc02 = torch.nn.Sequential(
            torch.nn.Linear(in_features=108, out_features=54, bias=True),
            torch.nn.ReLU()
        )

        self.fc03 = torch.nn.Sequential(
            torch.nn.Linear(in_features=54, out_features=18, bias=True)
        )

        ...

        self.fc06 = torch.nn.Sequential(
            torch.nn.Linear(in_features=108, out_features=input_nf*input_nf, bias=True),
            torch.nn.Sigmoid()
        )
#
### Encode
#
    def Encoding(self, adjacency):

        x      = self.fc01(adjacency)
        x      = self.fc02(x)
        latent = self.fc03(x)

        return latent
#
### Decode
#
    def Decoding(self, latent):


        return adjacency
#
### Forward pass
#
    def forward(self, adjacency):

        # Reshape the adjacency matrix to a vector
        adjacency  = adjacency.view(-1, self.input_nf*self.input_nf)

        latent     = self.Encoding(adjacency)
        adjacency  = self.Decoding(latent)

        # Reshape the output vector back to an adjacency matrix
        adjacency  = adjacency.view(-1, self.input_nf, self.input_nf)

        return adjacency


####################################################################################################

##     ##  #######  ########  ##     ## ##       ########  ######  
###   ### ##     ## ##     ## ##     ## ##       ##       ##    ## 
#### #### ##     ## ##     ## ##     ## ##       ##       ##       
## ### ## ##     ## ##     ## ##     ## ##       ######    ######  
##     ## ##     ## ##     ## ##     ## ##       ##             ## 
##     ## ##     ## ##     ## ##     ## ##       ##       ##    ## 
##     ##  #######  ########   #######  ######## ########  ######

####################################################################################################



####################################################################################################
