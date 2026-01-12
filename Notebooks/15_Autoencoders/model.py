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

        self.fc04 = torch.nn.Sequential(
            torch.nn.Linear(in_features=18, out_features=54, bias=True),
            torch.nn.ReLU()
        )

        self.fc05 = torch.nn.Sequential(
            torch.nn.Linear(in_features=54, out_features=108, bias=True),
            torch.nn.ReLU()
        )

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

        x         = self.fc04(latent)
        x         = self.fc05(x)
        adjacency = self.fc06(x)

        return adjacency
#
### Forward pass
#
    def forward(self, adjacency):

        adjacency  = adjacency.view(-1, self.input_nf*self.input_nf)
        latent     = self.Encoding(adjacency)
        adjacency  = self.Decoding(latent)
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
