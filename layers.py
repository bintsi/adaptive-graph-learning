import torch
from torch import nn

#########################################################
# This section of code has been adapted from lcosmo/DGM_pytorch#
# Modified by Margarita Bintsi#
#########################################################

#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x,x)**2
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x

#Cosine similarity
def pairwise_cosine_distances(x, dim=-1):
    dist = 1 - torch.mm(torch.nn.functional.normalize(x[0], p=2, dim=-1), torch.nn.functional.normalize(x[0], p=2, dim=-1).T).unsqueeze(0)
    return dist, x

class MLP(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))
            if li==len(layers_size)-1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))              
        self.MLP = nn.Sequential(*layers)      
    def forward(self, x, e=None):
        x = self.MLP(x)
        return x

class Attention(nn.Module): 
    def __init__(self, layers_size,final_activation=False, dropout=0):
        super(Attention, self).__init__()
        layers = []
        for li in range(1,len(layers_size)):
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1],layers_size[li]))

            if li==len(layers_size)-1 and not final_activation:
                continue 
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, e=None):
        x = self.mlp(x)
        # Return attention weights per phenotype
        x = torch.sum(x, 1)
        x = (x - torch.min(x, dim=1)[0])/(torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]) # normalization
        return x
    
class Identity(nn.Module):
    def __init__(self,retparam=None):
        self.retparam=retparam
        super(Identity, self).__init__()
        
    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params

class GraphLearning(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(GraphLearning, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())

        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k
        
        self.debug=False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        elif distance == 'hyperbolic':
            self.distance = pairwise_poincare_distances
        elif distance == 'cosine':
            self.distance = pairwise_cosine_distances
        else:
            raise ValueError('There is not this kind of distance.')
        
    def forward(self, x, A, phenotypes, not_used1=None, not_used2=None, fixedges=None):
        # Estimate attetion coefficients for every phenotype (weights)
        att_weights = self.embed_f(phenotypes)  
        # Give attention weights to the phenotypes
        phenotypes_weighted = att_weights * phenotypes
        
        if self.training:
            D, _x = self.distance(phenotypes_weighted) 
            #sampling here
            edges_hat, logprobs = self.sample_without_replacement(D)
                
        else:
            with torch.no_grad():
                D, _x = self.distance(phenotypes_weighted) 
                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)
        return x, edges_hat, phenotypes, logprobs, att_weights

    def sample_without_replacement(self, logits):
        b,n,_ = logits.shape
        logits = logits * torch.exp(torch.clamp(self.temperature,-5,5))
        
        q = torch.rand_like(logits) + 1e-8
        lq = (logits-torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(-lq,self.k)  
        rows = torch.arange(n).view(1,n,1).to(logits.device).repeat(b,1,self.k)
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2)

        if self.sparse:
            return (edges+(torch.arange(b).to(logits.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs