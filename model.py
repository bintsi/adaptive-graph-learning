#########################################################
# This section of code has been adapted from lcosmo/DGM_pytorch#
# Modified by Margarita Bintsi#
#########################################################

import torch
import numpy as np

from torch.nn import ModuleList
from torch_geometric.nn import EdgeConv, ChebConv, GCNConv, GATConv, SAGEConv

import pytorch_lightning as pl
from argparse import Namespace
import torchmetrics

from layers import *

class GraphLearningModel(pl.LightningModule):
    def __init__(self, hparams, config=None):
        super(GraphLearningModel,self).__init__()
        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        self.fc_layers = hparams.fc_layers
        self.dgm_layers = hparams.dgm_layers
        self.test_eval = hparams.test_eval
        self.dropout = hparams.dropout
        self.graph_loss_mae = hparams.graph_loss_mae
        self.k = hparams.k
        self.lr = hparams.lr
        self.task = hparams.task
        self.num_classes = hparams.num_classes
        self.phenotype_columns = hparams.phenotype_columns

        if self.task == "regression":
            self.criterion = 'torch.nn.HuberLoss()'
            # Metrics for regression
            self.mean_absolute_error = torchmetrics.MeanAbsoluteError()
            self.rscore = torchmetrics.PearsonCorrCoef()
        elif self.task == "classification":
            self.criterion= 'torch.nn.functional.binary_cross_entropy_with_logits()'
            # Metrics for classification
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.auc = torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes)
            self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        else:
            raise ValueError('Task should be either regression or classification.')

        # Here we build the models
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(self.dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams.ffun == 'phenotypes':
                    self.graph_f.append(GraphLearning(Attention(dgm_l),k=self.k,distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            
            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            elif hparams.gfun == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            elif hparams.gfun == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
            elif hparams.gfun == 'sage':
                self.node_g.append(SAGEConv(conv_l[0],conv_l[1]))
            elif hparams.gfun == 'chebconv':
                self.node_g.append(ChebConv(conv_l[0],conv_l[1],2))
            else: 
                raise Exception("Function %s not supported" % hparams.gfun)
        
        if self.fc_layers is not None and len(self.fc_layers)>0:
            self.fc = MLP(self.fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        self.avg_mae = None
        
        #torch lightning specific
        self.automatic_optimization = False
        self.debug=False
        
    def forward(self,x, edges=None, phenotypes= None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        graph_x = x.detach()
        lprobslist = []
        att_weights_list = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,phenotypes,lprobs, att_weights = f(graph_x,edges,phenotypes, None, None)
            b,n,d = x.shape

            self.edges=edges
            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.dropout, train=self.training), edges)).view(b,n,-1)

            if lprobs is not None:
                lprobslist.append(lprobs)
                att_weights_list.append(att_weights)
        
        if self.fc_layers is not None and len(self.fc_layers)>0:
            return self.fc(x), torch.stack(att_weights_list,-1) if len(att_weights_list)>0 else None, torch.stack(lprobslist,-1) if len(lprobslist)>0 else None
        return x, torch.stack(att_weights_list,-1) if len(att_weights_list)>0 else None, torch.stack(lprobslist,-1) if len(lprobslist)>0 else None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        X, y, mask, phenotypes, edges = train_batch
        edges = edges[0]
                
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
  
        pred, att_weights, logprobs = self(X, edges, phenotypes)
        if self.task == 'classification':
            train_pred = pred[:,mask.to(torch.bool),:]
            train_lab = y[:,mask.to(torch.bool),:]   
        elif self.task == 'regression':
            pred = pred.squeeze_(-1) 
            train_pred = pred[:,mask.to(torch.bool)]
            train_lab = y[:,mask.to(torch.bool)]
        else:
            raise ValueError('Task should be classification or regression.')

        if self.criterion == 'torch.nn.functional.cross_entropy()':
            loss = torch.nn.functional.cross_entropy(train_pred.view(-1,train_pred.shape[-1]),train_lab.argmax(-1).flatten())
        elif self.criterion == 'torch.nn.functional.binary_cross_entropy_with_logits()':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        elif (self.criterion == 'torch.nn.HuberLoss()'): 
            loss = torch.nn.HuberLoss()(train_pred.squeeze_(), train_lab.squeeze_())
        else: 
            raise ValueError('Choose appropriate loss function.')

        self.manual_backward(loss)

        # Estimate graph loss
        if self.task == 'classification':
            correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
            #GRAPH LOSS
            if logprobs is not None: 
                corr_pred = (train_pred.argmax(-1)==train_lab.argmax(-1)).float().detach() #0 or 1
                if self.avg_accuracy is None:
                    self.avg_accuracy = torch.ones_like(corr_pred)*0.5
                point_w = (self.avg_accuracy-corr_pred)
                graph_loss = point_w * logprobs[:,mask.to(torch.bool),:].exp().mean([-1,-2])
                graph_loss = graph_loss.mean()
                graph_loss.backward()  
                self.log('train_point_w', point_w.mean().detach().cpu())
                self.log('train_graph_loss', graph_loss.detach().cpu())                   
                self.avg_accuracy = self.avg_accuracy.to(corr_pred.device)*0.95 +  0.05*corr_pred   
            optimizer.step()

            self.log('train_acc', 100*correct_t)
            self.log('train_loss', loss.detach().cpu())
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    if weight.shape[1] > 1:
                        weight = torch.mean(weight)
                    name = self.phenotype_columns[i]
                    self.log(f'train_{name}', weight.detach().cpu())      
        elif self.task == 'regression':
            abs_error = abs(train_pred.squeeze_() - train_lab.squeeze_()).mean().item()
            #GRAPH LOSS
            if logprobs is not None: 
                mae = abs(train_pred.squeeze_() - train_lab.squeeze_()).detach()
                if self.avg_mae is None:
                    self.avg_mae = torch.ones_like(mae)*self.graph_loss_mae
                point_w = (mae - self.avg_mae)
                graph_loss = (point_w * logprobs[:,mask.to(torch.bool)].exp().mean([-1,-2]))
                graph_loss = graph_loss.mean()
                graph_loss.backward()
                self.log('train_point_w', point_w.mean().detach().cpu())
                self.log('train_graph_loss', graph_loss.detach().cpu())
                self.avg_mae = self.avg_mae.to(mae.device)*0.95 +  0.05*mae
            optimizer.step()
            self.log('train_abs_error', abs_error)
            self.log('train_loss', loss.detach().cpu())
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    if weight.shape[1] > 1:
                        weight = torch.mean(weight)
                    name = self.phenotype_columns[i]
                    self.log(f'train_{name}', weight.detach().cpu())    
        else:
            raise ValueError('Task should be either regression or classification.')
        
    def validation_step(self, val_batch, batch_idx):
        X, y, mask, phenotypes, edges = val_batch
        edges = edges[0]
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]

        pred, att_weights, logprobs = self(X, edges, phenotypes)
        if self.task == 'classification':
            pred=pred.softmax(-1)
            for i in range(1,self.test_eval):
                pred_, att_weights, logprobs = self(X, edges, phenotypes)
                pred+=pred_.softmax(-1)
            test_pred = pred[:,mask.to(torch.bool),:]
            test_lab = y[:,mask.to(torch.bool),:]
        elif self.task == 'regression':
            pred = pred.squeeze_(-1)
            for i in range(1,self.test_eval):
                pred_, att_weights, logprobs = self(X, edges, phenotypes)
                pred+=pred_.squeeze_(-1)
            pred = pred / self.test_eval
            test_pred = pred[:,mask.to(torch.bool)]
            test_lab = y[:,mask.to(torch.bool)]
        else:
            raise ValueError('Task should be classification or regression.')

        if self.criterion == 'torch.nn.functional.cross_entropy()':
            loss = torch.nn.functional.cross_entropy(test_pred.view(-1,test_pred.shape[-1]),test_lab.argmax(-1).flatten())
        elif self.criterion == 'torch.nn.functional.binary_cross_entropy_with_logits()':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        elif (self.criterion == 'torch.nn.HuberLoss()'):
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())
        else: 
            raise ValueError('Choose appropriate loss function.')
        
        if self.task == 'classification':
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            
            accuracy = self.accuracy(test_pred.argmax(-1), test_lab.argmax(-1))
            auc = self.auc(test_pred.squeeze_().softmax(-1), test_lab.squeeze_().argmax(-1))
            f1score = self.f1score(test_pred.squeeze_().argmax(-1), test_lab.squeeze_().argmax(-1))

            self.log('val_accuracy', accuracy)
            self.log('val_AUC', auc)
            self.log('val_f1score', f1score)
            
            self.log('val_loss', loss.detach())
            self.log('val_acc', 100*correct_t)
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    name = self.phenotype_columns[i]
                    self.log(f'val_{name}', weight.detach().cpu())
        elif self.task == 'regression':
            abs_error = abs(test_pred.squeeze_() - test_lab.squeeze_()).mean().item()

            mean_absolute_error = self.mean_absolute_error(test_pred.squeeze_(),test_lab.squeeze_())
            rscore = self.rscore(test_pred.squeeze_(),test_lab.squeeze_())

            self.log('val_mean_absolute_error', mean_absolute_error)
            self.log('val_rscore', rscore)

            self.log('val_loss', loss)
            self.log('val_abs_error', abs_error)
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    name = self.phenotype_columns[i]
                    self.log(f'val_{name}', weight.detach().cpu())
        else:
            raise ValueError('Task should be either regression or classification.')

    def test_step(self, test_batch, batch_idx):
        X, y, mask, phenotypes, edges = test_batch
        edges = edges[0]
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]

        pred, att_weights, logprobs = self(X, edges, phenotypes)
        if self.task == 'classification':
            pred=pred.softmax(-1)
            for i in range(1,self.test_eval):
                pred_, att_weights, logprobs = self(X, edges, phenotypes)
                pred+=pred_.softmax(-1)
            test_pred = pred[:,mask.to(torch.bool),:]
            test_lab = y[:,mask.to(torch.bool),:]
        elif self.task == 'regression':
            pred = pred.squeeze_(-1) 
            for i in range(1,self.test_eval):
                pred_, att_weights, logprobs = self(X, edges, phenotypes)
                pred+=pred_.squeeze_(-1)
            pred = pred / self.test_eval
            test_pred = pred[:,mask.to(torch.bool)]
            test_lab = y[:,mask.to(torch.bool)]
        else:
            raise ValueError('Task should be classification or regression.')

        if self.criterion == 'torch.nn.functional.cross_entropy()':
            loss = torch.nn.functional.cross_entropy(test_pred.view(-1,test_pred.shape[-1]),test_lab.argmax(-1).flatten())
        elif self.criterion == 'torch.nn.functional.binary_cross_entropy_with_logits()':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        elif (self.criterion == 'torch.nn.HuberLoss()'): 
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())
        else: 
            raise ValueError('Choose appropriate loss function.')

        if self.task == 'classification':
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()

            accuracy = self.accuracy(test_pred.argmax(-1), test_lab.argmax(-1))
            auc = self.auc(test_pred.squeeze_().softmax(-1), test_lab.squeeze_().argmax(-1))
            f1score = self.f1score(test_pred.squeeze_().argmax(-1), test_lab.squeeze_().argmax(-1))

            self.log('test_accuracy', accuracy)
            self.log('test_AUC', auc)
            self.log('test_f1score', f1score)
            
            self.log('test_loss', loss.detach().cpu())
            self.log('test_acc', 100*correct_t)
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    name = self.phenotype_columns[i]
                    self.log(f'test_{name}', weight.detach().cpu())
        elif self.task == 'regression':
            abs_error = abs(test_pred.squeeze_() - test_lab.squeeze_()).mean().item()

            mean_absolute_error = self.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            rscore = self.rscore(test_pred.squeeze_(),test_lab.squeeze_())

            self.log('test_mean_absolute_error', mean_absolute_error)
            self.log('test_rscore', rscore)

            self.log('test_loss', loss)
            self.log('test_abs_error', abs_error)
            if att_weights is not None:
                for i, weight in enumerate(torch.transpose(att_weights, 0, 1)):
                    name = self.phenotype_columns[i]
                    self.log(f'test_{name}', weight.detach().cpu())
        else:
            raise ValueError('Task should be either regression or classification.')