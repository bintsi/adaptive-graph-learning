import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

from argparse import ArgumentParser
from argparse import Namespace

from graph_constuction import PopulationGraphUKBB, UKBBageDataset
from model import GraphLearningModel

run_params = {
    "gpus":1,
    "log_every_n_steps": 100,
    "max_epochs": 150,
    "progress_bar_regresh_rate":1,
    "check_val_every_n_epoch":1,
    
    "conv_layers": [[68,512]], 
    "dgm_layers": [[35,35], []],
    "fc_layers": [512,128,1], 
    "pre_fc": None,
    
    "gfun":'gcn',
    "ffun": 'phenotypes',
    "k": 5,
    "pooling": 'add',
    "distance": 'euclidean',
    
    "dropout": 0,
    "lr": 0.001, 
    "test_eval": 10, 
    
    "num_node_features": 68,
    "num_classes": 1,
    "task": 'regression',
    
    "graph_loss_mae": 6,
    "edges": 'phenotypes',

    "phenotype_columns": ['Sex', 'Height', 'Body mass index (BMI)', 'Systolic blood pressure', 'Diastolic blood pressure', 'College education', 'Smoking status',
                        'Alcohol intake frequency', 'Stroke', 'Diabetes', 'Walking per week', 'Vigorous per week', 'Fluid intelligence', 'Tower rearranging: number of puzzles correct',
                        'Trail making task: duration to complete numeric path trail 1', 'Trail making task: duration to complete alphanumeric path trail 2', 'Matrix pattern completion: number of puzzles correctly solved',
                        'Matrix pattern completion: duration spent answering each puzzle', 
                        'Volume of grey matter (normalised for head size)', 'Volume of brain stem + 4th ventricle', 'Volume of grey matter in Putamen (right)', 'Volume of thalamus (right)',
                        'Volume of putamen (left)', 'Volume of grey matter in Thalamus (right)', 'Total volume of white matter hyperintensities (from T1 and T2_FLAIR images)', 'Weighted-mean ICVF in tract forceps minor',
                        'Weighted-mean L1 in tract anterior thalamic radiation (right)', 'Mean FA in cerebral peduncle on FA skeleton (left)',
                        'Mean FA in superior cerebellar peduncle on FA skeleton (left)', 'Mean L1 in middle cerebellar peduncle on FA skeleton', 'Mean ICVF in body of corpus callosum on FA skeleton', 
                        'Weighted-mean L3 in tract uncinate fasciculus (left)','Mean ISOVF in fornix on FA skeleton', 
                        'Weighted-mean L1 in tract parahippocampal part of cingulum (left)', 'Mean L2 in fornix cres+stria terminalis on FA skeleton (left)'],
}

def costruct_graph(run_params):
    """
    Extract an initial population graph to be used as input to the model.
    """

    # We have selected imaging features + non-imaging features 
    # that are found relevant to brain-age from J.Cole's paper
    # https://pubmed.ncbi.nlm.nih.gov/32380363/
    data_dir = 'data/'
    filename_train = 'train.csv' 
    filename_val = 'val.csv' 
    filename_test = 'test.csv'

    # Keep only the imaging features as node features 
    node_columns = [0, 1, 22, 90]
    num_node_features = node_columns[3] - node_columns[2]

    task = run_params.task
    num_classes = run_params.num_classes
    k = run_params.k
    edges = run_params.edges

    #Phenotypes chosen for the extraction of the edges
    phenotype_columns = run_params.phenotype_columns

    population_graph = PopulationGraphUKBB(data_dir, filename_train, filename_val, filename_test, phenotype_columns, node_columns, 
                            num_node_features, task, num_classes, k, edges)
    population_graph = population_graph.get_population_graph()
    return population_graph

def main(run_params, graph): 
    train_data = None
    test_data = None

    # Load data
    train_data = UKBBageDataset(graph=graph, split='train', device='cuda', num_classes = run_params.num_classes)
    val_data = UKBBageDataset(graph=graph, split='val', samples_per_epoch=1, num_classes = run_params.num_classes)
    test_data = UKBBageDataset(graph=graph, split='test', samples_per_epoch=1, num_classes = run_params.num_classes)
    
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0) 
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    if train_data is None:
        raise Exception("Dataset %s not supported" % run_params.dataset)

    #configure input feature sizes
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.phenotypes.shape[1]
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]= train_data.n_features
    
    if run_params.fc_layers is not None:
        run_params.fc_layers[-1] = train_data.num_classes

    model = GraphLearningModel(run_params)
    print(model)

    if run_params.task == 'regression':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min'
        )
    elif run_params.task == 'classification':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_acc',
            mode='max')
    else:
        raise ValueError('Task should be either regression or classification.')   

    callbacks = [checkpoint_callback]
    if val_data==test_data:
        callbacks = None

    logger = TensorBoardLogger("logs/regression/")
    trainer = pl.Trainer.from_argparse_args(run_params,logger=logger,
                                            callbacks=callbacks)

    trainer.fit(model, datamodule=MyDataModule())

    # Evaluate results on validation and test set
    val_results = trainer.validate(ckpt_path=checkpoint_callback.best_model_path, dataloaders=val_loader)
    test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=test_loader)
    
    # Save results   
    path = '/'.join(checkpoint_callback.best_model_path.split("/")[:-1])
    with open(path + "/val_results.json", "w") as outfile:
        json.dump(val_results, outfile)
    with open(path + "/test_results.json", "w") as outfile:
        json.dump(test_results, outfile)   
    return val_results, test_results, path

if type(run_params) is not Namespace:
    run_params = Namespace(**run_params)

population_graph = costruct_graph(run_params)
val_results, test_results, best_model_checkpoint_path = main(run_params, population_graph)