import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

"""
Population-graph construction. 
Takes the imaging and non-imaging data and creates the initial population graph that will be used in the network.
"""

class PopulationGraphUKBB:
    def __init__(self, data_dir, filename_train, filename_val, filename_test, phenotype_columns, columns_kept, num_node_features, task, num_classes, k, edges):
        self.data_dir = data_dir
        self.filename_train = filename_train
        self.filename_val = filename_val
        self.filename_test = filename_test
        self.phenotype_columns = phenotype_columns
        self.columns_kept = columns_kept
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.k = k
        self.edges = edges
        self.task = task

    def load_data(self):
        """
        Loads the dataframes for the train, val, and test, and returns 1 dataframe for all.
        """

        # Read csvs for tran, val, test
        data_df_train = pd.read_csv(self.data_dir + self.filename_train)
        data_df_val = pd.read_csv(self.data_dir+self.filename_val)
        data_df_test = pd.read_csv(self.data_dir+self.filename_test)
        
        # Give labels for classification 
        if self.task == 'classification':        
                frames = [data_df_train, data_df_val, data_df_test]
                df = pd.concat(frames)

                labels = list(range(0,self.num_classes))
                df['Age'] = pd.qcut(df['Age'], q=self.num_classes, labels=labels).astype('int') #Balanced classes
                # df['Age'] = pd.cut(df['Age'], bins=self.num_classes, labels=labels).astype('int') #Not balanced classes
                
                a = data_df_train.shape[0]
                b = data_df_val.shape[0]

                data_df_train = df.iloc[:a, :]
                data_df_val = df.iloc[a:a+b, :]
                data_df_test = df.iloc[a+b:, :]
        
        a = data_df_train.shape[0] 
        b = data_df_train.shape[0]+data_df_val.shape[0] 
        num_nodes = b + data_df_test.shape[0] 

        train_idx = np.arange(0, a, dtype=int)
        val_idx = np.arange(a, b, dtype=int)
        test_idx = np.arange(b, num_nodes, dtype=int)
        frames = [data_df_train, data_df_val, data_df_test] 

        data_df = pd.concat(frames, ignore_index=True)

        return data_df, train_idx, val_idx, test_idx, num_nodes

    def get_phenotypes(self, data_df):
        """
        Takes the dataframe for the train, val, and test, and returns 1 dataframe with only the phenotypes.
        """
        phenotypes_df = data_df[self.phenotype_columns]      

        return phenotypes_df  

    def get_features_demographics(self, phenotypes_df):
        """
        Returns the phenotypes of every node, meaning for every subject. 
        The node features are defined by the non-imaging information
        """
        phenotypes = phenotypes_df.to_numpy()
        phenotypes = torch.from_numpy(phenotypes).float()
        return phenotypes

    def get_node_features(self, data_df):
        """
        Returns the features of every node, meaning for every subject.
        """
        df_node_features = data_df.iloc[:, 2:]
        node_features = df_node_features.to_numpy()
        node_features = torch.from_numpy(node_features).float()
        return node_features

    def get_subject_masks(self, train_index, validate_index, test_index):
        """Returns the boolean masks for the arrays of integer indices.

        inputs:
        train_index: indices of subjects in the train set.
        validate_index: indices of subjects in the validation set.
        test_index: indices of subjects in the test set.

        returns:
        a tuple of boolean masks corresponding to the train/validate/test set indices.
        """

        num_subjects = len(train_index) + len(validate_index) + len(test_index)

        train_mask = np.zeros(num_subjects, dtype=bool)
        train_mask[train_index] = True
        train_mask = torch.from_numpy(train_mask)

        validate_mask = np.zeros(num_subjects, dtype=bool)
        validate_mask[validate_index] = True
        validate_mask = torch.from_numpy(validate_mask)

        test_mask = np.zeros(num_subjects, dtype=bool)
        test_mask[test_index] = True
        test_mask = torch.from_numpy(test_mask)

        return train_mask, validate_mask, test_mask

    def get_labels(self, data_df):
        """
        Returns the labels for every node, in our case, age.

        """
        if self.task == 'regression':
            labels = data_df['Age'].values   
            labels = torch.from_numpy(labels).float()
        elif self.task == 'classification':
            labels = data_df['Age'].values 
            print(np.unique(labels, return_counts=True))
            labels = torch.from_numpy(labels)
        else:
            raise ValueError('Task should be either regression or classification.')
        return labels
                        
    def get_edges_using_KNNgraph(self, dataset, k):
        """
        Extracts edge index based on the cosine similarity of the node features.

        Inputs:
        dataset: the population graph (without edge_index).
        k: number of edges that will be kept for every node.

        Returns: 
        dataset: graph dataset with the acquired edges.
        """
        
        if self.edges == 'phenotypes':
            # Edges extracted based on the similarity of the selected phenotypes (imaging+non imaging)
            dataset.pos = dataset.phenotypes   
        elif self.edges == 'imaging':
            # Edges extracted based on the similarity of the node features
            dataset.pos = dataset.x  
        else:
            raise ValueError('Choose appropriate edge connection.')

        dataset.cuda()
        dataset = KNNGraph(k=k, force_undirected=True)(dataset)
        dataset.to('cpu')
        dataset = Data(x = dataset.x, y = dataset.y, phenotypes = dataset.phenotypes, train_mask=dataset.train_mask, 
                        val_mask= dataset.val_mask, test_mask=dataset.test_mask, edge_index=dataset.edge_index, 
                        num_nodes=dataset.num_nodes)
        return dataset
    
    def get_population_graph(self):
        """
        Creates the population graph.
        """
        # Load data
        data_df, train_idx, val_idx, test_idx, num_nodes = self.load_data()

        # Take phenotypes and node_features dataframes
        phenotypes_df = self.get_phenotypes(data_df)
        phenotypes = self.get_features_demographics(phenotypes_df)
        node_features = self.get_node_features(data_df) 

        # Mask val & test subjects
        train_mask, val_mask, test_mask = self.get_subject_masks(train_idx, val_idx, test_idx)
        # Get the labels
        labels = self.get_labels(data_df) 

        if  self.task == 'classification':
            labels= one_hot_embedding(labels,abs(self.num_classes)) 

        population_graph = Data(x = node_features, y= labels, phenotypes= phenotypes, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes, k=self.k)        
        # Get edges using existing pyg KNNGraph class
        population_graph = self.get_edges_using_KNNgraph(population_graph, k=self.k)
        return population_graph

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class UKBBageDataset(torch.utils.data.Dataset):
    def __init__(self, graph, split='train', samples_per_epoch=100, device='cpu', num_classes=2) -> None:
        dataset = graph
        self.n_features = dataset.num_node_features
        self.num_classes = abs(num_classes)
        self.X = dataset.x.float().to(device)
        self.y = dataset.y.float().to(device)
        self.y = dataset.y.float().to(device)
        self.phenotypes = dataset.phenotypes.float().to(device)
        self.edge_index = dataset.edge_index.to(device)

        if split=='train':
            self.mask = dataset.train_mask.to(device)
        if split=='val':
            self.mask = dataset.val_mask.to(device)
        if split=='test':
            self.mask = dataset.test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.phenotypes, self.edge_index