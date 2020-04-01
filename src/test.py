import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import sys

import numpy as np
import time
import sys
import pandas as pd
import os


# class to define a Graph Convolutional Layer

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation  = 'relu', skip = False, skip_in_features = None):
        super(GraphConv, self).__init__()
        self.W = torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        
        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim = 1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")
            
        self.skip = skip
        if self.skip:
            if skip_in_features == None:
                raise ValueError("pass input feature size of the skip connection")
            self.W_skip = torch.nn.Parameter(torch.DoubleTensor(skip_in_features, out_features)) 
            nn.init.xavier_uniform_(self.W)
        
    def forward(self, A, H_in, H_skip_in = None):
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in
        A_ = torch.add(self.A, torch.eye(self.A.shape[0]).double())
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix, 
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values 
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n
        
        H_out = torch.mm(torch.mm(A_norm, H_in), self.W)
        # shape of H_out will be n x out_features
        
        if self.skip:
            H_skip_out = torch.mm(H_skip_in, self.W_skip)
            H_out = torch.add(H_out, H_skip_out)
        
        if self.set_act:
            H_out = self.activation(H_out)
            
        return H_out
        

# class for 2 layer Graph Convolutional Network

class GCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, skip = False):
        super(GCN_2layer, self).__init__()
        self.skip = skip
        
        self.gcl1 = GraphConv(in_features, hidden_features)
        
        if self.skip:
            self.gcl_skip = GraphConv(hidden_features, out_features, activation = 'softmax', skip = self.skip,
                                  skip_in_features = in_features)
        else:
            self.gcl2 = GraphConv(hidden_features, out_features, activation = 'softmax')
        
    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)
            
        return out

def load_data(start_ts, end_ts):
	classes_csv = 'data/elliptic/elliptic_txs_classes.csv'
	edgelist_csv = 'data/elliptic/elliptic_txs_edgelist.csv'
	features_csv = 'data/elliptic/elliptic_txs_features.csv'

	classes = pd.read_csv(classes_csv, index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
	edgelist = pd.read_csv(edgelist_csv, index_col = 'txId1') # directed edges between transactions
	features = pd.read_csv(features_csv, header = None, index_col = 0) # features of the transactions
	
	num_features = features.shape[1]
	num_tx = features.shape[0]	
	total_tx = list(classes.index)

	# select only the transactions which are labelled
	labelled_classes = classes[classes['class'] != 'unknown']
	labelled_tx = list(labelled_classes.index)

	# to calculate a list of adjacency matrices for the different timesteps

	adj_mats = []
	features_labelled_ts = []
	classes_ts = []
	num_ts = 49 # number of timestamps from the paper

	for ts in range(start_ts, end_ts):
	    features_ts = features[features[1] == ts+1]
	    tx_ts = list(features_ts.index)
	    
	    labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]
	    
	    # adjacency matrix for all the transactions
	    # we will only fill in the transactions of this timestep which have labels and can be used for training
	    adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index = total_tx, columns = total_tx)
	    
	    edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
	    for i in range(edgelist_labelled_ts.shape[0]):
	        adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1
	    
	    adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
	    features_l_ts = features.loc[labelled_tx_ts]
	    
	    adj_mats.append(adj_mat_ts)
	    features_labelled_ts.append(features_l_ts)
	    classes_ts.append(classes.loc[labelled_tx_ts])

	return adj_mats, features_labelled_ts, classes_ts

	
num_features = 166
num_classes = 2
num_ts = 49
epochs = 50
lr = 0.001
max_train_ts = 34
train_ts = np.arange(max_train_ts)

adj_mats, features_labelled_ts, classes_ts = load_data( 0, max_train_ts)

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.long))

gcn = GCN_2layer(num_features, 32, num_classes)
train_loss = nn.CrossEntropyLoss(weight = torch.DoubleTensor([0.7, 0.3]))
optimizer = torch.optim.Adam(gcn.parameters(), lr = lr)

# Training

for ts in train_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    for ep in range(epochs):
        t_start = time.time()
        
        gcn.train()
        optimizer.zero_grad()
        out = gcn(A, X)

        loss = train_loss(out, L)
        train_pred = out.max(1)[1].type_as(L)
        acc = (train_pred.eq(L).double().sum())/L.shape[0]

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r Epoch %d/%d Timestamp %d/%d training loss: %f training accuracy: %f Time: %s"
                         %(ep, epochs, ts, max_train_ts, loss, acc, time.time() - t_start)
                        )

# torch.save(gcn.state_dict(), str(os.path.join(MODEL_DIR, "gcn_weights.pth")))
