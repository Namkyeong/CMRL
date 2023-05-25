import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import numpy as np

from torch_geometric.nn import Set2Set, global_mean_pool

from embedder import embedder
from layers import GCN
from utils import create_batch_mask, create_interv_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

import random
import time

class CMRL_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, fold)

        self.num_classes = 2

        self.model = CMRL(device = self.device, num_step_message_passing = self.args.num_layers, num_classes = self.num_classes, intervention = self.args.intervention, conditional = self.args.conditional).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True)
        
    def train(self):
        
        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='mean')
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            self.loss_pos = 0
            self.loss_neg = 0
            self.loss_inv = 0
            self.importance = 0

            start = time.time()

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)

                pred = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_function_BCE(pred, samples[2].reshape(-1, 1).to(self.device).float()).mean() # Supervised Loss

                # Causal Loss
                pos, neg, rand = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], causal = True)
                
                loss_pos = loss_function_BCE(pos, samples[2].reshape(-1, 1).to(self.device).float()) # Positive Loss
                
                random_label = torch.ones_like(neg, dtype=torch.float).to(self.device) / self.num_classes
                loss_neg = F.kl_div(neg, random_label, reduction = 'batchmean')
                
                loss = loss + loss_pos + self.args.lam1 * loss_neg
                
                # Intervention Loss
                if self.args.intervention:
                    loss_inv = self.args.lam2 * loss_function_BCE(rand, samples[2].reshape(-1, 1).to(self.device).float())
                    loss += loss_inv
                    self.loss_inv += loss_inv
                
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                
                self.loss_pos += loss_pos
                self.loss_neg += loss_neg
                self.importance += torch.sigmoid(self.model.importance).mean()
            
            self.epoch_time = time.time() - start

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_roc_score)

            # Early stopping
            if len(self.best_val_rocs) > int(self.args.es / self.args.eval_freq):
                if self.best_val_rocs[-1] == self.best_val_rocs[-int(self.args.es / self.args.eval_freq)]:
                    if self.best_val_accs[-1] == self.best_val_accs[-int(self.args.es / self.args.eval_freq)]:
                        self.is_early_stop = True
                        break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_roc, self.best_test_ap, self.best_test_f1, self.best_test_acc


class CMRL(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                device,
                node_input_dim=10,
                node_hidden_dim=128,
                num_step_message_passing=3,
                num_step_set2_set=2,
                num_layer_set2set=1,
                num_classes = 2,
                intervention = False,
                conditional = False
                ):
        super(CMRL, self).__init__()

        self.device = device
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.intervention = intervention
        self.conditional = conditional

        self.gather = GCN(node_input_dim = self.node_input_dim, 
                        node_hidden_dim = self.node_hidden_dim, 
                        out_dim = self.node_hidden_dim,
                        num_step_message_passing = self.num_step_message_passing)
        
        self.compressor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )

        self.predictor = nn.Linear(8 * self.node_hidden_dim, 1)

        self.neg_predictor = nn.Linear(2 * self.node_hidden_dim, num_classes)

        self.rand_predictor = nn.Linear(10 * self.node_hidden_dim, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_pos_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

        self.init_model()
    
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    

    def compress(self, solute_features):
        
        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    

    def interaction(self, solute_features, solvent_features):

        # Do normalization
        normalized_solute_features = F.normalize(solute_features, dim = 1)
        normalized_solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(self.solute_len.t(), self.solvent_len)

        interaction_map = torch.mm(normalized_solute_features, normalized_solvent_features.t())
        interaction_map = interaction_map * len_map.to_dense()

        solvent_prime = torch.mm(interaction_map.t(), normalized_solute_features)
        solute_prime = torch.mm(interaction_map, normalized_solvent_features)

        # Prediction phase
        solute_features = torch.cat((normalized_solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((normalized_solvent_features, solvent_prime), dim=1)

        return solute_features, solvent_features
    

    def forward(self, data, causal = False, test = False):

        solute = data[0]
        solvent = data[1]
        self.solute_len = data[2]
        self.solvent_len = data[3]
        # node embeddings after interaction phase
        _solute_features = self.gather(solute)
        _solvent_features = self.gather(solvent)

        solute_features, solvent_features = self.interaction(_solute_features, _solvent_features)

        if (test == True) or (causal == False):

            solute_features_s2s = self.set2set_pos_solute(solute_features, solute.batch)
            solvent_features_s2s = self.set2set_solvent(solvent_features, solvent.batch)

            solute_solvent = torch.cat((solute_features_s2s, solvent_features_s2s), 1)
            predictions = self.predictor(solute_solvent)

            if test: 
                return torch.sigmoid(predictions)
            else:
                return predictions

        else:

            lambda_pos, self.importance = self.compress(solute_features)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos

            # Inject Noise
            static_solute_feature = solute_features.clone().detach()

            node_feature_mean = scatter_mean(static_solute_feature, solute.batch, dim = 0)[solute.batch]
            node_feature_std = scatter_std(static_solute_feature, solute.batch, dim = 0)[solute.batch]

            noisy_node_feature_mean = lambda_pos * solute_features + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            pos_solute = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            neg_solute = lambda_neg * solute_features

            pos_solute_s2s = self.set2set_pos_solute(pos_solute, solute.batch)
            neg_solute = global_mean_pool(neg_solute, solute.batch)
            solvent_features_s2s = self.set2set_solvent(solvent_features, solvent.batch)

            pos_solute_solvent = torch.cat((pos_solute_s2s, solvent_features_s2s), 1)

            pos_predictions = self.predictor(pos_solute_solvent)
            neg_predictions = self.neg_predictor(neg_solute)

            if self.intervention == True:

                num = pos_solute_s2s.shape[0]
                l = [i for i in range(num)]
                random.shuffle(l)
                random_idx = torch.tensor(l)

                if self.conditional == True:

                    # Create Intervention Interaction Map
                    rand_batch = random_idx[solute.batch]
                    self.solute_len = create_interv_mask(rand_batch).to(self.device)
                    solute_features, _ = self.interaction(_solute_features, _solvent_features)

                    lambda_pos, self.importance = self.compress(solute_features)
                    lambda_pos = lambda_pos.reshape(-1, 1)
                    lambda_neg = 1 - lambda_pos

                    rand_solute = lambda_neg * solute_features

                    rand_solute = global_mean_pool(rand_solute, solute.batch)
                    rand_solute_solvent = torch.cat((pos_solute_s2s, solvent_features_s2s, rand_solute[random_idx]), 1)
                    random_predictions = self.rand_predictor(rand_solute_solvent)
                
                else:
                    rand_solute_solvent = torch.cat((pos_solute_s2s, solvent_features_s2s, neg_solute[random_idx]), 1)
                    random_predictions = self.rand_predictor(rand_solute_solvent)

                return pos_predictions, F.log_softmax(neg_predictions, dim = -1), random_predictions
        
            return pos_predictions, F.log_softmax(neg_predictions, dim = -1), None