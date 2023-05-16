from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error
import numpy as np
from torch import nn
from torch.functional import F
import torch
import logging

def evaluate_metrics(y_true, y_pred, metrics, **kwargs):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            #print(y_pred[:20], y_true[:20])
            y_pred = np.around(y_pred)
            result[metric] = accuracy_score(y_true, y_pred)
        elif metric == "MSE":
            result[metric] = mean_squared_error(y_true, y_pred)
        elif metric == 'CE':
            result[metric] = cross_entropy_metric(y_true, y_pred)
        elif metric == "proportion":
            result[metric] = proportion_metric(y_true, y_pred)
        elif 'logprob' in metric:
            distribution = metric.split('_')[1]
            result[metric] = dist_logprob(y_true, y_pred, distribution)
        else:
            assert "group_index" in kwargs, "group_index is required for GAUC"
            group_index = kwargs["group_index"]
            if metric == "GAUC":
                pass
            elif metric == "NDCG":
                pass
            elif metric == "MRR":
                pass
            elif metric == "HitRate":
                pass
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result

def dist_logprob(target, output, distribution):
    output = torch.tensor(output)
    output = output.view(output.size()[0]//2,2)
    target = torch.tensor(target)
    chosen_dist = getattr(torch.distributions, distribution.split('.')[0])
    if '.' in distribution:
        chosen_dist = getattr(torch.distributions, distribution.split('.')[1])
    
    output = F.softplus(output)
    outputs = [ output[:,i] for i in range(output.size()[1])]
    dists = chosen_dist(*outputs)
    log_probs = dists.log_prob(target)
    loss = -torch.mean(log_probs)
    return loss

def proportion_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)/y_true)

def cross_entropy_metric(y_true, y_pred):
    batch_size = len(y_true)
    y_pred = torch.tensor(y_pred).reshape((batch_size,-1))
    y_true = torch.tensor(y_true).long()
    return nn.CrossEntropyLoss()(y_pred, y_true).item()