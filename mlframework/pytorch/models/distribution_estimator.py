from tqdm.notebook import tqdm
import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from . import BaseModel
from ..layers import EmbeddingLayer, MLP_Layer


class distribution_estimator(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id='distribution_estimator',
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 embedding_regularizer=None,
                 net_regularizer=None,

                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 patience=2,
                 every_x_epochs=1,
                 reduce_lr_on_plateau=True,
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",

                 **kwargs):
        super(distribution_estimator, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 2)

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)

        out = F.relu(self.linear1(flat_feature_emb))
        out = F.relu(self.linear2(out))
        y_pred = self.linear3(out)

        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
    
    # redefining this function because metrics caluclations automatically reshape the outputs
    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy())
                y_true.extend(batch_data[1].data.cpu().numpy())
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs = self.evaluate_metrics(y_true, y_pred, self._validation_metrics)
            return val_logs
