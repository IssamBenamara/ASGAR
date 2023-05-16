import torch
from torch import nn
from . import BaseModel
from ..layers import EmbeddingLayer, MLP_Layer


class DNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DNN",
                 gpu=0,
                 task="binary_classification",
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 stacked_dnn_hidden_units=[],
                 parallel_dnn_hidden_units=[],
                 base_dnn_hidden=[128,64,32],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 final_layer=2,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        gpu = 0
        super(DNN, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        input_dim = feature_map.num_fields * embedding_dim

        #HEREE
        self.base_dnn = MLP_Layer(input_dim=input_dim,
                                         output_dim=final_layer, # output hidden layer
                                         hidden_units=base_dnn_hidden,
                                         hidden_activations=dnn_activations,
                                         output_activation=None,
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm,
                                         use_bias=True)

        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        #flat_feature_emb = feature_emb.view((len(inputs), -1))
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        
        y_pred = self.base_dnn(flat_feature_emb)

        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict