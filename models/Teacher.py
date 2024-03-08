from pickle import NONE
import sys
sys.path.append('./')

from configs.DiCoSConfig import DiCoSConfig
from models.Layers import Encoder
from transformers import AlbertModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class Teacher(nn.Module):
    def __init__(self, config:DiCoSConfig) -> None:
        super(Teacher, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(0.5)
        self.fc_seq = nn.Linear(config.hidden_size, 2)
        self.fc_dia = nn.Linear(config.hidden_size, 2)

    def forward(self, sentencesTokens:torch.Tensor, attentionMask:torch.Tensor, segmentEmbedding:torch.Tensor=None, output_attentions=None):
        '''
            给定句子,返回两个分类的分类情况

            sentencesTokens: [batchSize, seqLen]
            attentionMask: [batchSize, seqLen]
            segmentEmbedding: [batchSize, seqLen]


            output: [batchSize, 2], [batchSize, 2], [layers, batchSize, num_head, seq_len, seq_len]

        '''

        embedding = self.encoder(sentencesTokens, attentionMask, segmentEmbedding, output_attentions=output_attentions)
        
        pooler_output = embedding['pooler_output']

        pooler_output = self.dropout(pooler_output)

        sequential = self.fc_seq(pooler_output)
        contextual = self.fc_dia(pooler_output)
        if output_attentions == None:
            return sequential, contextual
        else:
            attentionMap = embedding['attentions']
            return sequential, contextual, attentionMap
