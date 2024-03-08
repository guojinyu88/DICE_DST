import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Layers import SentencesSlotCrossEncoder
from configs.DiCoSConfig import DiCoSConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
import numpy as np
from typing import List

class SN_DH(nn.Module):
    def __init__(self, config:DiCoSConfig) -> None:
        '''
            计算 当前轮 slot 与 历史上各轮次之间的匹配程度 
        '''
        super(SN_DH, self).__init__()

        self.config = config
        self.crossRepresentation = SentencesSlotCrossEncoder(config) # 计算当前对话slot和历史中的对话的交叉表示

    def forward(self, slotsHiddenStates:torch.Tensor, history_sentencesHiddenStates:torch.Tensor, history_mask_sentences:torch.Tensor):
        '''
            并行化方式
            slotsHiddenStates: 每句话的每一个的[slot]表示  [batchSize, slotType, hiddenSize]
            history_sentencesHiddenStates: Tensor [batchSize, max_historyNum, seqLen, hiddenSize]
            history_mask_sentences : Tensor [batchSize, max_historyNum, sqeLen] 句子的地方填写1, slot的地方填写0 padding 的位置填写0

            return:
            crossRelation: Tensor  [batchSize, max_historyNum, slotType, hiddenSize] 历史对话 和 每个slot 的关系
        '''
        # 对slotsHiddenStates进行复制，调整形状
        batch_size = slotsHiddenStates.shape[0] # 保证多卡情况下读取到正确的batchsize
        slotsHiddenStates = slotsHiddenStates.unsqueeze(1).repeat([1, history_sentencesHiddenStates.shape[1], 1, 1]) # [batchSize, max_historyNum, slotType, hiddenSize]
        slotsHiddenStates = slotsHiddenStates.reshape([-1, self.config.slotTypeNum, self.config.hidden_size]) # [batchSize*max_historyNum, slotType, hiddenSize]
        # 对history_sentencesHiddenStates，调整形状
        history_sentencesHiddenStates = history_sentencesHiddenStates.reshape([-1, self.config.pad_size, self.config.hidden_size]) # [batchSize*max_historyNum, seqLen, hiddenSize]
        # 对history_mask_sentences，调整形状
        history_mask_sentences = history_mask_sentences.reshape([-1, self.config.pad_size]) # [batchSize*max_historyNum, sqeLen]
        # 计算slot与每个历史对话的相似度
        crossRelation = self.crossRepresentation(history_sentencesHiddenStates, slotsHiddenStates, history_mask_sentences) # [batchSize*max_historyNum, slotType, hiddenSize]
        # 重建形状
        crossRelation = torch.reshape(crossRelation, [batch_size, -1, self.config.slotTypeNum, self.config.hidden_size]) # [batchSize, max_historyNum, slotType, hiddenSize]

        return crossRelation

    def forward_split(self, slotsHiddenStates:torch.Tensor, history_sentencesHiddenStates:torch.Tensor, history_mask_sentences:torch.Tensor):
        '''
            每个样本串行化方式
            slotsHiddenStates: 一句话的每一个的[slot]表示  [slotType, hiddenSize]
            history_sentencesHiddenStates: Tensor [historyNum, seqLen, hiddenSize]
            history_mask_sentences : Tensor [historyNum, sqeLen] 句子的地方填写1, slot的地方填写0

            return:
            crossRelation: Tensor  [historyNum, slotType, hiddenSize] 历史对话 和 每个slot 的关系
        '''
        
        slotsHiddenStates = slotsHiddenStates.unsqueeze(0).repeat([history_sentencesHiddenStates.shape[0], 1, 1]) # 对每个历史都使用相同的 slot表示
        crossRelation = self.crossRepresentation(history_sentencesHiddenStates, slotsHiddenStates, history_mask_sentences)

        return crossRelation

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config:DiCoSConfig):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = config
        self.linear_Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_V = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input:torch.Tensor, mask:torch.Tensor):
        '''
            input: [batchSize, seqLen, hiddenSize] 
            mask:[batchSize, seqLen] 1的地方可以分配attention  0 的地方不分配atteention

            return:  [batchSize, seqLen, hiddenSize] 
        '''
        input_Q = self.linear_Q(input) # [batchSize, seqLen, hiddenSize] 
        input_K = self.linear_Q(input) # [batchSize, seqLen, hiddenSize] 
        input_V = self.linear_Q(input) # [batchSize, seqLen, hiddenSize] 

        # 拆分头
        multihead_Q = torch.reshape(input_Q, [input.shape[0], input.shape[1], self.config.num_multiHead, self.config.hidden_size//self.config.num_multiHead]).transpose(1,2) # [batchSize, multihead, seqLen, hiddenSize/multihead]
        multihead_K = torch.reshape(input_K, [input.shape[0], input.shape[1], self.config.num_multiHead, self.config.hidden_size//self.config.num_multiHead]).transpose(1,2) # [batchSize, multihead, seqLen, hiddenSize/multihead]
        multihead_V = torch.reshape(input_V, [input.shape[0], input.shape[1], self.config.num_multiHead, self.config.hidden_size//self.config.num_multiHead]).transpose(1,2) # [batchSize, multihead, seqLen, hiddenSize/multihead]
        mask = mask.unsqueeze(1).repeat([1, input.shape[1], 1])
        mask = mask.unsqueeze(1).repeat([1, self.config.num_multiHead, 1, 1]) # [batchSize, multihead, seqLen, seqLen]

        # 计算attention score
        scores = torch.matmul(multihead_Q, multihead_K.transpose(-1, -2)) / np.sqrt(self.config.hidden_size//self.config.num_multiHead) # [batchSize, multihead, seqLen_Q, seqLen_K]
        scores = torch.masked_fill(scores, mask==0, 1e-9)
        attn = F.softmax(scores, dim=-1) # [batchSize, multihead, seqLen_Q, seqLen_K]

        # 多头分别取值
        output = torch.matmul(attn, multihead_V) # [batchSize, multihead, seqLen_Q, hiddenSize/multihead]

        # 拼接然后过线性层
        output = output.transpose(1,2).reshape(input.shape) # [batchSize, seqLen, hiddenSize] 
        output = self.fc(output) # [batchSize, seqLen, hiddenSize] 

        return output

class CT_DH(nn.Module):
    '''
        计算 当前轮 对话 与 历史上对话的匹配程度
    '''
    def __init__(self, config:DiCoSConfig) -> None:
        super(CT_DH, self).__init__()

        self.config = config
        self.MHSA = MultiHeadSelfAttention(config)

    def forward(self, currentCLSHiddenStates:torch.Tensor, history_CLSHiddenStates:torch.Tensor, history_padingMask:torch.Tensor):
        '''
            currentCLSHiddenStates: 每句话的CLS表示 [batchSize, hiddenSize]
            history_CLSHiddenStates: Tensor [batchSize, max_historyNum, hiddenSize]
            history_padingMask: Tensor [batchSize, max_historyNum]  1的地方为真实的句子, 可以分配attention

            return: 
            new_CLSHiddenStates: [batchSize, 1+max_historyNum, hiddenSize] 过了MHSA的CLS表示
            
            history_CLSHiddenStates: [batchSize, max_historyNum, hiddenSize] 每个历史轮次与当前轮次之间的相关性
        '''
        batch_size = currentCLSHiddenStates.shape[0] # 保证多卡情况下的动态batch
        # 将当前轮对话加入
        all_CLSHiddenStates = torch.cat([currentCLSHiddenStates.unsqueeze(1), history_CLSHiddenStates], dim=1) # [batchSize, 1+max_historyNum, hiddenSize]
        all_padingMask = torch.cat([torch.ones([batch_size, 1], dtype=torch.long, device=self.config.device), history_padingMask], dim=1) # [batchSize, 1+max_historyNum]
        # 输入MHSA 获取新的表示
        all_CLSHiddenStates = self.MHSA(all_CLSHiddenStates, all_padingMask) # [batchSize, 1+max_historyNum, hiddenSize]
        # 用于输入IMOR
        new_CLSHiddenStates = torch.clone(all_CLSHiddenStates)
        # 拆分出来当前轮和历史轮
        currentCLSHiddenStates = torch.unsqueeze(all_CLSHiddenStates[:,0,:], dim=1) # [batchSize, 1, hiddenSize]
        history_CLSHiddenStates = all_CLSHiddenStates[:,1:,:] #  [batchSize, max_historyNum, hiddenSize]
        # 作attention加权
        atten = torch.matmul(history_CLSHiddenStates, currentCLSHiddenStates.transpose(-1, -2)) / np.sqrt(self.config.hidden_size) # [batchSize, max_historyNum, 1]
        history_CLSHiddenStates = history_CLSHiddenStates + atten * currentCLSHiddenStates

        return new_CLSHiddenStates, history_CLSHiddenStates

    
    def forward_split(self, currentCLSHiddenStates:torch.Tensor, history_CLSHiddenStates:torch.Tensor):
        '''
            currentCLSHiddenStates: 一句话的CLS表示 [hiddenSize]
            history_CLSHiddenStates: Tensor [historyNum, hiddenSize]

            return: [historyNum, hiddenSize] 每个历史轮次与当前轮次之间的相关性
        '''
        all_CLSHiddenStates = torch.cat([history_CLSHiddenStates, currentCLSHiddenStates.unsqueeze(0)], dim=0).unsqueeze(0) # [1, historyNum+1, hiddenSize]
        attentionMask = torch.ones([1, history_CLSHiddenStates.shape[0]+1]) # [1, historyNum+1]

        all_CLSHiddenStates = self.MHSA(all_CLSHiddenStates, attentionMask) # [1, historyNum+1, hiddenSize]
        all_CLSHiddenStates = torch.unsqueeze(all_CLSHiddenStates) # [historyNum+1, hiddenSize]

        # 切分出来历史和当前
        history_CLSHiddenStates = all_CLSHiddenStates[:-1] # [historyNum, hiddenSize]
        currentCLSHiddenStates = all_CLSHiddenStates[-1] # [hiddenSize]
        
        atten = torch.matmul(history_CLSHiddenStates, currentCLSHiddenStates.unsqueeze(-1)) # [historyNum, 1]

        history_CLSHiddenStates = history_CLSHiddenStates + currentCLSHiddenStates.unsqueeze(0).expand_as(history_CLSHiddenStates) * atten # [historyNum, hiddenSize]

        return history_CLSHiddenStates

class IMOR(nn.Module):
    def __init__(self, config:DiCoSConfig):
        super(IMOR, self).__init__()
        self.config = config
        self.f_rs = nn.ModuleList()
        for i in range(self.config.num_relationType):
            self.f_rs.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        self.f_s = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.f_g = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size)

    def forward(self, slotsHiddenStates:torch.Tensor, CLSHiddenStates:torch.Tensor, updateSlot2current:torch.Tensor, updateSlot2allSlot:torch.Tensor, slot2lastUpdateTurn:torch.Tensor, slot_domain_connect:torch.Tensor):
        '''
            slotsHiddenStates : [batchSize, slotType, hiddenSize]
            CLSHiddenStates, [batchSize, 1+max_historyNum, hiddenSize] 经过了MHSA的
            updateSlot2current, [batchSize, slotType, 1+max_historyNum]
            updateSlot2allSlot, [batchSize, slotType, SlotType]
            slot2lastUpdateTurn, [batchSize, slotType, 1+max_historyNum]
            slot_domain_connect  [batchSize, slotType, SlotType]

            return: 

            slot_node, [turnNum, slotType, hiddenSize]
            dialogue_node [turnNum, turnNum, hiddenSize]
        '''
        for i in range(self.config.num_GNNLayer):
            # 计算当前状态的表示
            CLSHiddenStates_u = self.f_s(CLSHiddenStates) # [batchSize, 1+max_historyNum, hiddenSize] 
            slotsHiddenStates_u = self.f_s(slotsHiddenStates) # [batchSize, slotType, hiddenSize]
            # 计算每个节点充当邻居节点的时候的表示
            relation_CLSHiddenState_neighbour = [] # [relationType, batchSize, 1+max_historyNum, hiddenSize]
            relation_slotsHiddenStates_neighbour = [] # [relationType, batchSize, slotType, hiddenSize]
            for f_r in self.f_rs:
                relation_CLSHiddenState_neighbour.append(f_r(CLSHiddenStates))
                relation_slotsHiddenStates_neighbour.append(f_r(slotsHiddenStates))

            # updateSlot2current 关系
            num_neighbours = updateSlot2current.transpose(-1,-2).sum(dim=-1, keepdim=True).expand_as(CLSHiddenStates_u) # [batchSize, 1+max_historyNum, hiddenSize]
            CLSHiddenStates_u += torch.matmul(updateSlot2current.transpose(-1,-2), relation_slotsHiddenStates_neighbour[0]) / (num_neighbours+1e-5) #  [batchSize, 1+max_historyNum, slotType] * [batchSize, slotType, hiddenSize] TODO  除0了
            num_neighbours = updateSlot2current.sum(dim=-1,keepdim=True).expand_as(slotsHiddenStates_u) # [batchSize, slotType, hiddenSize]
            slotsHiddenStates_u += torch.matmul(updateSlot2current, relation_CLSHiddenState_neighbour[0]) / (num_neighbours+1e-5) # [batchSize, slotType, 1+max_historyNum] * [batchSize, 1+max_historyNum, hiddenSize]

            # updateSlot2allSlot 关系
            num_neighbours = updateSlot2allSlot.sum(dim=-1,keepdim=True).expand_as(slotsHiddenStates_u)
            slotsHiddenStates_u += torch.matmul(updateSlot2allSlot, relation_slotsHiddenStates_neighbour[1]) / (num_neighbours+1e-5)

            # slot2lastUpdateTurn 关系
            num_neighbours = slot2lastUpdateTurn.transpose(-1,-2).sum(dim=-1, keepdim=True).expand_as(CLSHiddenStates_u) # [batchSize, 1+max_historyNum, hiddenSize]
            CLSHiddenStates_u += torch.matmul(slot2lastUpdateTurn.transpose(-1,-2), relation_slotsHiddenStates_neighbour[2]) / (num_neighbours+1e-5) #  [batchSize, 1+max_historyNum, slotType] * [batchSize, slotType, hiddenSize]
            num_neighbours = slot2lastUpdateTurn.sum(dim=-1,keepdim=True).expand_as(slotsHiddenStates_u) # [batchSize, slotType, hiddenSize]
            slotsHiddenStates_u += torch.matmul(slot2lastUpdateTurn, relation_CLSHiddenState_neighbour[2]) / (num_neighbours+1e-5) # [batchSize, slotType, 1+max_historyNum] * [batchSize, 1+max_historyNum, hiddenSize]

            # slot_domain_connect 关系
            num_neighbours = slot_domain_connect.sum(dim=-1,keepdim=True).expand_as(slotsHiddenStates_u)
            slotsHiddenStates_u += torch.matmul(slot_domain_connect, relation_slotsHiddenStates_neighbour[3]) / (num_neighbours+1e-5)

            # 状态更新
            gate_slot = torch.sigmoid(self.f_g(torch.cat([slotsHiddenStates_u, slotsHiddenStates], dim=-1)))
            gate_CLS = torch.sigmoid(self.f_g(torch.cat([CLSHiddenStates_u, CLSHiddenStates], dim=-1)))

            slotsHiddenStates = (F.relu(slotsHiddenStates_u) * gate_slot) + (slotsHiddenStates * (1-gate_slot))
            CLSHiddenStates = (F.relu(CLSHiddenStates_u) * gate_CLS) + (CLSHiddenStates * (1-gate_CLS))

        # 丢弃当前轮直取历史对话出来
        history_CLSHiddenStates = CLSHiddenStates[:,1:,:] # [batchSize, max_historyNum, hiddenSize] 
        

        return history_CLSHiddenStates

class MultiPrespectiveSelector(nn.Module):
    def __init__(self, config:DiCoSConfig):
        super(MultiPrespectiveSelector, self).__init__()
        
        self.config = config
        self.SN_DH = SN_DH(config)
        self.CT_DH = CT_DH(config)
        self.IMOR = IMOR(config)

        # 融合 slot value 的表示
        self.merge_slot_node = nn.Linear(2*config.hidden_size, config.hidden_size)

        self.SN_DH_score = nn.Linear(config.hidden_size, 1)
        self.CT_DH_score = nn.Linear(config.hidden_size, 1)
        self.IMOR_score = nn.Linear(config.hidden_size, 1)

        self.hiddenState2score = nn.Linear(config.hidden_size, 1)
        


    def forward(self, currentEmbedding:BaseModelOutputWithPooling, slotTokenPosition:torch.Tensor, valueTokenPosition:torch.Tensor, 
                    historyEmbedding:BaseModelOutputWithPooling, history_mask_sentences:torch.Tensor, 
                    updateSlot2current:torch.Tensor, updateSlot2allSlot:torch.Tensor, slot2lastUpdateTurn:torch.Tensor, slot_domain_connect:torch.Tensor):
        '''
            并行化方式. 且已防止padding的对话的干扰
            currentEmbedding [batchSize, seqLen, hiddenSize]
            slotTokenPosition [batchSize, slotType] 标记 [slot] token 的位置
            valueTokenPosition [batchSize, slotType] 标记 [value] token 的位置
            historyEmbedding: [batchSize*max_historyNum, seqLen, hiddenSize] 
            history_mask_sentences : Tensor [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
            updateSlot2current, [batchSize, slotType, 1+max_historyNum] 更新槽与当前轮  需要保证padiing出来的history 不与任何点相连
            updateSlot2allSlot, [batchSize, slotType, SlotType] 本轮要更新的槽链接所有的槽
            slot2lastUpdateTurn, [batchSize, slotType, 1+max_historyNum] 槽与最近更新轮  需要保证padiing出来的history 不与任何点相连
            slot_domain_connect  [batchSize, slotType, SlotType] 同domain的槽相连
            

            return:
            crossRelation: Tensor  [batchSize, max_historyNum, slotType] 历史对话 和 每个slot 的关系 (padding出来的history已经处理过了)
        '''
        batch_size = slotTokenPosition.shape[0] # 保证多卡下的动态batchsize
        # 取出 special token 的表示
        slotsHiddenStates = currentEmbedding.last_hidden_state.gather(dim=1, index=slotTokenPosition.unsqueeze(-1).repeat([1,1,self.config.hidden_size]))
        valuesHiddenStates = currentEmbedding.last_hidden_state.gather(dim=1, index=valueTokenPosition.unsqueeze(-1).repeat([1,1,self.config.hidden_size]))

        # slot name 视角 
        history_sentencesHiddenStates = historyEmbedding.last_hidden_state.reshape([batch_size, -1, self.config.pad_size, self.config.hidden_size])
        slotName_ = self.SN_DH(slotsHiddenStates, history_sentencesHiddenStates, history_mask_sentences) # [batchSize, max_historyNum, slotType, hiddenSize]

        # 当前对话 视角
        history_CLSHiddenStates = historyEmbedding.pooler_output.reshape([batch_size, -1, self.config.hidden_size])
        history_padingMask = history_mask_sentences.sum(-1).gt(0).long() # 加和比0大  说明是真实的history [batchSize, max_historyNum] 防止padding的历史对话干扰
        new_CLSHiddenStates, currentTurn_ = self.CT_DH(currentEmbedding.pooler_output, history_CLSHiddenStates, history_padingMask) # [batchSize, max_historyNum, hiddenSize]

        # GNN
        slotNode = self.merge_slot_node(torch.cat([slotsHiddenStates, valuesHiddenStates], dim=-1))
        dialogueNode = new_CLSHiddenStates
        imor_ = self.IMOR(slotNode, dialogueNode, updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect) # [batchSize, max_historyNum, hiddenSize]

        # 计算权重
        slotName_score = self.SN_DH_score(slotName_) # [batchSize, max_historyNum, slotType, 1]
        currentTurn_score = self.CT_DH_score(currentTurn_) # [batchSize, max_historyNum, 1]
        imor_socre = self.IMOR_score(imor_) # [batchSize, max_historyNum, 1]

        # 计算总得embedding
        hiddenState = torch.mul(slotName_, slotName_score) + torch.mul(currentTurn_, currentTurn_score).unsqueeze(2) + torch.mul(imor_, imor_socre).unsqueeze(2) # [batchSize, max_historyNum, slotType, hiddenSize]

        # 计算得分
        score = self.hiddenState2score(hiddenState)  # [batchSize, max_historyNum, slotType, 1]
        score = torch.squeeze(score, dim=-1) # [batchSize, max_historyNum, slotType] 降维

        # mask掉不存在的历史轮的分数
        history_padingMask = history_padingMask.unsqueeze(-1).expand_as(score) # [batchSize, max_historyNum, slotType]
        score = score.masked_fill(history_padingMask==0, -1e9) # 不可能的对话的分数无穷小 [batchSize, max_historyNum, slotType]

        return score


class MultiRelationalGCN(nn.Module):
    def __init__(self, hidden_size, layer_nums, relation_type):
        '''
            hiddenSize: 节点隐藏层维度
            layer_nums: 层次数
            relation_type: 关系种类
        '''
        super(MultiRelationalGCN, self).__init__()

        self.hidden_size = hidden_size
        self.f_rs = nn.ModuleList()
        for i in range(relation_type):
            self.f_rs.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_nums = layer_nums
        self.f_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_g = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, slot_node:torch.Tensor, dialogue_node:torch.Tensor, update_current_mm:torch.Tensor, slot_all_connect:torch.Tensor, update_mm:torch.Tensor, slot_domain_connect:torch.Tensor):
        '''
        slot_node : [turnNum, slotType, hiddenSize]
        dialogue_node, [turnNum, hiddenSize] 
        update_current_mm, [turnNum, slotType, turnNum]
        slot_all_connect, [turnNum, slotType, SlotType] bugs: 当前非对称 需要改为对称
        update_mm, [turnNum, slotType, turnNum]
        slot_domain_connect  [turnNum, slotType, SlotType] 对称

        return: 

        slot_node, [turnNum, slotType, hiddenSize]
        dialogue_node [turnNum, turnNum, hiddenSize]
        '''
        dialogue_node = dialogue_node.unsqueeze(0).repeat(dialogue_node.shape[0],1,1) # [turnNum, hiddenSize] -> [turnNum, turnNum, hiddenSize] 
        for i in range(self.layer_nums):
            dialogue_node_current = self.f_s(dialogue_node) # [turnNum, turnNum, hiddenSize]  -> [turnNum, turnNum, hiddenSize] 
            slot_node_current = self.f_s(slot_node) # [turnNum, slotType, hiddenSize] -> [turnNum, slotType, hiddenSize]
            
            relation_dialogue_node_neighbour = [] # [relationType, turnNum, turnNum, hiddenSize]
            relation_slot_node_neighbour = [] # [relationType, turnNum, slotType, hiddenSize]

            # 计算每种关系得邻居节点表示
            for f_r in self.f_rs:
                relation_dialogue_node_neighbour.append(f_r(dialogue_node))
                relation_slot_node_neighbour.append(f_r(slot_node))

            # 处理第一种关系 更新的slot 与 此turn相连
            update_current_mm_d2s = update_current_mm.matmul(relation_dialogue_node_neighbour[0]) / (update_current_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) # [turnNum, slotType, turnNum] * [turnNum, turnNum, hiddenSize] / [turnNum, slotType, hiddenSize]
            update_current_mm_s2d = update_current_mm.transpose(1,2).matmul(relation_slot_node_neighbour[0]) / (update_current_mm.transpose(1,2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4) # [turnNum, turnNum, slotType] * [turnNum, slotType, hiddenSize] / [turnNum, turnNum, hiddenSize]
            
            # 处理第二种关系 要更新的slot 连接 其他的slot
            slot_all_connect_s2s = slot_all_connect.matmul(relation_slot_node_neighbour[1]) / (slot_all_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) # [turnNum, slotType, SlotType] * [turnNum, turnNum, hiddenSize] * [turnNum, slotType, hiddenSize] / [turnNum, slotType, hiddenSize]
            
            # 处理第三种关系 slot与最近要更新此slot的对话turn
            update_mm_d2s = update_mm.matmul(relation_dialogue_node_neighbour[2]) / (update_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) # [turnNum, slotType, turnNum] * [turnNum, turnNum, hiddenSize] / [turnNum, slotType, hiddenSize]
            update_mm_s2d = update_mm.transpose(1,2).matmul(relation_slot_node_neighbour[2]) / (update_mm.transpose(1,2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4) # [turnNum, turnNum, slotType] * [turnNum, slotType, hiddenSize] / [turnNum, turnNum, hiddenSize]

            # 处理第四种关系 同领域之间的slot相连接
            slot_domain_connect_s2s = slot_domain_connect.matmul(relation_slot_node_neighbour[3]) / (slot_domain_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4) # [turnNum, slotType, SlotType] * [turnNum, turnNum, hiddenSize] * [turnNum, slotType, hiddenSize] / [turnNum, slotType, hiddenSize]

            # 计算当前轮的节点状态
            dialogue_node_current = dialogue_node_current + update_current_mm_s2d + update_mm_s2d # [turnNum, turnNum, hiddenSize]
            slot_node_current = slot_node_current + update_current_mm_d2s + slot_all_connect_s2s + update_mm_d2s + slot_domain_connect_s2s # [turnNum, slotType, hiddenSize]

            # 门控更新
            slot_gate = F.sigmoid(self.f_g(torch.cat([slot_node_current, slot_node], dim=-1))) # [turnNum, slotType, hiddenSize]
            slot_node = (F.relu(slot_node_current) * slot_gate) + (slot_node * (1-slot_gate)) # [turnNum, slotType, hiddenSize]

            dialogue_gate = F.sigmoid(self.f_g(torch.cat([dialogue_node_current, dialogue_node], dim=-1))) # [turnNum, turnNum, hiddenSize]
            dialogue_node = (F.relu(dialogue_node_current) * dialogue_gate) + (dialogue_node * (1-dialogue_gate)) # [turnNum, turnNum, hiddenSize]

        return slot_node, dialogue_node
