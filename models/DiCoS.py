import sys
sys.path.append('./')

import torch
import torch.nn as nn
from models.Layers import Encoder, SentencesSlotCrossEncoder, Generator, PreliminarySelector, UltimateSelector
from models.MultiPrespective import MultiPrespectiveSelector
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
import torch.nn.functional as F

class DSS(nn.Module):
    def __init__(self, config:DiCoSConfig, schema:MultiWoZSchema):
        super(DSS, self).__init__()
        self.config = config
        self.schema = schema

        self.encoder = Encoder(config)

        # Preliminary Selector
        self.sentenceEnhanceSlot = SentencesSlotCrossEncoder(config)
        self.preliminarySelector = PreliminarySelector(config)

        # Ultimate Selector
        self.ultimateGenerator = Generator(config, schema.mask_cate_value,  schema.mask_cateSlot)
        self.ultimateSelector = UltimateSelector(config, self.ultimateGenerator)

    def forward(self, sentencesTokens:torch.Tensor, attentionMask:torch.Tensor, segmentEmbedding:torch.Tensor, slotPosition:torch.Tensor, mask_sentences:torch.Tensor):
        '''
            sentencesTokens: [batchSize, seqLen]
            attentionMask: [batchSize, seqLen]
            segmentEmbedding: [batchSize, seqLen]
            slotPosition: [batchSize, slotType]
            mask_sentences : [batchSize, sqeLen] : 句子的地方填写1, slot的地方填写0, padding的地方写0
        
            return:

            prob_update, [batchSize, slotType, 2]
            pre_score, [batchSize, slotType]
            startProb, [batchSize, slotType, seqLen] 各个词为起始值的概率
            endProb,  [batchSize, slotType, seqLen] 各个词为末尾词的概率
            slotValueProb [batchSize, slotType, maxSlotValue] 各slot的分类概率值
            extUltScore [batchSize, slotType] 抽取的话得出的分数
            clsUltScore [batchSize, slotType] 分类的话得出的分数 抽取槽为1
        '''

        sentencesHiddenStates = self.encoder(sentencesTokens, attentionMask, segmentEmbedding).last_hidden_state # [batchSize, seqLen, hiddenSize]
        # 取出[slot]的表示
        slotsHiddenStates = torch.gather(sentencesHiddenStates, dim=1, index=slotPosition.unsqueeze(-1).repeat([1, 1, self.config.hidden_size])) # [batchSize, slotType, hiddenSize]
        # 计算每个[slot]与整句话的交互表示
        sentencesSlotsHiddenStates = self.sentenceEnhanceSlot(sentencesHiddenStates, slotsHiddenStates, mask_sentences) # [batchSize, slotType, hiddenSize]
        
        # Preliminary Selector
        prob_update, pre_score = self.preliminarySelector(sentencesSlotsHiddenStates) # isOverWrite: [batchSize, slotType, 2], pre_score: [batchSize, slotType]
        

        # 可以预测为cls
        mask_answerArea = mask_sentences.clone().detach()
        mask_answerArea[:,0] = 1 # 允许预测为cls

        # Ultimate Selector
        startProb, endProb, slotValueProb, extUltScore, clsUltScore = self.ultimateSelector(sentencesHiddenStates, slotsHiddenStates, mask_answerArea, sentencesSlotsHiddenStates)

        return prob_update, pre_score, startProb, endProb, slotValueProb, extUltScore, clsUltScore

class DiCoS(nn.Module):

    def __init__(self, config:DiCoSConfig, schema:MultiWoZSchema):
        super(DiCoS, self).__init__()

        self.config = config
        self.schema = schema

        self.encoder = Encoder(config)

        # Multi-Prespective Selector
        self.multiPrespectiveSelector = MultiPrespectiveSelector(config)

        # Slot Value Generator
        self.sentenceEnhanceSlot = SentencesSlotCrossEncoder(config)
        self.SVG = Generator(config, schema.mask_cate_value,  schema.mask_cateSlot)

    def forward(self, current_sentencesTokens:torch.Tensor, current_attentionMask:torch.Tensor, current_segmentEmbedding:torch.Tensor, 
                current_slotPosition:torch.Tensor, current_valuePosition:torch.Tensor,
                history_sentencesTokens:torch.Tensor, history_attentionMask:torch.Tensor, history_segmentEmbedding:torch.Tensor, history_mask_sentences:torch.Tensor,
                updateSlot2current:torch.Tensor, updateSlot2allSlot:torch.Tensor, slot2lastUpdateTurn:torch.Tensor, slot_domain_connect:torch.Tensor,
                selectedHistory_sentencesTokens:torch.Tensor,selectedHistory_attentionMask:torch.Tensor, selectedHistory_segmentEmbedding:torch.Tensor,
                selectedHistory_slotPosition:torch.Tensor, selectedHistory_mask_sentences:torch.Tensor,
                output_attentions=None):
        
        '''
            current_sentencesTokens:torch.Tensor, [batchSize, seqLen] 当前轮对话token
            current_attentionMask:torch.Tensor, [batchSize, seqLen] 当前轮对话token
            current_segmentEmbedding:torch.Tensor, [batchSize, seqLen] 当前轮对话token
            current_slotPosition:torch.Tensor, [batchSize, slotType] 当前轮对话 [slot] 在哪
            current_valuePosition:torch.Tensor, [batchSize, slotType] 当前轮对话 [value] 在哪

            history_sentencesTokens:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_attentionMask:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_segmentEmbedding:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_mask_sentences:torch.Tensor, [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0, 决定选择历史对话能选择的范围
            updateSlot2current, [batchSize, slotType, 1+max_historyNum] 更新槽与当前轮    需要保证padiing出来的history 不与任何点相连   [相当重要 第一列直接决定了我要更新哪些slot]
            updateSlot2allSlot, [batchSize, slotType, SlotType] 本轮要更新的槽链接所有的槽
            slot2lastUpdateTurn, [batchSize, slotType, 1+max_historyNum] 槽与最近更新轮    需要保证padiing出来的history 不与任何点相连
            slot_domain_connect  [batchSize, slotType, SlotType] 同domain的槽相连

            selectedHistory_sentencesTokens: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
            selectedHistory_attentionMask: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
            selectedHistory_segmentEmbedding: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
            selectedHistory_slotPosition: [batchSize, maxHistoryNum, slotType] [slot]的所在位置
            selectedHistory_mask_sentences: [batchSize, maxHistoryNum, seqLen] 句子的地方填写1, slot的地方填写0, padding的地方写0

            output_attentions=None 是否要返回 attention map 用于蒸馏操作



            return:
            score,  [batchSize, slotType, max_historyNum]
            selectedScore, [updateSlotNum]           每个更新slot选择当前对话的分数
            startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
            endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
            slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果
            selectedHistory_sentencesTokens [updateSlotNum, seqLen], 从哪一句话中抽取出来的


        
        '''
        
        # 编码当前轮对话
        current_embedding = self.encoder.forward(current_sentencesTokens, current_attentionMask, current_segmentEmbedding)

        # 编码历史对话
        batchsize = history_sentencesTokens.shape[0]
        maxHistoryNum = history_sentencesTokens.shape[1] # 获取当前batch的最大历史对话数量
        history_sentencesTokens = history_sentencesTokens.reshape([-1, self.config.pad_size]) # [batchSize*maxHistoryNum, seqLen]
        history_attentionMask = history_attentionMask.reshape([-1, self.config.pad_size]) # [batchSize*maxHistoryNum, seqLen]
        history_segmentEmbedding = history_segmentEmbedding.reshape([-1, self.config.pad_size]) # [batchSize*maxHistoryNum, seqLen]

        with torch.no_grad(): # 历史对话的embedding不记录梯度了
            history_embedding = self.encoder.forward(history_sentencesTokens, history_attentionMask, history_segmentEmbedding) # [batchSize*maxHistoryNum, seqLen, hiddenSize]

        # 计算历史对话的得分
        score = self.multiPrespectiveSelector.forward(current_embedding, current_slotPosition, current_valuePosition, # 当前轮
                                                        history_embedding, history_mask_sentences, # 历史轮
                                                        updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect # 图上的四个关系
        ) # [batchSize, max_historyNum, slotType]
        score = score.transpose(-2,-1) # [batchSize, slotType, max_historyNum]
        score = torch.sigmoid(score)

        # 人工修改分数让他选择最后一次可用对话 #
        score = torch.arange(1, maxHistoryNum+1, dtype=torch.float32, device=self.config.device).unsqueeze(0).unsqueeze(0).repeat([batchsize, self.config.slotTypeNum, 1]) # [batchSize, slotType, max_historyNum]
        history_padingMask = history_mask_sentences.sum(-1).gt(0).long() # 加和比0大  说明是真实的history [batchSize, max_historyNum] 防止padding的历史对话干扰
        history_padingMask = history_padingMask.unsqueeze(1).expand_as(score) # [batchSize, slotType, max_historyNum]
        score = score.masked_fill(history_padingMask==0, 0) # 将不可能的历史对话的分数全部归0
        # 人工修改分数让他选择最后一次可用对话 #


        # 取出选择的历史对话

        # 判断每个样本要更新哪些槽
        updateSlot = updateSlot2current[:, :, 0] # [batchSize, slotType] 1的地方的slot需要更新
        # 判断每个槽选择了哪些对话
        selectedScore, selectedHistoryIdx = score.max(dim=-1) # [batchSize, slotType] 选择第几轮对话

        # 拿出历史对话的配套信息（全slot）
        selectedHistoryIdx_seqLen = selectedHistoryIdx.unsqueeze(-1).repeat([1,1,self.config.pad_size]) # [batchSize, slotType, seqLen] 取seqLen长度的
        selectedHistoryIdx_slotType = selectedHistoryIdx.unsqueeze(-1).repeat([1,1,self.config.slotTypeNum]) # [batchSize, slotType, slotType] 取slotType长度的
        selectedHistory_sentencesTokens = selectedHistory_sentencesTokens.gather(dim=1, index=selectedHistoryIdx_seqLen) # [batchSize, slotType, seqLen]
        selectedHistory_attentionMask = selectedHistory_attentionMask.gather(dim=1, index=selectedHistoryIdx_seqLen) # [batchSize, slotType, seqLen]
        selectedHistory_segmentEmbedding = selectedHistory_segmentEmbedding.gather(dim=1, index=selectedHistoryIdx_seqLen) # [batchSize, slotType, seqLen]
        selectedHistory_slotPosition = selectedHistory_slotPosition.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]
        selectedHistory_mask_sentences = selectedHistory_mask_sentences.gather(dim=1, index=selectedHistoryIdx_seqLen) # [batchSize, slotType, seqLen]

        # 拿出历史对话的配套信息（update slot）
        updateSlot_seqLen = updateSlot.unsqueeze(-1).repeat([1,1,self.config.pad_size]) # [batchSize, slotType, seqLen]
        updateSlot_slotType = updateSlot.unsqueeze(-1).repeat([1,1,self.config.slotTypeNum]) # [batchSize, slotType, slotType]
        selectedHistory_sentencesTokens = selectedHistory_sentencesTokens.masked_select(updateSlot_seqLen==1).reshape([-1, self.config.pad_size]) # [updateSlotNum, seqLen]
        selectedHistory_attentionMask = selectedHistory_attentionMask.masked_select(updateSlot_seqLen==1).reshape([-1, self.config.pad_size]) # [updateSlotNum, seqLen]
        selectedHistory_segmentEmbedding = selectedHistory_segmentEmbedding.masked_select(updateSlot_seqLen==1).reshape([-1, self.config.pad_size]) # [updateSlotNum, seqLen]
        selectedHistory_slotPosition = selectedHistory_slotPosition.masked_select(updateSlot_slotType==1).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]
        selectedHistory_mask_sentences = selectedHistory_mask_sentences.masked_select(updateSlot_seqLen==1).reshape([-1, self.config.pad_size]) # [updateSlotNum, seqLen]
        selectedScore = selectedScore.masked_select(updateSlot==1) # [updateSlotNum]


        # SVG
        selectedHistory_embedding = self.encoder.forward(selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding, output_attentions=output_attentions)
        sentencesHiddenStates = selectedHistory_embedding.last_hidden_state # [updateSlotName, seqLen, hiddenSize]
        # sentencesHiddenStates = sentencesHiddenStates * (1+torch.unsqueeze(selectedScore, -1).unsqueeze(-1))
        # 取出[slot]的表示
        slotsHiddenStates = torch.gather(sentencesHiddenStates, dim=1, index=selectedHistory_slotPosition.unsqueeze(-1).repeat([1, 1, self.config.hidden_size])) # [updateSlotNum, slotType, hiddenSize]
        # 计算每个[slot]与整句话的交互表示
        sentencesSlotsHiddenStates = self.sentenceEnhanceSlot(sentencesHiddenStates, slotsHiddenStates, selectedHistory_mask_sentences) # [updateSlotNum, slotType, hiddenSize]

        startProb, endProb, slotValueProb = self.SVG.forward(sentencesHiddenStates, slotsHiddenStates, selectedHistory_mask_sentences, sentencesSlotsHiddenStates)


        if output_attentions == None:
            return score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens
        else:
            return score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens, selectedHistory_embedding.attentions

if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    schema = MultiWoZSchema(config)
    dicos = DiCoS(config, schema)
    dicos.to(config.device)


    batchsize = 8
    seqLen = 400
    maxHistoryNum = 16
    slotType = 30
    for i in range(100):

        current_sentencesTokens = torch.randint(0,1000, [batchsize, seqLen], dtype=torch.long).to(config.device)
        current_attentionMask = torch.randint(0,2, [batchsize, seqLen], dtype=torch.long).to(config.device)
        current_segmentEmbedding = torch.randint(0,2, [batchsize, seqLen], dtype=torch.long).to(config.device)
        current_slotPosition = torch.range(0, slotType-1, dtype=torch.long).unsqueeze(0).repeat([batchsize, 1]).to(config.device)
        current_valuePosition = torch.range(0, slotType-1, dtype=torch.long).unsqueeze(0).repeat([batchsize, 1]).to(config.device)

        history_sentencesTokens = torch.randint(0,1000, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        history_attentionMask = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        history_segmentEmbedding = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        history_mask_sentences = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        updateSlot2current = torch.cat([torch.tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32).unsqueeze(-1), torch.zeros([slotType,maxHistoryNum], dtype=torch.float32)], dim=-1).unsqueeze(0).repeat([batchsize, 1, 1]).to(config.device)
        updateSlot2allSlot = torch.randint(0,2, [batchsize, slotType, slotType], dtype=torch.float32).to(config.device)
        slot2lastUpdateTurn = torch.randint(0,2, [batchsize, slotType, 1+maxHistoryNum], dtype=torch.float32).to(config.device)
        slot_domain_connect = torch.randint(0,2, [batchsize, slotType, slotType],dtype=torch.float32).to(config.device)
                    
        selectedHistory_sentencesTokens = torch.randint(0,1000, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        selectedHistory_attentionMask = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        selectedHistory_segmentEmbedding = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)
        selectedHistory_slotPosition = torch.range(0, slotType-1, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat([batchsize, maxHistoryNum, 1]).to(config.device)
        selectedHistory_mask_sentences = torch.randint(0,2, [batchsize, maxHistoryNum, seqLen], dtype=torch.long).to(config.device)

        
        score, selectedScore, startProb, endProb, slotValueProb = dicos(current_sentencesTokens, current_attentionMask, current_segmentEmbedding, 
                    current_slotPosition, current_valuePosition,
                    history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences,
                    updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect,
                    selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding,
                    selectedHistory_slotPosition, selectedHistory_mask_sentences)


        


    