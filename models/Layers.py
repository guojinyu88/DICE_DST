from audioop import mul
import sys
from typing import List
sys.path.append('./')

from configs.DiCoSConfig import DiCoSConfig
from transformers import AlbertModel, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn as nn
import torch.nn.functional as F
import torch




class SAM(nn.Module):
    def __init__(self, config:DiCoSConfig):
        super(SAM, self).__init__()

        self.config = config

    def forward(self, sentencesHiddenStates:torch.Tensor, slotsHiddenStates:torch.Tensor, mask_sentences:torch.Tensor) -> torch.Tensor:
        '''

        sentencesHiddenStates : [batchSize, seqLen, hiddenSize] : 句子得表示
        slotsHiddenStates : [batchSize, slotType, hiddenSize] : 所有[slot]的表示
        mask_sentences : [batchSize, sqeLen] : 希望存在attention的地方标记1, 其余地方标记0

        获取slot 与 seqLen 之间的相关度

        output : [batchSize, slotType, seqLen]

        '''
        
        attentionScore = torch.matmul(sentencesHiddenStates, slotsHiddenStates.transpose(-1, -2)) # [batchSize, seqLen, hiddenSize] * [batchSize, hiddenSize, slotType] -> [batchSize, seqLen, slotType]
        attentionScore = torch.div(attentionScore, self.config.hidden_size**(1/2))
        attentionScore = attentionScore.transpose(-1, -2) # [batchSize, seqLen, slotType] -> [batchSize, slotType, seqLen]

        # 将mask区域的attention值设置为0
        mask = mask_sentences.unsqueeze(1).expand_as(attentionScore) # [batchSize, sqeLen] -> [batchSize, slotType, seqLen]
        attentionScore = attentionScore.masked_fill(mask==0, -1e9) # [batchSize, slotType, seqLen]

        attentionScore = F.softmax(attentionScore, dim=-1) # [batchSize, slotType, seqLen]

        return attentionScore

class Encoder(nn.Module):
    def __init__(self, config:DiCoSConfig):
        super(Encoder, self).__init__()

        self.config = config
        self.albert = AlbertModel.from_pretrained(config.bert_path)
        self.albert.resize_token_embeddings(len(config.tokenizer)) # 适应新的词表大小

    def forward(self, sentencesTokens:torch.Tensor, attentionMask:torch.Tensor, segmentEmbedding:torch.Tensor, output_attentions=None) -> BaseModelOutputWithPooling:
        '''
        sentencesTokens: [batchSize, seqLen]
        attentionMask: [batchSize, seqLen]
        segmentEmbedding: [batchSize, seqLen]

        output: BaseModelOutputWithPooling
        
        '''
        sentencesHiddenStates = self.albert(sentencesTokens,  # 句子表示
                                            attention_mask = attentionMask, # attention mask 
                                            token_type_ids = segmentEmbedding,
                                            output_attentions=output_attentions) # segment embedding
        return sentencesHiddenStates

class SentencesSlotCrossEncoder(nn.Module):
    '''
        获取样本的每一个slot的表示
    '''
    def __init__(self, config:DiCoSConfig):
        super(SentencesSlotCrossEncoder, self).__init__()

        self.config = config
        self.sam = SAM(config)

    def forward(self, sentencesHiddenStates:torch.Tensor, slotsHiddenStates:torch.Tensor, mask_sentences:torch.Tensor):
        '''

        sentencesHiddenStates : [batchSize, seqLen, hiddenSize] : 句子得表示
        slotsHiddenStates : [batchSize, slotType, hiddenSize] : 所有[slot]的表示
        mask_sentences : [batchSize, sqeLen] : 句子的地方填写1, slot的地方填写0, padding的地方写0

        output : [batchSize, slotType, hiddenSize] 每个句子的每个slot的表示

        '''

        attentionScore = self.sam(sentencesHiddenStates, slotsHiddenStates, mask_sentences) # [batchSize, slotType, seqLen]

        # 复制句子表示，每个句子复制slotType次
        sentencesSlotsHiddenStates = sentencesHiddenStates.unsqueeze(dim=1) # [batchSize, 1, seqLen, hiddenSize]
        sentencesSlotsHiddenStates = sentencesSlotsHiddenStates.repeat(1, self.config.slotTypeNum, 1, 1) # [batchSize, slotType, seqLen, hiddenSize]

        # 拓展attention表示
        attentionScore = torch.unsqueeze(attentionScore, dim=-1) # [batchSize, slotType, seqLen, 1]
        attentionScore = attentionScore.repeat(1, 1, 1, self.config.hidden_size)# [batchSize, slotType, seqLen, hiddenSize]
        
        # 加权求和得到每个slot的句子表示
        sentencesSlotsHiddenStates = attentionScore * sentencesSlotsHiddenStates # [batchSize, slotType, seqLen, hiddenSize]
        sentencesSlotsHiddenStates = torch.sum(sentencesSlotsHiddenStates, dim=2) # [batchSize, slotType, hiddenSize]

        return sentencesSlotsHiddenStates

class PreliminarySelector(nn.Module):
    '''
        根据句子的slot表示分类，我们是不是要更新这个槽值
    '''
    def __init__(self, config:DiCoSConfig):
        super(PreliminarySelector, self).__init__()
        
        self.config = config
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)

    def forward(self, sentencesSlotsHiddenStates:torch.Tensor):
        '''

        sentencesSlotsHiddenStates: [batchSize, slotType, hiddenSize] : 每个句子的每个slot的表示

        output : prob_update: [batchSize, slotType, 2], pre_score: [batchSize, slotType]

        '''


        # 分类是否重写
        sentencesSlotsHiddenStates = self.dropout(sentencesSlotsHiddenStates) # [batchSize, slotType, hiddenSize]
        logit_update = self.fc(sentencesSlotsHiddenStates) #  [batchSize, slotType, 2]
        prob_update = F.softmax(logit_update, dim=-1)#  [batchSize, slotType, 2]

        pre_score = prob_update[:,:,self.config.dict_update["update"]] - prob_update[:,:,self.config.dict_update["inherit"]] # 计算得分

        return prob_update, pre_score

class Generator(nn.Module):
    '''
        给出分类和抽取两种策略的答案信息
    '''
    def __init__(self, config:DiCoSConfig, mask_cate_value:torch.Tensor, mask_cateSlot:torch.Tensor):
        '''
        mask_cate_value: [slotType, maxValueNum]: 表示这个slot有多少个可能得value, 可能得地方填1, 不可能的地方填0
        mask_cateSlot : [slotType] : 可以被分类的mask_cateSlot标注为1
        '''
        super(Generator, self).__init__()

        self.config = config
        self.mask_cate_value = nn.Parameter(mask_cate_value, requires_grad=False)
        self.mask_cateSlot = nn.Parameter(mask_cateSlot, requires_grad=False)

        self.dropout = nn.Dropout(0.5)

        self.sentencesSlotCrossEncoder = SentencesSlotCrossEncoder(config) # 获取每个slot的分类聚合表述

        self.fc_start = nn.Linear(config.hidden_size, config.hidden_size) # 获取每个slot的抽取起始位置
        self.fc_end = nn.Linear(config.hidden_size, config.hidden_size) # 获取每个slot的抽取起始位置

        self.sam = SAM(config) # 获取每个slot的抽取起始位置

        self.fcForEachSlot = nn.Parameter(torch.randn([config.slotTypeNum, config.hidden_size, config.maxSlotValue], dtype=torch.float32, device=config.device), requires_grad=True) # 分类使用的参数

    def forward(self, sentencesHiddenStates:torch.Tensor, slotsHiddenStates:torch.Tensor, mask_answerArea:torch.Tensor, sentencesSlotsHiddenStates:torch.Tensor) -> torch.Tensor:
        '''

        sentencesHiddenStates : [batchSize, seqLen, hiddenSize] : 句子得表示
        slotsHiddenStates : [batchSize, slotType, hiddenSize] : 所有[slot]的表示
        mask_answerArea: [batchSize, sqeLen]: start 和 end 可能出现的位置 (包不包括[CLS])
        sentencesSlotsHiddenStates: [batchSize, slotType, hiddenSize] : 每个句子的每个slot的表示

        output : 
            startProb, [batchSize, slotType, seqLen] 各个词为起始词的概率
            endProb,  [batchSize, slotType, seqLen] 各个词为末尾词的概率
            slotValueProb [batchSize, slotType, maxSlotValue] 各slot的分类概率值

        '''

        # 抽取
        sentencesHiddenStates = self.dropout(sentencesHiddenStates)
        sentencesHiddenStates_start = self.fc_start(sentencesHiddenStates) # [batchSize, seqLen, hiddenSize]
        sentencesHiddenStates_end = self.fc_end(sentencesHiddenStates) # [batchSize, seqLen, hiddenSize]

        startProb = self.sam(sentencesHiddenStates_start, slotsHiddenStates, mask_answerArea) # [batchSize, slotType, seqLen]
        endProb = self.sam(sentencesHiddenStates_end, slotsHiddenStates, mask_answerArea) # [batchSize, slotType, seqLen]


        # 分类

        sentencesSlotsHiddenStates = sentencesSlotsHiddenStates.transpose(0, 1) # [batchSize, slotType, hiddenSize] -> [slotType, batchSize, hiddenSize] 转换表示方式
        sentencesSlotsHiddenStates = self.dropout(sentencesSlotsHiddenStates)
        slotValueProb = torch.matmul(sentencesSlotsHiddenStates, self.fcForEachSlot) # [slotType, batchSize, hiddenSize] * [slotType, hiddenSize, maxSlotValue] -> [slotType, batchSize, maxSlotValue] 对每一个slot用不同的分类参数做分类处理
        slotValueProb = slotValueProb.transpose(0, 1) # [batchSize, slotType, maxSlotValue] 表示方式再转换回去

        # mask 掉不可能的答案 (答案种类padding)
        slotValueMask = self.mask_cate_value.unsqueeze(dim=0) # [1, slotType, maxSlotValue]
        slotValueMask = slotValueMask.expand_as(slotValueProb) # [batchSize, slotType, maxSlotValue]
        slotValueProb = slotValueProb.masked_fill(slotValueMask==0, -1e9) # 将不可能的地方填充上 -1e9

        # 拿到答案概率分布
        slotValueProb = F.softmax(slotValueProb, dim=-1) # [batchSize, slotType, maxSlotValue]

        # 归零非分类的概率 （不可分类）
        slotValueProbMask = self.mask_cateSlot.unsqueeze(dim=0).unsqueeze(dim=2) # [1, slotType, 1]
        slotValueProbMask = slotValueProbMask.expand_as(slotValueProb) # [batchSize, slotType, maxSlotValue]
        slotValueProb = slotValueProb.masked_fill(slotValueProbMask==0, 0) # 概率全部归0 [batchSize, slotType, maxSlotValue]
        

        return startProb, endProb, slotValueProb

class UltimateSelector(nn.Module):
    def __init__(self, config:DiCoSConfig, generator:Generator):
        super(UltimateSelector, self).__init__()
        self.config = config

        self.generator = generator
 
    def forward(self, sentencesHiddenStates:torch.Tensor, slotsHiddenStates:torch.Tensor, mask_answerArea:torch.Tensor, sentencesSlotsHiddenStates:torch.Tensor):
        '''
        sentencesHiddenStates : [batchSize, seqLen, hiddenSize] : 句子得表示
        slotsHiddenStates : [batchSize, slotType, hiddenSize] : 所有[slot]的表示
        mask_answerArea: [batchSize, sqeLen]: start 和 end 可能出现的位置 (包括[CLS]) 1 可以出现 要求0的位置为1
        sentencesSlotsHiddenStates: [batchSize, slotType, hiddenSize] : 每个句子的每个slot的表示

        output : 
            startProb, [batchSize, slotType, seqLen] 各个词为起始值的概率
            endProb,  [batchSize, slotType, seqLen] 各个词为末尾词的概率
            slotValueProb [batchSize, slotType, maxSlotValue] 各slot的分类概率值
            extUltScore [batchSize, slotType] 抽取的话得出的分数
            clsUltScore [batchSize, slotType] 分类的话得出的分数
        '''

        startProb, endProb, slotValueProb = self.generator(sentencesHiddenStates, slotsHiddenStates, mask_answerArea, sentencesSlotsHiddenStates)
        # startProb, [batchSize, slotType, seqLen]
        # endProb,  [batchSize, slotType, seqLen]
        # slotValueProb [batchSize, slotType, maxSlotValue]


        # extUltScore
        
        # 分母

        # 构建合法start end 位置的mask
        ext_mask = torch.ones([startProb.shape[0], self.config.slotTypeNum, self.config.pad_size, self.config.pad_size], dtype=torch.int8, device=self.config.device) # [batchSize, slotType, start, end] 1为合法
        ext_mask = torch.triu(ext_mask, 0)
        ext_mask = ext_mask * mask_answerArea.unsqueeze(1).unsqueeze(-1) # start 不合法的mask掉
        ext_mask = ext_mask * mask_answerArea.unsqueeze(1).unsqueeze(-2) # end 不合法的mask掉
        ext_mask[:,:,0,1:] = 0 # start 为cls 则 end 只能是cls
        
        probNorm = torch.exp(torch.unsqueeze(startProb, -1) + torch.unsqueeze(endProb, -2)) # [batchSize, slotType, start, end]
        allSpanProb = torch.masked_fill(probNorm, ext_mask==0, 0) # 将不合法的位置全部归0
        probNorm = allSpanProb.sum(-1).sum(-1) # [batchSize, slotType]

        # 分子
        spanProb = allSpanProb[:,:,1:,:].max(-1).values.max(-1).values # [batchSize, slotType] start不为0的情况下 最大概率span 对应的值
        nullProb = torch.exp(startProb[:,:,0]+endProb[:,:,0])

        # 计算分数
        extUltScore = torch.div(spanProb, probNorm) - torch.div(nullProb, probNorm) # update - inherit
                

        # clsUltScore

        slotMaxProb, _ = torch.max(slotValueProb[:,:,1:], dim=-1) # 除去none之外最大的
        slotNoneProb = slotValueProb[:,:,0] # [batchSize, slotType] 认为没有候选词的概率值

        clsUltScore = slotMaxProb - slotNoneProb # # update - inherit


        return startProb, endProb, slotValueProb, extUltScore, clsUltScore


if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DSS_DST/config.json')

    asm = SAM(config)

    sentencesHiddenStates = torch.rand([2,8,768])
    slotsHiddenStates = torch.rand([2, 30, 768])
    mask = torch.tensor([[1,1,1,1,0,0,0,0], [1,1,1,0,0,0,0,0]])
    print(asm(sentencesHiddenStates, slotsHiddenStates, mask))


