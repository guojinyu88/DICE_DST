from models.DiCoS import DiCoS
from models.Teacher import Teacher
import torch.nn as nn
import torch

from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema

class Distillation(nn.Module):
    def __init__(self, config:DiCoSConfig, schema:MultiWoZSchema) -> None:
        super(Distillation, self).__init__()

        self.config = config
        self.schema = schema

        self.teacher = Teacher(config)

        self.dicos = DiCoS(config, schema)

    def forward(self, current_sentencesTokens:torch.Tensor, current_attentionMask:torch.Tensor, current_segmentEmbedding:torch.Tensor, 
                current_slotPosition:torch.Tensor, current_valuePosition:torch.Tensor,
                history_sentencesTokens:torch.Tensor, history_attentionMask:torch.Tensor, history_segmentEmbedding:torch.Tensor, history_mask_sentences:torch.Tensor,
                updateSlot2current:torch.Tensor, updateSlot2allSlot:torch.Tensor, slot2lastUpdateTurn:torch.Tensor, slot_domain_connect:torch.Tensor,
                selectedHistory_sentencesTokens:torch.Tensor,selectedHistory_attentionMask:torch.Tensor, selectedHistory_segmentEmbedding:torch.Tensor,
                selectedHistory_slotPosition:torch.Tensor, selectedHistory_mask_sentences:torch.Tensor):
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



            return:
            score,  [batchSize, slotType, max_historyNum]
            selectedScore, [updateSlotNum]           每个更新slot选择当前对话的分数
            startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
            endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
            slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果
            selectedHistory_sentencesTokens [updateSlotNum, seqLen], 从哪一句话中抽取出来的
            
            gold_attentionMap, 
            student_attentionMap


        
        '''
        score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens, student_attentionMap = self.dicos.forward(current_sentencesTokens, current_attentionMask, current_segmentEmbedding, 
                                                                                        current_slotPosition, current_valuePosition,  # 当前轮信息
                                                                                        history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, # 历史轮信息
                                                                                        updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, # 四个边
                                                                                        selectedHistory_sentencesTokens,selectedHistory_attentionMask, selectedHistory_segmentEmbedding, 
                                                                                        selectedHistory_slotPosition, selectedHistory_mask_sentences, # 当前轮+·被选择对话的信息
                                                                                        output_attentions = True) # 返回attention map
        
        attentionMask = selectedHistory_sentencesTokens.not_equal(self.config.tokenizer.pad_token_id).long()
        with torch.no_grad():
            _, _, teacher_attentionMap = self.teacher.forward(selectedHistory_sentencesTokens, attentionMask, output_attentions=True)

        # student_attentionMap = torch.stack(student_attentionMap, dim=1)
        # teacher_attentionMap = torch.stack(teacher_attentionMap, dim=1)
        gold_attentionMap = 0.9 * student_attentionMap[-1] + 0.1 * teacher_attentionMap[-1]

        del teacher_attentionMap


        return score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens, gold_attentionMap, student_attentionMap[-1]
        