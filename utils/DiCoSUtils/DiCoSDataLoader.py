import sys
from sklearn import config_context
sys.path.append('./')

import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from utils.DiCoSUtils.DiCoSMultiWoZDataUtils import SentencesInstences
from utils.DiCoSUtils.DiCoSMultiWoZDataUtils import MultiWoZDataset

def dicos_collate_fn(datas:List[List[SentencesInstences]], config:DiCoSConfig, buildSupervise:bool=True):
    '''
        对层层封装进行拆包，拆包成模型可以直接输入的格式
        [
            [ # 封装一个样本我们需要用到的所有内容
                turnid, # 当前轮的id
                current_turn, # 当前轮的输入 SentencesInstences
                list_sentencesInstence_history.copy(), # 用于对话选择过程
                list_select_history, # 用于选择好历史对话之后的SVG阶段
                graph, # 存储了四种边
                last_state,# 上一阶段的状态真值 用于验证acc
                current_state # 当前阶段的状态真值 用于验证acc
            ],
            ......
        ]

        return:
        update_slot: torch.Tensor, [batchSize, slotType] 0继承 1更新

        current_sentencesTokens:torch.Tensor, [batchSize, seqLen] 当前轮对话token
        current_attentionMask:torch.Tensor, [batchSize, seqLen] 当前轮对话token
        current_segmentEmbedding:torch.Tensor, [batchSize, seqLen] 当前轮对话token
        current_slotPosition:torch.Tensor, [batchSize, slotType] 当前轮对话 [slot] 在哪
        current_valuePosition:torch.Tensor, [batchSize, slotType] 当前轮对话 [value] 在哪

        history_sentencesTokens:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
        history_attentionMask:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
        history_segmentEmbedding:torch.Tensor, [batchSize, maxHistoryNum, seqLen] 历史对话token
        history_mask_sentences:torch.Tensor, [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
        updateSlot2current, [batchSize, slotType, 1+max_historyNum] 更新槽与当前轮    需要保证padiing出来的history 不与任何点相连
        updateSlot2allSlot, [batchSize, slotType, SlotType] 本轮要更新的槽链接所有的槽
        slot2lastUpdateTurn, [batchSize, slotType, 1+max_historyNum] 槽与最近更新轮    需要保证padiing出来的history 不与任何点相连
        slot_domain_connect  [batchSize, slotType, SlotType] 同domain的槽相连

        selectedHistory_sentencesTokens: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
        selectedHistory_attentionMask: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
        selectedHistory_segmentEmbedding: [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
        selectedHistory_slotPosition: [batchSize, maxHistoryNum, slotType] [slot]的所在位置
        selectedHistory_mask_sentences: [batchSize, maxHistoryNum, seqLen] 句子的地方填写1, slot的地方填写0, padding的地方写0

        cata_target, 分类的监督信号 [batchSize, maxHistoryNum, slotType] 0表示删除槽值 其他表示更新
        cate_mask,  0表示不需要监督 1 表示需要监督 [batchSize, maxHistoryNum, slotType]
        noncate_start, 抽取开始位置得监督信号 [batchSize, maxHistoryNum, slotType]
        noncate_end, 抽取结束位置得监督信号 [batchSize, maxHistoryNum, slotType]
        noncate_mask  0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到 [batchSize, maxHistoryNum, slotType]

        list_last_state, [batchSize] 里面装的是一个字典，存储了上一轮状态
        list_current_state [batchSize] 里面装的是一个字典，存储了当前轮状态

        


    '''
    # 我们要更新那些slot
    list_update_slot = [] # [batchSize, slotType] 0继承 1更新
    # 当前轮信息
    list_current_sentencesTokens = [] # [batchSize, seqLen] 当前轮对话token
    list_current_attentionMask = [] # [batchSize, seqLen] 当前轮对话token
    list_current_segmentEmbedding = [] # [batchSize, seqLen] 当前轮对话token
    list_current_slotPosition = [] # [batchSize, slotType] 当前轮对话 [slot] 在哪
    list_current_valuePosition = [] # [batchSize, slotType] 当前轮对话 [value] 在哪

    # 对话选择信息
    list_history_sentencesTokens = [] # [batchSize, maxHistoryNum, seqLen] 历史对话token
    list_history_attentionMask = [] # [batchSize, maxHistoryNum, seqLen] 历史对话token
    list_history_segmentEmbedding = [] # [batchSize, maxHistoryNum, seqLen] 历史对话token
    list_history_mask_sentences = [] # [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
    list_updateSlot2current = [] #  [batchSize, slotType, 1+max_historyNum] 更新槽与当前轮    需要保证padiing出来的history 不与任何点相连
    list_updateSlot2allSlot = [] # [batchSize, slotType, SlotType] 本轮要更新的槽链接所有的槽
    list_slot2lastUpdateTurn = [] # [batchSize, slotType, 1+max_historyNum] 槽与最近更新轮    需要保证padiing出来的history 不与任何点相连
    list_slot_domain_connect = [] # [batchSize, slotType, SlotType] 同domain的槽相连

    # 槽值生成信息
    list_selectedHistory_sentencesTokens = [] # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    list_selectedHistory_attentionMask = [] # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    list_selectedHistory_segmentEmbedding = [] # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    list_selectedHistory_slotPosition = [] # [batchSize, maxHistoryNum, slotType] [slot]的所在位置
    list_selectedHistory_mask_sentences = [] # [batchSize, maxHistoryNum, seqLen] 句子的地方填写1, slot的地方填写0, padding的地方写0

    # 监督信号
    list_cata_target = [] # 分类的监督信号 [batchSize, maxHistoryNum, slotType] 0表示删除槽值 其他表示更新槽值
    list_cate_mask = [] # 0表示不需要监督 1 表示需要监督 [batchSize, maxHistoryNum, slotType]
    list_noncate_start = [] # 抽取开始位置得监督信号 [batchSize, maxHistoryNum, slotType]
    list_noncate_end = [] # 抽取结束位置得监督信号 [batchSize, maxHistoryNum, slotType]
    list_noncate_mask = [] # 0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到[batchSize, maxHistoryNum, slotType]

    # 历史状态
    list_last_state = []
    list_current_state = []



    ######### process #########
    maxHistoryNum = 0 # 记录当前batch最长的history多长
    for data in datas:
        if len(data[2]) > maxHistoryNum:
            maxHistoryNum = len(data[2])
    maxHistoryNum = max(maxHistoryNum, 1) # 保证不会出现 maxHistoryNum = 0 的情况
    # 用于填充到最长的history的padding
    history_sentencesTokens_padding = [config.tokenizer.pad_token_id]*config.pad_size
    history_attentionMask_padding = [1]*config.pad_size
    history_segmentEmbedding_padding = [0]*config.pad_size
    history_mask_sentences_padding = [0]*config.pad_size
    history_slotPosition_padding = list(range(0, config.slotTypeNum))

    # 用于填充到最长的history的监督信号padding
    cata_target_sample_padding = [int(1e9)]*config.slotTypeNum
    cate_mask_sample_padding = [0]*config.slotTypeNum
    noncate_start_sample_padding = [int(1e9)]*config.slotTypeNum
    noncate_end_sample_padding = [int(1e9)]*config.slotTypeNum
    noncate_mask_sample_padding = [0]*config.slotTypeNum

    
    for data in datas:

        # 当前轮信息 #
        current_turn:SentencesInstences = data[1]

        list_current_sentencesTokens.append(current_turn.tokens_idx)
        list_current_attentionMask.append(current_turn.attentionMask)
        list_current_segmentEmbedding.append(current_turn.segmentEmbedding)
        list_current_slotPosition.append(current_turn.slotTokenPosition)
        list_current_valuePosition.append(current_turn.valueTokenPosition)

        if buildSupervise:
            list_update_slot.append(current_turn.supervised_update) 

        # 对话选择信息 #
        history_turns:List[SentencesInstences] = data[2]
        graph:List[List] = data[4]

        history_sentencesTokens = []
        history_attentionMask = []
        history_segmentEmbedding = []
        history_mask_sentences = []
        for history_turn in history_turns:
            history_sentencesTokens.append(history_turn.tokens_idx)
            history_attentionMask.append(history_turn.attentionMask)
            history_segmentEmbedding.append(history_turn.segmentEmbedding)
            history_mask_sentences.append(history_turn.sentencesMask)
        padding = maxHistoryNum - len(history_turns)
        history_sentencesTokens += ([history_sentencesTokens_padding]*padding) # [historyNum, seqLen] -> [maxHistoryNum, seqLen]
        history_attentionMask += ([history_attentionMask_padding]*padding)
        history_segmentEmbedding += ([history_segmentEmbedding_padding]*padding)
        history_mask_sentences += ([history_mask_sentences_padding]*padding)

        updateSlot2current = np.array(graph[0])
        updateSlot2current:np.ndarray = np.concatenate([updateSlot2current.reshape(config.slotTypeNum, 1), np.zeros([config.slotTypeNum, maxHistoryNum])], -1) # [slotType, 1+max_historyNum]
        updateSlot2current = updateSlot2current.tolist()
        updateSlot2allSlot = graph[1]
        slot2lastUpdateTurn = np.array(graph[2])
        slot2lastUpdateTurn:np.ndarray = np.concatenate([np.zeros([config.slotTypeNum, 1]), slot2lastUpdateTurn, np.zeros([config.slotTypeNum, padding])], -1) # [slotType, 1+max_historyNum]
        slot2lastUpdateTurn = slot2lastUpdateTurn.tolist()
        slot_domain_connect = graph[3]

        list_history_sentencesTokens.append(history_sentencesTokens)
        list_history_attentionMask.append(history_attentionMask)
        list_history_segmentEmbedding.append(history_segmentEmbedding)
        list_history_mask_sentences.append(history_mask_sentences)
        list_updateSlot2current.append(updateSlot2current)
        list_updateSlot2allSlot.append(updateSlot2allSlot)
        list_slot2lastUpdateTurn.append(slot2lastUpdateTurn)
        list_slot_domain_connect.append(slot_domain_connect)
        
        # 槽值生成信息与监督信号 #
        selectedHistory_turns:List[SentencesInstences] = data[3]

        selectedHistory_sentencesTokens = []
        selectedHistory_attentionMask = []
        selectedHistory_segmentEmbedding = []
        selectedHistory_slotPosition = []
        selectedHistory_mask_sentences = []

        # 一个sample所有可能历史对话的监督信号
        cata_target = []  # [maxHistoryNum, slotType]
        cate_mask = []  # [maxHistoryNum, slotType]
        noncate_start = []  # [maxHistoryNum, slotType]
        noncate_end = []  # [maxHistoryNum, slotType]
        noncate_mask = [] # [maxHistoryNum, slotType] 0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到

        for selectedHistory_turn in selectedHistory_turns: 
            selectedHistory_sentencesTokens.append(selectedHistory_turn.tokens_idx)
            selectedHistory_attentionMask.append(selectedHistory_turn.attentionMask)
            selectedHistory_segmentEmbedding.append(selectedHistory_turn.segmentEmbedding)
            selectedHistory_slotPosition.append(selectedHistory_turn.slotTokenPosition)
            selectedHistory_mask_sentences.append(selectedHistory_turn.sentencesMask)
            if buildSupervise:
                # 一个历史对话的监督信号
                cata_target_sample = [] # [slotType]
                cate_mask_sample = [] # [slotType] 
                noncate_start_sample = [] # [slotType]
                noncate_end_sample = [] # [slotType]
                noncate_mask_sample = [] # [slotType]  0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到
    
                for target in selectedHistory_turn.supervised_cata:
                    if target == None: # 分类继承 + 抽取槽
                        cata_target_sample.append(int(1e9))
                        cate_mask_sample.append(0)
                    else:
                        cata_target_sample.append(target)
                        cate_mask_sample.append(1)

                for (start, end) in selectedHistory_turn.supervised_noncate:
                    if start == None and end == None: # 分类槽抽取不到 或者 抽取槽继承
                        noncate_start_sample.append(int(1e9))
                        noncate_end_sample.append(int(1e9))
                        noncate_mask_sample.append(0)
                    elif start == -1 and end == -1: # 抽取槽抽取不到
                        noncate_start_sample.append(start)
                        noncate_end_sample.append(end)
                        noncate_mask_sample.append(2)
                    else: # 抽取槽更新抽取的到+分类槽抽取的到
                        noncate_start_sample.append(start)
                        noncate_end_sample.append(end)
                        noncate_mask_sample.append(1)

                cata_target.append(cata_target_sample)
                cate_mask.append(cate_mask_sample)
                noncate_start.append(noncate_start_sample)
                noncate_end.append(noncate_end_sample)
                noncate_mask.append(noncate_mask_sample)

        padding = maxHistoryNum - len(selectedHistory_turns)
        if len(selectedHistory_turns) == 0: # 处理第0轮无历史对话的情况，选谁都一样, 都是当前轮
            selectedHistory_sentencesTokens += ([current_turn.tokens_idx]*padding)
            selectedHistory_attentionMask += ([current_turn.attentionMask]*padding)
            selectedHistory_segmentEmbedding += ([current_turn.segmentEmbedding]*padding)
            selectedHistory_slotPosition += ([current_turn.slotTokenPosition]*padding)
            selectedHistory_mask_sentences += ([current_turn.sentencesMask]*padding)

            if buildSupervise:
                # 当前轮对话的监督信号
                cata_target_sample = [] # [slotType]
                cate_mask_sample = [] # [slotType] 
                noncate_start_sample = [] # [slotType]
                noncate_end_sample = [] # [slotType]
                noncate_mask_sample = [] # [slotType]  0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到

                for target in current_turn.supervised_cata:
                    if target == None: # 分类继承 + 抽取槽
                        cata_target_sample.append(int(1e9))
                        cate_mask_sample.append(0)
                    else:
                        cata_target_sample.append(target)
                        cate_mask_sample.append(1)

                for (start, end) in current_turn.supervised_noncate:
                    if start == None and end == None: # 分类槽抽取不到 或者 抽取槽继承
                        noncate_start_sample.append(int(1e9))
                        noncate_end_sample.append(int(1e9))
                        noncate_mask_sample.append(0)
                    elif start == -1 and end == -1: # 抽取槽抽取不到
                        noncate_start_sample.append(start)
                        noncate_end_sample.append(end)
                        noncate_mask_sample.append(2)
                    else: # 抽取槽更新抽取的到+分类槽抽取的到
                        noncate_start_sample.append(start)
                        noncate_end_sample.append(end)
                        noncate_mask_sample.append(1)

                cata_target += ([cata_target_sample]*padding)
                cate_mask += ([cate_mask_sample]*padding)
                noncate_start += ([noncate_start_sample]*padding)
                noncate_end += ([noncate_end_sample]*padding)
                noncate_mask += ([noncate_mask_sample]*padding)

        else:
            selectedHistory_sentencesTokens += ([history_sentencesTokens_padding]*padding) # [historyNum, seqLen] -> [maxHistoryNum, seqLen]
            selectedHistory_attentionMask += ([history_attentionMask_padding]*padding)
            selectedHistory_segmentEmbedding += ([history_segmentEmbedding_padding]*padding)
            selectedHistory_slotPosition += ([history_slotPosition_padding]*padding)
            selectedHistory_mask_sentences += ([history_mask_sentences_padding]*padding)
            if buildSupervise:
                cata_target += ([cata_target_sample_padding]*padding)
                cate_mask += ([cate_mask_sample_padding]*padding)
                noncate_start += ([noncate_start_sample_padding]*padding)
                noncate_end += ([noncate_end_sample_padding]*padding)
                noncate_mask += ([noncate_mask_sample_padding]*padding)

        list_selectedHistory_sentencesTokens.append(selectedHistory_sentencesTokens)
        list_selectedHistory_attentionMask.append(selectedHistory_attentionMask)
        list_selectedHistory_segmentEmbedding.append(selectedHistory_segmentEmbedding)
        list_selectedHistory_slotPosition.append(selectedHistory_slotPosition)
        list_selectedHistory_mask_sentences.append(selectedHistory_mask_sentences)
        if buildSupervise:
            list_cata_target.append(cata_target)
            list_cate_mask.append(cate_mask)
            list_noncate_start.append(noncate_start)
            list_noncate_end.append(noncate_end)
            list_noncate_mask.append(noncate_mask)

        list_last_state.append(data[5])
        list_current_state.append(data[6])

    # 我们要更新那些slot
    update_slot = torch.tensor(list_update_slot, dtype=torch.long) # [batchSize, slotType] 0继承 1更新
    # 当前轮信息
    current_sentencesTokens = torch.tensor(list_current_sentencesTokens, dtype=torch.long) # [batchSize, seqLen] 当前轮对话token
    current_attentionMask = torch.tensor(list_current_attentionMask, dtype=torch.long) # [batchSize, seqLen] 当前轮对话token
    current_segmentEmbedding = torch.tensor(list_current_segmentEmbedding, dtype=torch.long) # [batchSize, seqLen] 当前轮对话token
    current_slotPosition = torch.tensor(list_current_slotPosition, dtype=torch.long) # [batchSize, slotType] 当前轮对话 [slot] 在哪
    current_valuePosition = torch.tensor(list_current_valuePosition, dtype=torch.long) # [batchSize, slotType] 当前轮对话 [value] 在哪

    # 对话选择信息
    history_sentencesTokens = torch.tensor(list_history_sentencesTokens, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 历史对话token
    history_attentionMask = torch.tensor(list_history_attentionMask, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 历史对话token
    history_segmentEmbedding = torch.tensor(list_history_segmentEmbedding, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 历史对话token
    history_mask_sentences = torch.tensor(list_history_mask_sentences, dtype=torch.long) # [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
    updateSlot2current = torch.tensor(list_updateSlot2current, dtype=torch.float32) #  [batchSize, slotType, 1+max_historyNum] 更新槽与当前轮    需要保证padiing出来的history 不与任何点相连
    updateSlot2allSlot = torch.tensor(list_updateSlot2allSlot, dtype=torch.float32) # [batchSize, slotType, SlotType] 本轮要更新的槽链接所有的槽
    slot2lastUpdateTurn = torch.tensor(list_slot2lastUpdateTurn, dtype=torch.float32) # [batchSize, slotType, 1+max_historyNum] 槽与最近更新轮    需要保证padiing出来的history 不与任何点相连
    slot_domain_connect = torch.tensor(list_slot_domain_connect, dtype=torch.float32) # [batchSize, slotType, SlotType] 同domain的槽相连

    # 槽值生成信息
    selectedHistory_sentencesTokens = torch.tensor(list_selectedHistory_sentencesTokens, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    selectedHistory_attentionMask = torch.tensor(list_selectedHistory_attentionMask, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    selectedHistory_segmentEmbedding = torch.tensor(list_selectedHistory_segmentEmbedding, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 拼接了当前轮的历史对话token
    selectedHistory_slotPosition = torch.tensor(list_selectedHistory_slotPosition, dtype=torch.long) # [batchSize, maxHistoryNum, slotType] [slot]的所在位置
    selectedHistory_mask_sentences = torch.tensor(list_selectedHistory_mask_sentences, dtype=torch.long) # [batchSize, maxHistoryNum, seqLen] 句子的地方填写1, slot的地方填写0, padding的地方写0


    # 监督信号
    cata_target = torch.tensor(list_cata_target, dtype=torch.long) # 分类的监督信号 [batchSize, maxHistoryNum, slotType] 0表示删除槽值 其他表示更新槽值
    cate_mask = torch.tensor(list_cate_mask, dtype=torch.long) # 0表示不需要监督 1 表示需要监督 [batchSize, maxHistoryNum, slotType]
    noncate_start = torch.tensor(list_noncate_start, dtype=torch.long) # 抽取开始位置得监督信号 [batchSize, maxHistoryNum, slotType]
    noncate_end = torch.tensor(list_noncate_end, dtype=torch.long) # 抽取结束位置得监督信号 [batchSize, maxHistoryNum, slotType]
    noncate_mask = torch.tensor(list_noncate_mask, dtype=torch.long) # 0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到[batchSize, maxHistoryNum, slotType]


    return update_slot, current_sentencesTokens, current_attentionMask, current_segmentEmbedding, current_slotPosition, current_valuePosition,\
            history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, \
            updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, \
            selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding, selectedHistory_slotPosition, selectedHistory_mask_sentences, \
            cata_target, cate_mask, noncate_start, noncate_end, noncate_mask, \
            list_last_state, list_current_state












def loadDataSet_DiCoS(config:DiCoSConfig, isTrain:bool, schema:MultiWoZSchema):
    return DataLoader(
        dataset = MultiWoZDataset(config, isTrain, schema),
        batch_size = config.batch_size,
        shuffle = config.shuffle,
        drop_last = config.drop_last,
        collate_fn = lambda datas: dicos_collate_fn(datas, config),
        num_workers = config.num_workers
    )
        
if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    schema = MultiWoZSchema(config)
    test = loadDataSet_DiCoS(config, False, schema)

    for t in test:
        print(t)




    