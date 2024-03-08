import sys
sys.path.append('./')

import json
from typing import Tuple
from utils.multiWoZUtils import MultiWoZSchema, getDialogueTurnToken, getStateToken
from utils.DiCoSUtils.DiCoSMultiWoZDataUtils import SentencesInstences
from utils.DiCoSUtils.DSSDataLoader import dss_collate_fn
from utils.DiCoSUtils.DiCoSDataLoader import dicos_collate_fn
from configs.DiCoSConfig import DiCoSConfig
from models.DiCoS import DSS, DiCoS
from tqdm import tqdm
import torch
import numpy as np
import difflib
import random

def to_device(params:Tuple[torch.Tensor], config:DiCoSConfig):
    list_params_device = []
    for param in params:
        if type(param) == torch.Tensor:
            list_params_device.append(param.to(config.device))
        else:
            list_params_device.append(param)
    return list_params_device


def update_dict_slot_updateTurn(dict_update:dict, dict_slot_updateTurn:dict, currentTurnNo:int):
    '''
            dict_update:dict, 
            {
                slotName: update,
                slotName: inhert,
                ...
            }
            
            dict_slot_updateTurn:dict
            {
                slotName: -1, 
                slotName: 2,
                ....
            }

            return: 
            {
                slotName: -1, 
                slotName: 2,
                ....
            }
        
    '''
    for slotName, value in dict_update.items():
        if value == 'update': # 如果在本轮更新则更新其更新轮次信息
            dict_slot_updateTurn[slotName] = currentTurnNo

    return dict_slot_updateTurn


def getPredState(dict_update:dict, last_state:dict, startProb:torch.Tensor, endProb:torch.Tensor, slotValueProb:torch.Tensor, selectedHistory_sentencesTokens:torch.Tensor, config:DiCoSConfig, schema:MultiWoZSchema) -> dict:
    '''
        dict_update:dict, 
        last_state:dict, 
        startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
        endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
        slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果
        selectedHistory_sentencesTokens [updateSlotNum, seqLen], 从哪一句话中抽取出来的
        config:DiCoSConfig, 
        schema:MultiWoZSchema
    
    
    
    '''
    startProb = startProb.masked_fill(startProb==0, -1e9)
    endProb = endProb.masked_fill(startProb==0, -1e9) # 加保护 防止其他句子位置的概率值太小 从而与slot区域区分不开
    ext_mask = torch.ones([startProb.shape[0], config.slotTypeNum, config.pad_size, config.pad_size], dtype=torch.int8, device=config.device) # [updateSlotNum, slotType, start, end] 1为合法
    ext_mask = torch.triu(ext_mask, 0)
    probNorm = torch.unsqueeze(startProb, -1) + torch.unsqueeze(endProb, -2) # 概率相加 [updateSlotNum, slotType, start, end]
    allSpanProb = torch.masked_fill(probNorm, ext_mask==0, 0) # 将不合法的位置全部归0  [updateSlotNum, slotType, start, end]

    updateIdx = 0 # 记录是第几个update
    for slotName, option in dict_update.items():
        if option == 'update': # 需要更新的槽
            slotIdx = schema.slotName2slotIdx[slotName]
            spansProb:np.ndarray = allSpanProb[updateIdx][slotIdx].cpu().detach().numpy() # [start, end]
            index = np.unravel_index(spansProb.argmax(), spansProb.shape)	# 最大值索引
            
            extValue = config.tokenizer.decode(selectedHistory_sentencesTokens[updateIdx].cpu().detach().tolist()[index[0]:index[1]+1])

            if schema.isCatagorical(slotName): # 如果是分类槽
                possibleValue = schema.catagorical[slotName]
                if extValue in possibleValue and extValue != 'dontcare' and extValue != '[NONE]': # 抽取的是可能的值
                    last_state[slotName] = extValue 
                else: # 抽取的不可用用分类的
                    _, clsIdx = slotValueProb[updateIdx][slotIdx].max(-1)
                    last_state[slotName] = possibleValue[clsIdx] # 取出分类结果

            else: # 抽取槽
                last_state[slotName] = extValue
            updateIdx += 1

    current_state = last_state
    return current_state

@torch.no_grad()
def eval_multiWoZ(dss:DSS, dicos:DiCoS, config:DiCoSConfig, schema:MultiWoZSchema):

    dss.eval()
    dicos.eval()
    list_data_save = []

    # 加载数据
    with open(config.test_path, 'r', encoding='utf-8') as in_:
        list_dialogue = json.load(in_)
    
    list_isRight = [] # 1 是正确 0 是错误
    random.shuffle(list_dialogue)
    for dialogueIdx, dict_dialogue in tqdm(enumerate(list_dialogue[:50]), 'test'):
        '''
            {
                "dialogue_idx": "MUL0012.json",
                "domains": [],
                "dialogue": []
            }
                
        '''
        last_state = schema.init_state() # 初始化状态字典
        gt_last_state = schema.init_state() # 初始化状态字典
        list_sentencesTokens_history = [] #  只存储句子的token 用于对话选择后的SVG的输入构建
        dict_slot_updateTurn = {slotName:-1 for slotName in schema.list_slotName} # 存储槽最近一次更新是在哪一个历史轮（不包括当前轮更新的槽） -1表示从未被更新过
        for dialogueTurn in dict_dialogue['dialogue']:
            '''
                {
                    "system_transcript": "",
                    "turn_idx": 0,
                    "belief_state": [],
                    "turn_label": [],
                    "transcript": "i need information on a hotel that include -s free parking please .",
                    "system_acts": [],
                    "domain": "hotel"
                }
            '''
            turnid = dict_dialogue['dialogue_idx'] + '_' + str(dialogueTurn['turn_idx'])
            gt_current_state = schema.build_state(dialogueTurn['belief_state'])
            gt_dict_update, gt_dict_cate_true, gt_dict_nocate_true = schema.getUpdateInfo(gt_last_state, gt_current_state)

            currentTurn_dialogue_token = getDialogueTurnToken(config.tokenizer, dialogueTurn) # 构造语句的 token
            lastTurn_state_token = getStateToken(config.tokenizer, last_state) # 构造 state 的 token

            # dss 输入
            currentTurn_tokens = [config.tokenizer.cls_token] + lastTurn_state_token + [config.tokenizer.additional_special_tokens[2]] + [config.tokenizer.additional_special_tokens[3]] + currentTurn_dialogue_token # 当前轮的输入构造 
            current_turn = SentencesInstences(currentTurn_tokens, config, schema, None, None, None)
            data = [ # 封装一个样本我们需要用到的所有内容
                    turnid, # 当前轮的id
                    current_turn, # 当前轮的输入
                    None, # 用于对话选择过程
                    None, # 用于选择好历史对话之后的SVG阶段
                    None, # 存储了四种边
                    None,# 上一阶段的状态真值 用于验证acc
                    None # 当前阶段的状态真值 用于验证acc
            ]
            params = dss_collate_fn([data], config, False)
            currentTurn_tokens_idx, currentTurn_attentionMask, currentTurn_segmentEmbedding, currentTurn_slotTokenPosition, currentTurn_sentencesMask, _, _, _, _, _, _ = to_device(params, config)
            
            # 哪些槽被更新
            prob_update, _, _, _, _, _, _ = dss.forward(currentTurn_tokens_idx, currentTurn_attentionMask, currentTurn_segmentEmbedding, currentTurn_slotTokenPosition, currentTurn_sentencesMask)
            _, is_update = torch.max(prob_update, dim=-1) # [1, slotType]
            dict_update = {}
            update_idx2str = list(config.dict_update.keys())
            for slotName, sup in zip(schema.list_slotName, is_update.cpu().detach().tolist()[0]):
                dict_update[slotName] = update_idx2str[sup]

            # gt sop 的情况
            # dict_update = gt_dict_update
            # 建图
            '''
                建图: slot 的 顺序 按照 self.schema.list_slotName
                      history 的 顺序 按照 当前轮+历史轮+padding
            '''
            # updateSlot2current [slotType] 哪些槽在当前轮更新了
            updateSlot2current = [0 for slotName in schema.list_slotName]
            for slotName, value in dict_update.items():
                if value == 'update':
                    updateSlot2current[schema.slotName2slotIdx[slotName]] = 1

            # updateSlot2allSlot [slotType, slotType] 当前轮更新的槽于所有的槽相连
            updateSlot2allSlot = [[0 for slotName in schema.list_slotName] for slotName in schema.list_slotName]
            for slotName, value in dict_update.items():
                if value == 'update': 
                    for i in range(len(schema.list_slotName)):
                        updateSlot2allSlot[i][schema.slotName2slotIdx[slotName]] = 1

                    for i in range(len(schema.list_slotName)):
                        updateSlot2allSlot[schema.slotName2slotIdx[slotName]][i] = 1 # 对称矩阵

            # slot2lastUpdateTurn [slotType, historyNum] 槽在哪些历史轮更新了
            slot2lastUpdateTurn = [[0 for i in range(len(list_sentencesTokens_history))] for slotName in schema.list_slotName]
            for slotName, updateTurn in dict_slot_updateTurn.items():
                if updateTurn != -1: # 被更新过, updateTurn 可以取的最大值为 len(list_sentencesTokens_history)-1
                    slot2lastUpdateTurn[schema.slotName2slotIdx[slotName]][updateTurn] = 1 # 建立一个边
                
            # slot_domain_connect [slotType, SlotType]
            slot_domain_connect = [[0 for slotName in schema.list_slotName] for slotName in schema.list_slotName]
            for slotName_row in schema.list_slotName:
                for slotName_colum in schema.list_slotName:
                    if slotName_row.split('-')[0] == slotName_colum.split('-')[0]:
                        slot_domain_connect[schema.slotName2slotIdx[slotName_row]][schema.slotName2slotIdx[slotName_colum]] = 1

            graph = [updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect]
            # 更新槽的 最近更新轮次 信息
            dict_slot_updateTurn = update_dict_slot_updateTurn(dict_update, dict_slot_updateTurn, dialogueTurn['turn_idx'])

            '''
                构建对话选择所用的历史轮
            '''
            list_sentencesInstence_history = []
            for history_dialogue_token in list_sentencesTokens_history:
                history_token = [config.tokenizer.cls_token] + lastTurn_state_token + [config.tokenizer.additional_special_tokens[2]] + [config.tokenizer.additional_special_tokens[3]] + history_dialogue_token # 历史轮的输入构造
                list_sentencesInstence_history.append(SentencesInstences(history_token, config, schema, dict_update, None, None))

            '''
                构建包含选择的历史轮的输入
            '''
            list_select_history = []
            for sentencesTokens_history in list_sentencesTokens_history:
                tokens_withHistory = currentTurn_tokens + sentencesTokens_history # 当前轮的token拼接上历史对话
                list_select_history.append(SentencesInstences(tokens_withHistory, config, schema, dict_update, None, None)) # 构造输入
            
            # 构造dicos的输入
            data = [ # 封装一个样本我们需要用到的所有内容
                    turnid, # 当前轮的id
                    current_turn, # 当前轮的输入
                    list_sentencesInstence_history, # 用于对话选择过程
                    list_select_history, # 用于选择好历史对话之后的SVG阶段
                    graph, # 存储了四种边
                    last_state,# 上一阶段的状态真值 用于验证acc
                    gt_current_state # 当前阶段的状态真值 用于验证acc
            ]
            params = dicos_collate_fn([data], config, False)
            _, current_sentencesTokens, current_attentionMask, current_segmentEmbedding, current_slotPosition, current_valuePosition,\
            history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, \
            updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, \
            selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding, selectedHistory_slotPosition, selectedHistory_mask_sentences, \
            _, _, _, _, _, \
            _, _ = to_device(params, config)

            _, _, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens = dicos.forward(current_sentencesTokens, current_attentionMask, current_segmentEmbedding, 
                                                                                    current_slotPosition, current_valuePosition,  # 当前轮信息
                                                                                    history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, # 历史轮信息
                                                                                    updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, # 四个边
                                                                                    selectedHistory_sentencesTokens,selectedHistory_attentionMask, selectedHistory_segmentEmbedding, 
                                                                                    selectedHistory_slotPosition, selectedHistory_mask_sentences)# 当前轮+·被选择对话的信息
            # 获取 state
            current_state = getPredState(dict_update, last_state, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens, config, schema)
            
            # # dss下准确率上限
            # current_state = last_state.copy()
            # for slotName, option in dict_update.items():
            #     if option == 'update':
            #         current_state[slotName] = gt_current_state[slotName]
            
            # 判断是否正确
            isright = 1
            for trueState, predState in zip(list(gt_current_state.values()), list(current_state.values())):
                if difflib.SequenceMatcher(None, trueState, predState).quick_ratio() < 0.85:
                    isright = 0
                    break
            if isright == 1 and random.random()>0.5:
                dialogueTurn['pred'] = current_state
                list_data_save.append(dialogueTurn)

                    
            
            list_isRight.append(isright)

            list_sentencesTokens_history.append(currentTurn_dialogue_token) # 仅本轮对话保存在历史信息中，用于之后的输入构建
            # slot 状态转换
            last_state = current_state
            gt_last_state = gt_current_state

    with open('./result.json', 'w', encoding='utf-8') as out_:
        json.dump(list_data_save, out_, ensure_ascii=False, indent=4)

    return np.array(list_isRight).sum()/len(list_isRight)



if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    schema = MultiWoZSchema(config)

    dss = DSS(config, schema)
    dss.load_state_dict(torch.load('./savedModels/DSS/2022-05-24-16-45-58.pth', map_location='cpu'))

    dicos = DiCoS(config, schema)
    dicos.load_state_dict(torch.load('./savedModels/DiCoS/2022-05-24-00-06-39.pth', map_location='cpu'))
    
    jga = eval(dss, dicos, config, schema)

    print(jga)
