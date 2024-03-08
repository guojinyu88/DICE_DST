import sys
from typing import List
sys.path.append('./')

from torch.utils.data import Dataset
from configs.DiCoSConfig import DiCoSConfig
from tqdm import tqdm
import json
from utils.multiWoZUtils import getDialogueTurnToken, getStateToken, getUpdataInPred, MultiWoZSchema, getPositionOfAnswer


class MultiWoZDataset(Dataset):
    '''
        gt的预处理程序，其中所有的轮次转换信息（dict_update, dict_cate_true, dict_nocate_true）都使用gt来构建
    '''
    def __init__(self, config:DiCoSConfig, isTrain:bool, schema:MultiWoZSchema) -> None:
        super().__init__()
        self.config = config
        self.isTrain = isTrain
        self.schema = schema
        self.datas = self.loadDataset()

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)

    def loadDataset(self) -> list:

        datasetFilePath = self.config.train_path if self.isTrain else self.config.test_path

        with open(datasetFilePath, 'r', encoding='utf-8') as in_: # 读取数据集
            list_dialogue = json.load(in_)

        list_datas = []
        for dialogueIdx, dict_dialogue in tqdm(enumerate(list_dialogue), 'load dataset '):
            '''
                {
                    "dialogue_idx": "MUL0012.json",
                    "domains": [],
                    "dialogue": []
                }
                
            '''
            last_state = self.schema.init_state() # 初始化状态字典
            list_sentencesTokens_history = [] #  只存储句子的token 用于对话选择后的SVG的输入构建
            dict_slot_updateTurn = {slotName:-1 for slotName in self.schema.list_slotName} # 存储槽最近一次更新是在哪一个历史轮（不包括当前轮更新的槽） -1表示从未被更新过
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

                current_state = self.schema.build_state(dialogueTurn['belief_state'])
                dict_update, dict_cate_true, dict_nocate_true = self.schema.getUpdateInfo(last_state, current_state)

                '''
                    构造当前轮的输入
                '''
                token_turn = getDialogueTurnToken(self.config.tokenizer, dialogueTurn) # 构造语句的 token
                token_state = getStateToken(self.config.tokenizer, last_state) # 构造 state 的 token

                tokens = [self.config.tokenizer.cls_token] + token_state + [self.config.tokenizer.additional_special_tokens[2]] + [self.config.tokenizer.additional_special_tokens[3]] + token_turn # 当前轮的输入构造 

                assert len(tokens) <= self.config.pad_size # 确保仅仅只有当前轮的情况下，不会做截断
                current_turn = SentencesInstences(tokens, self.config, self.schema, dict_update, dict_cate_true, dict_nocate_true) # 给定 tokens 构造我们需要的各种mask


                
                '''
                    建图: slot 的 顺序 按照 self.schema.list_slotName
                          history 的 顺序 按照 当前轮+历史轮+padding
                '''
                # updateSlot2current [slotType] 哪些槽在当前轮更新了
                updateSlot2current = [0 for slotName in self.schema.list_slotName]
                for slotName, value in dict_update.items():
                    if value == 'update':
                        updateSlot2current[self.schema.slotName2slotIdx[slotName]] = 1

                # updateSlot2allSlot [slotType, slotType] 当前轮更新的槽于所有的槽相连
                updateSlot2allSlot = [[0 for slotName in self.schema.list_slotName] for slotName in self.schema.list_slotName]
                for slotName, value in dict_update.items():
                    if value == 'update': 
                        for i in range(len(self.schema.list_slotName)):
                            updateSlot2allSlot[i][self.schema.slotName2slotIdx[slotName]] = 1

                        for i in range(len(self.schema.list_slotName)):
                            updateSlot2allSlot[self.schema.slotName2slotIdx[slotName]][i] = 1 # 对称矩阵

                # slot2lastUpdateTurn [slotType, historyNum] 槽在哪些历史轮更新了
                slot2lastUpdateTurn = [[0 for i in range(len(list_sentencesTokens_history))] for slotName in self.schema.list_slotName]
                for slotName, updateTurn in dict_slot_updateTurn.items():
                    if updateTurn != -1: # 被更新过, updateTurn 可以取的最大值为 len(list_sentencesTokens_history)-1
                        slot2lastUpdateTurn[self.schema.slotName2slotIdx[slotName]][updateTurn] = 1 # 建立一个边
                
                # slot_domain_connect [slotType, SlotType]
                slot_domain_connect = [[0 for slotName in self.schema.list_slotName] for slotName in self.schema.list_slotName]
                for slotName_row in self.schema.list_slotName:
                    for slotName_colum in self.schema.list_slotName:
                        if slotName_row.split('-')[0] == slotName_colum.split('-')[0]:
                            slot_domain_connect[self.schema.slotName2slotIdx[slotName_row]][self.schema.slotName2slotIdx[slotName_colum]] = 1

                graph = [updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect]
                '''
                    构建对话选择所用的历史轮
                '''
                list_sentencesInstence_history = []

                for sentencesTokens_history in list_sentencesTokens_history:
                    history_token = [self.config.tokenizer.cls_token] + token_state + [self.config.tokenizer.additional_special_tokens[2]] + [self.config.tokenizer.additional_special_tokens[3]] + sentencesTokens_history # 历史轮的输入构造
                    list_sentencesInstence_history.append(SentencesInstences(history_token, self.config, self.schema, dict_update, dict_cate_true, dict_nocate_true))
                
                '''
                    构建包含选择的历史轮的输入,以及相应的监督信号
                '''
                list_select_history = []
                
                for sentencesTokens_history in list_sentencesTokens_history:
                    tokens_withHistory = tokens + sentencesTokens_history # 当前轮的token拼接上历史对话
                    list_select_history.append(SentencesInstences(tokens_withHistory, self.config, self.schema, dict_update, dict_cate_true, dict_nocate_true)) # 构造输入与监督信号
                
                
                data = [ # 封装一个样本我们需要用到的所有内容
                    turnid, # 当前轮的id
                    current_turn, # 当前轮的输入
                    list_sentencesInstence_history, # 用于对话选择过程
                    list_select_history, # 用于选择好历史对话之后的SVG阶段
                    graph, # 存储了四种边
                    last_state,# 上一阶段的状态真值 用于验证acc
                    current_state # 当前阶段的状态真值 用于验证acc
                ]

                list_datas.append(data) # 当前句子构造完毕

                # 轮次转换 

                list_sentencesTokens_history.append(token_turn) # 仅本轮对话保存在历史信息中，用于之后的输入构建
                # 更新记录slot更新时机的dict
                dict_slot_updateTurn = self.update_dict_slot_updateTurn(dict_update, dict_slot_updateTurn, dialogueTurn['turn_idx'])
                # slot 状态转换
                last_state = current_state
        
        return list_datas

    def update_dict_slot_updateTurn(self, dict_update:dict, dict_slot_updateTurn:dict, currentTurnNo:int):
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

    
class SentencesInstences():
    '''
        封装一轮对话token的所有东西, 包括监督信号+模型的输入
    '''
    def __init__(self, tokens:list, config:DiCoSConfig, schema:MultiWoZSchema, dict_update, dict_cate_true:dict, dict_nocate_true:dict) -> None:
        '''
        
        
            dict_cate:
                none 则说明不用构建监督信号 str 则说明需要构建监督信号
                    {
                        slotName: value,
                        slotName: value,
                        ...
                    }

            dict_nocate:
                none 则说明不用构建监督信号 str 则说明需要构建监督信号
                    {
                        slotName: value,
                        slotName: value,
                        ...
                    }
        '''
        self.config = config
        self.schema = schema
        self.dict_update = dict_update # 保存槽值受否需要更新的监督信号
        self.dict_cate_true = dict_cate_true # 保存分类需要监督的slot
        self.dict_nocate_true = dict_nocate_true # 保存抽取需要监督的slot
        self.slotTokenPosition = [position for position, token in enumerate(tokens) if token == '[SLOT]'] # 拿到 [SLOT] 的位置 [slotType]
        self.valueTokenPosition = [position for position, token in enumerate(tokens) if token == '[VALUE]'] # 拿到 [VALUE] 的位置 [slotType]
        self.tokens = tokens # 转化为idx之前
        self.tokens_idx = None # 转化为idx 并 padiing
        self.attentionMask = None # 对应的attention
        self.segmentEmbedding = None # 对应的segmentEmbedding
        self.sentencesMask = None # 哪些地方是句子 1 是句子  0 是其他的

        # 监督信号
        self.supervised_update = None # 是否更新的监督信号
        self.supervised_cata = None # cata的监督信号，none 为不需要监督
        self.supervised_noncate = None # nonecate的监督信号，(none,none)为不需要监督，(-1,-1) 为需要监督 但是找不到这个词 对话轮选择出错


        self.__doPadding(config.tokenizer.convert_tokens_to_ids(tokens))
        self.__buildSegmentEmbedding()
        self.__buildSentencesMask()
        if self.dict_cate_true != None and self.dict_nocate_true != None: # 如果不提供gt则不构造监督信号，用于推理阶段
            self.__buildSupervisedSignal()

    def __doPadding(self, tokens_idx):
        '''
            进行padding, 输出出来attentionMask
            padding 的地方是0，其余地方是1
        '''
        attentionMask = [1] * len(tokens_idx)
        if len(tokens_idx) <= self.config.pad_size: # 做padding
            pad_len = self.config.pad_size-len(tokens_idx)
            tokens_idx += ([self.config.tokenizer.pad_token_id] * pad_len)
            attentionMask += ([0] * pad_len)
        else: # 做截断
            # assert False # 其实不允许做截断
            tokens_idx = tokens_idx[:self.config.pad_size]
            attentionMask = attentionMask[:self.config.pad_size]

        self.tokens_idx = tokens_idx
        self.attentionMask = attentionMask

    def __buildSegmentEmbedding(self):
        '''
            slot是 0，sentences 和 padding 是 1
        '''
        self.segmentEmbedding = []

        beforeSEP = True
        for token_idx in self.tokens_idx:
            if beforeSEP: # sep 之前放 0
                self.segmentEmbedding.append(0)
            else: # sep 之后放 1
                self.segmentEmbedding.append(1)

            if token_idx == self.config.tokenizer.sep_token_id: 
                beforeSEP = False
        
        assert len(self.segmentEmbedding) == self.config.pad_size # 保证长度正确

    def __buildSentencesMask(self):
        '''
            句子的部分是1，其余部分是0
        '''
        self.sentencesMask = []
        for atten, seg in zip(self.attentionMask, self.segmentEmbedding):
            if atten+seg == 2:
                self.sentencesMask.append(1)
            else:
                self.sentencesMask.append(0)

    def __buildSupervisedSignal(self):
        '''
            根据这些tokens构造监督信号

            cata的监督信号，none 为不需要监督 (继承或者抽取槽)
            nonecate的监督信号，(none,none)为不需要监督（继承+分类槽抽不到），(-1,-1) 为需要监督（抽取槽） 但是找不到这个词
        '''
        # 构建更新监督信号
        self.supervised_update = []
        for slotName in self.schema.list_slotName:
            if self.dict_update[slotName] == 'update':
                self.supervised_update.append(self.config.dict_update['update'])
            else:
                self.supervised_update.append(self.config.dict_update['inherit'])

        # 构建分类监督信号
        self.supervised_cata = []
        for slotName in self.schema.list_slotName:
            if self.dict_cate_true[slotName] == None: # 不需要构建监督信号 (继承、抽取槽)
                self.supervised_cata.append(None)
            else:
                self.supervised_cata.append(self.schema.getCatagoricalIdx(slotName, self.dict_cate_true[slotName]))

        # 构建抽取监督信号
        self.supervised_noncate = []
        for slotName in self.schema.list_slotName:
            if self.dict_nocate_true[slotName] == None: # 不需要监督信号 (继承)
                self.supervised_noncate.append((None, None))
                continue # 下一个slot
            answerTokens_tokens = self.config.tokenizer.tokenize(self.dict_nocate_true[slotName]) # 分词
            answerTokens_idx = self.config.tokenizer.convert_tokens_to_ids(answerTokens_tokens) # 转化为idx
            startIdx, endIdx = getPositionOfAnswer(self.tokens_idx, self.sentencesMask, answerTokens_idx)
            
            if self.schema.isCatagorical(slotName): # 分类槽
                if startIdx == -1:
                    self.supervised_noncate.append((None, None)) # 分类槽抽取不到 就不监督了
                else:
                    self.supervised_noncate.append((startIdx, endIdx)) # 分类槽抽取的到 就监督

            else: # 抽取槽
                self.supervised_noncate.append((startIdx, endIdx))


if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    schema = MultiWoZSchema(config)

    data = MultiWoZDataset(config, True, schema)