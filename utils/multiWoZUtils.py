import sys
sys.path.append('./')

import json
import torch
from configs.DiCoSConfig import DiCoSConfig
from transformers import AlbertTokenizer
import numpy as np


class MultiWoZSchema():
    '''
        真实和数据集文件中的数据结构打交道的地方，
        负责各种数据过滤和修正工作
    '''
    def __init__(self, config:DiCoSConfig) -> None:
        self.config = config

        self.list_slotName = config.track_slots # 记录所有关心的slot名 slotIdx 映射到 slotName
        self.slotName2slotIdx = {} # slotName 映射到 slotIdx

        self.catagorical = {} # 记录分类的槽并且记录其可能的槽值
        self.classifierMerge = [] # [slotType] 记录每个slot用哪一个分类器
        self.classifier = [] # [cataSlotNum] 记录一共存在多少种分类器
        self.catagorical_str2idx = {} # 将分类的 slotValue 映射到 idx上
        self.noneCatagorical = [] # 记录非分类的槽

        self.mask_cateSlot = None # [slotType] 1 为 分类
        self.mask_cate_value = None # [slotType, maxValueNum] 1 为可以分类的slot value

        self.__loadSchema() # 构造slot信息
        if config.mergeClassifier: # 只有在merge的情况下才对分类信息进行标准化
            self.__buildClassifierMerge() # 构建分类器融合所需要的信息，保证共用一个分类器的slot拥有相同的idx
        self.__buildMaskForCata() # 构造mask，并对分类slot进行padding

    def __loadSchema(self):
        '''
            从文件中读取出来schema信息
        '''
        with open(self.config.schema_path, 'r', encoding='utf-8') as in_:
            list_service = json.load(in_)

        for service in list_service:
            '''
                {
                    "service_name": "hotel",
                    "slots": [],
                    "description": "hotel reservations and vacation stays",
                    "intents": []
                }
            '''
            if service['service_name'] not in self.config.domain: # 跳过我们不关注的domain
                continue
            for slot in service['slots']:
                '''
                    {
                        "name": "hotel-pricerange",
                        "description": "price budget of the hotel",
                        "possible_values": [
                            "expensive",
                            "cheap",
                            "moderate"
                        ],
                        "is_categorical": true
                    }
                '''
                if slot['name'] in self.list_slotName:
                    if slot['is_categorical']:
                        self.catagorical[slot['name']] = slot['possible_values'] 
                    else:
                        self.noneCatagorical.append(slot['name'])

        # 构建 slotName 映射到 slotIdx
        for i, slotName in enumerate(self.list_slotName):
            self.slotName2slotIdx[slotName] = i

    def __buildMaskForCata(self):
        '''
            构建 分类slot value 的时候要使用的 mask
        '''
        # 构造分类mask
        mask_cateSlot = []
        for slotName in self.list_slotName:
            if slotName in list(self.catagorical.keys()):
                mask_cateSlot.append(1)
            else:
                mask_cateSlot.append(0)
        self.mask_cateSlot = torch.tensor(mask_cateSlot, dtype=torch.int8, device=self.config.device)

        # padding分类 同时 mask
        mask_cate_value = []
        for slotName in self.list_slotName:
            cate_value = []
            if slotName in list(self.catagorical.keys()): # 处理分类槽
                self.catagorical[slotName] = ['[NONE]'] + self.catagorical[slotName] + ['dontcare'] # 拼接上通用的槽值
                assert self.config.maxSlotValue >= len(self.catagorical[slotName]) # 确保设置的maxvalue够用
                cate_value += [1] * len(self.catagorical[slotName]) # 可以分类的地方为1
                cate_value += [0] * (self.config.maxSlotValue - len(self.catagorical[slotName]))
            else: # 非分类的槽
                cate_value += [0] * self.config.maxSlotValue
            
            mask_cate_value.append(cate_value)
        self.mask_cate_value = torch.tensor(mask_cate_value, dtype=torch.int8, device=self.config.device)

        for slotName, list_slotValue in self.catagorical.items():
            self.catagorical_str2idx[slotName] = {}
            for idx, slotValue in enumerate(list_slotValue):
                self.catagorical_str2idx[slotName][slotValue] = idx

    def __buildClassifierMerge(self):
        
        
        self.classifier = [
            ['expensive', 'cheap', 'moderate'],
            ['guesthouse', 'hotel'],
            ['free', 'no', 'yes'],
            ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '15'],
            ['centre', 'east', 'north', 'south', 'west'],
            ['birmingham new street', 'bishops stortford', 'broxbourne', 'cambridge', 'ely', 'kings lynn', 'leicester', 'london kings cross', 'london liverpool street', 'norwich', 'peterborough', 'stansted airport', 'stevenage'],
            ['architecture', 'boat', 'cinema', 'college', 'concerthall', 'entertainment', 'museum', 'multiple sports', 'nightclub', 'park', 'swimmingpool', 'theatre'],
            []
        ]
        for slotName in self.list_slotName:
            if self.isCatagorical(slotName):
                cataValues = self.catagorical[slotName]
                for idx, list_labels in enumerate(self.classifier):
                    isContain = True # 当前 list_labels 可以包含 cataValues
                    for cataValue in cataValues:
                        if cataValue not in list_labels:
                            isContain = False
                            break
                    if isContain: # 包含则加入
                        self.catagorical[slotName] = list_labels # 规范label顺序
                        self.classifierMerge.append(idx)
            else:
                self.classifierMerge.append(8)

    def isCatagorical(self, slotName:str) -> bool:
        '''
            判断一个槽是不是分类槽
        '''
        assert slotName in self.list_slotName

        return slotName in list(self.catagorical.keys())

    def getCatagoricalIdx(self, slotName:str, slotValue:str)-> int:
        '''
            给定槽和槽值 返回分类idx, 构建监督信号用
            需要将各种不规范的表示映射到规范的slotValue身上
            none 代表不监督
        '''
        try:
            idx = self.catagorical_str2idx[slotName][slotValue]
        except KeyError: # 尝试修正标签
            GENERAL_TYPO = {
                # type
                "guest house":"guesthouse", "guesthouses":"guesthouse", "guest":"guesthouse", "mutiple sports":"multiple sports", 
                "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimmingpool", "concerthall":"concerthall", 
                "concert":"concerthall", "pool":"swimmingpool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
                "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
                # area
                "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
                "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
                "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
                "centre of town":"centre", "cb30aq": "none",
                # price
                "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
                # day
                "next friday":"friday", "monda": "monday", 
                # parking
                "free parking":"free",
                # internet
                "free internet":"free",
                # star
                "4 star":"4", "4 stars":"4", "0 star rarting":"none",
                # others 
                "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
                '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none"
            }

            
            try:
                slotValue = GENERAL_TYPO[slotValue]
                idx = self.catagorical_str2idx[slotName][slotValue]
            except KeyError: # 修正不了的直接变成none 我们就不对他进行监督了
                print('分类slot -> {}:{} 找不到对应的分类标签'.format(slotName, slotValue))
                idx = None

        return idx







    def init_state(self) -> dict:
        '''
            构建一个空的slot:value字典
            顺序按照 list_slotName
        '''
        dict_slot_value = {}

        for slotName in self.list_slotName:
            dict_slot_value[slotName] = ''

        return dict_slot_value
    
    def build_state(self, belief_state:list) -> dict:
        '''
            根据数据集中的数据构造出state
            state的顺序按照 list_slotName

            belief_state:
            [
                {
                    "slots": [
                        [
                            "taxi-arriveby",
                            []
                        ]
                    ],
                    "act": "inform"
                },
            ]

            return:
            {
                slotName: '',
                slotName: ''         ''代表为空
            }
        '''

        def fix_slotName(slotName:str):
            '''
                修正文件中slotName与shema对不上的地方
            '''
            fix = {
                'restaurant-book day': 'restaurant-bookday',
                'restaurant-book people': 'restaurant-bookpeople',
                'restaurant-book time': 'restaurant-booktime',
                'hotel-book day': 'hotel-bookday',
                'hotel-book people': 'hotel-bookpeople',
                'hotel-book stay': 'hotel-bookstay',
                'train-book people': 'train-bookpeople'
            }

            return fix.get(slotName, slotName)

        def fix_dict_slot_value(dict_slot_value):
            '''
                文件中有些地方没有将slot写全，我们需要将他没写全的补全
            '''

        def fix_dict_slot(dict_slot):
            '''
                修复文件中belief_state格式不正确的地方
            
            '''
            
            if dict_slot['slots'][0][1] == 'dontcare':
                dict_slot['slots'][0][1] = ['dontcare']

            # 当多于一个答案的时候，我们选择短的答案作为gt
            if len(dict_slot['slots'][0][1]) > 1:
                trueSlotValue = ' '*1000
                for slotValue in dict_slot['slots'][0][1]:
                    if len(slotValue) < len(trueSlotValue):
                        trueSlotValue = slotValue
                dict_slot['slots'][0][1] = [trueSlotValue]
            

            return dict_slot

        def fix_slotValue(slotValue):
            '''
                修正一些文件中不符合定义的slot value
            '''



        dict_slot_value = {} # 文件中有什么我们读取出来什么
        # 按照 turn_label 中的 顺序构建
        for dict_slot in belief_state:
            '''
                [
                    "taxi-arriveby",
                    []
                ]
            '''
            slotName = fix_slotName(dict_slot['slots'][0][0])
            
            dict_slot = fix_dict_slot(dict_slot)

            assert len(dict_slot['slots'][0][1]) <= 1 # 确保每一个slot都只取一个值,格式不合法
            if dict_slot['slots'][0][1] == []: # 没有值的话
                slotValue = ''
            else: # 有值的话
                slotValue = dict_slot['slots'][0][1][0]
            dict_slot_value[slotName] = slotValue


        dict_slot_value_sort = {} # 按照 list_slotName 中的顺序, 同时过滤掉我们不关心的slot
        for slotName in self.list_slotName:
            try:
                dict_slot_value_sort[slotName] = dict_slot_value[slotName]
            except KeyError: # 他们的文件中slot不全，需要我们进行一些补充
                print('补入{}'.format(slotName))
                print('原有{}'.format(list(dict_slot_value.keys())))
                dict_slot_value_sort[slotName] = ''

        return dict_slot_value_sort

    def getUpdateInfo(self, dict_last_state, dict_current_state) -> dict:
        '''
            按照 list_slotName 顺序, 表述每一个 slot 是否要更新, 分类的值是多少, 非分类的值是多少

            dict_nocate none 表示 继承
            dict_cate none 表示 抽取槽 或者 继承

            return:
            dict_update:
                {
                    slotName: 'update',
                    slotName: 'inherit',
                    ...
                }
            dict_cate:
            none 则说明不用构建监督信号(继承或者抽取槽) str 则说明需要构建监督信号
                {
                    slotName: value,
                    slotName: value,
                    ...
                }

            dict_nocate:
            none 则说明不用构建监督信号（继承） str 则说明需要构建监督信号
                {
                    slotName: value,
                    slotName: value,
                    ...
                }

        '''
        dict_update = {}
        dict_cata = {}
        dict_nocate = {}
        for slotName in self.list_slotName:
            if dict_last_state[slotName] != dict_current_state[slotName]: # 更新的槽
                dict_update[slotName] = 'update'
                if self.isCatagorical(slotName): # 分类槽
                    if dict_current_state[slotName] == '': # 说明之前不为空，现在为空，也就是说要删除这个槽值
                        dict_cata[slotName] = '[NONE]'
                        dict_nocate[slotName] = '[NONE]'
                    else:
                        dict_cata[slotName] = dict_current_state[slotName]
                        dict_nocate[slotName] = dict_current_state[slotName] # 分类槽也可以抽取获得
                else: # 抽取槽
                    if dict_current_state[slotName] == '': # 说明之前不为空，现在为空，也就是说要删除这个槽值
                        dict_cata[slotName] = None # 抽取槽不能分类获得
                        dict_nocate[slotName] = '[NONE]'
                    else:
                        dict_cata[slotName] = None # 抽取槽不能分类获得
                        dict_nocate[slotName] = dict_current_state[slotName]

            else:
                dict_cata[slotName] = None
                dict_nocate[slotName] = None
                dict_update[slotName] = 'inherit'

        return dict_update, dict_cata, dict_nocate


def getDialogueTurnToken(tokenizer:AlbertTokenizer, dialogueTurn:dict) -> list:
    '''
        给定 1 turn 的信息，返回系统回复和用户输入的拼接
        {
            "system_transcript": "",
            "turn_idx": 0,
            "belief_state": [],
            "turn_label": [],
            "transcript": "i need information on a hotel that include -s free parking please .",
            "system_acts": [],
            "domain": "hotel"
        }

        return: R_t + ; + U_t + [SEP]
            
    '''
    text = dialogueTurn['system_transcript'] + ';' + dialogueTurn['transcript']
    token_turn = tokenizer.tokenize(text)
    token_turn =  token_turn + [tokenizer.sep_token]

    return token_turn
        
def getStateToken(tokenizer:AlbertTokenizer, last_state:dict) -> list:
    '''
        给定上一轮得状态，历史状态得拼接
        last_state: {
                slotName': slotValue,
            ...
        }

        return：
        [slot] + slotName + [value] + value + .... + [SEP]
            
    '''
    token_state = []
    for slotName, slotValue in last_state.items():
        token_state +=  [tokenizer.additional_special_tokens[0]] + tokenizer.tokenize(slotName) + [tokenizer.additional_special_tokens[1]] + tokenizer.tokenize(slotValue)
    token_state += [tokenizer.sep_token]
    
    return token_state

def getUpdataInPred(config:DiCoSConfig, schema:MultiWoZSchema, pred_update:dict, turnId:str):
    '''
        从CLS文件的dict中获得 我们应该更新哪些槽值, 顺序按照schema对象中封装的list_slotName的顺序
        pred_update:
        {
            turnId:[
                [0,1],
                [1,0],
                ...
            ]
        }
    '''
    list_slotName = [
            'hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-bookday', 'hotel-bookpeople', 'hotel-bookstay', 'hotel-stars', 'hotel-internet', 'hotel-name', 'hotel-area', 
            'train-arriveby', 'train-departure', 'train-day', 'train-bookpeople', 'train-leaveat', 'train-destination',
            'attraction-area', 'attraction-name', 'attraction-type', 
            'restaurant-pricerange', 'restaurant-area', 'restaurant-food', 'restaurant-name', 'restaurant-bookday','restaurant-bookpeople', 'restaurant-booktime', 
            'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'taxi-arriveby'
    ] # 记录这里面使用的slot顺序

    pred = np.array(pred_update[turnId]).argmax(axis=-1)

    dict_update = {}
    for slotName, label in zip(list_slotName, pred):
        dict_update[slotName] = list(config.dict_update.keys())[label] # 拿到 update 或 inherit
        
    dict_update_sort = {}

    for slotName in schema.list_slotName:
        dict_update_sort[slotName] = dict_update[slotName]


    assert len(dict_update_sort) == 30 # 确定没有少项

    return dict_update_sort

def getPositionOfAnswer(sentencesTokens_idx:list, sentencesMask:list, answerTokens_idx:list):
    '''
        给定输入模型的idx序列、哪里里是句子的标识、需要抽取的slot的idx的序列

        return：
        slot的起始位置  -1, -1 为未找到相应的slot value的位置

    
    '''
    fast = 0
    for slow in range(len(sentencesMask)):
        if sentencesMask[slow] == 0: # 不是句子的地方不允许抽取
            continue
        fast = slow
        index = 0
        while sentencesTokens_idx[fast] == answerTokens_idx[index]:
            fast += 1
            index += 1
            if index >= len(answerTokens_idx): # answer全部对上了
                return slow, fast-1
            

    return -1, -1 # 未找到满足要求的子串

if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')

    schema = MultiWoZSchema(config)
    print(123)


