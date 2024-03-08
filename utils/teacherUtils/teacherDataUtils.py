import sys
sys.path.append('./')

from torch.utils.data import Dataset, DataLoader
from configs.DiCoSConfig import DiCoSConfig
import torch
import numpy as np
from tqdm import tqdm
import json
from utils.multiWoZUtils import getDialogueTurnToken

class MultiWoZ_teacher(Dataset):
    def __init__(self, config:DiCoSConfig, isTrain:bool, isSequential:bool) -> None:
        self.config = config
        self.isTrain = isTrain
        self.isSequential = isSequential
        self.list_datas = self.loadDataset()
 
    def __len__(self) -> int:
        return len(self.list_datas)

    def __getitem__(self, index):
        idx_token_input, attentionMask, str_label = self.list_datas[index]

        sentencesTokens = torch.tensor(idx_token_input, dtype=torch.long)
        attentionMask = torch.tensor(attentionMask, dtype=torch.long)
        if self.isSequential: # 如果是生成Sequential数据集 则这样处理
            label = torch.tensor(self.config.dict_sequential[str_label], dtype=torch.long)
        else: # 如果是生成contextual数据集 则这样处理
            label = torch.tensor(self.config.dict_contextual[str_label], dtype=torch.long)

        dict_tarinSample = {
            "sentencesTokens": sentencesTokens,
            "attentionMask": attentionMask,
            "label": label, # 真值情况
            "is_sequential": self.isSequential # 标记训练样例（因为训练的时候的batch，一半是sequential任务，一半是contextual任务）
        }
        return dict_tarinSample

    def loadDataset(self) -> list:

        def getSequentialSample(list_token_turn):
            '''
                生成一个sequential任务的用例，随机调换顺序
                返回 D_1 + [SEP] + ... + [SEP] token

                list_token_turn:
                    [[R_t + ; + U_t + [SEP]],[R_t + ; + U_t + [SEP]],...]


                return: 
                    [R_t + ; + U_t + [SEP] + R_t + ; + U_t + [SEP]], str
            '''
            changeSequential =  (np.random.rand() < self.config.sequentialChangeProb) # 是否改变Sequential

            # 如果需要随机调换顺序就调整一下 list_trunNo 的顺序
            list_trunNo = np.arange(len(list_token_turn)) # [turnNum_split]
            if changeSequential:
                change = np.random.randint(0, len(list_token_turn), size=[2])
                while change[0] == change[1]: # 必须生成两个不同的数
                    change = np.random.randint(0, len(list_token_turn), size=[2])
                
                # 互换 No.
                list_trunNo[change[0]] = change[1]
                list_trunNo[change[1]] = change[0]

            # 拼接生成最终的输入
            token_input = []
            for turnNo in list_trunNo:
                token_input += list_token_turn[turnNo]

            # 生成 str_label
            str_label = 'unsequential' if changeSequential else 'sequential'

            return token_input, str_label

        def getContextualSample(list_token_turn, list_dialogue, dialogueIdx):
            '''
                生成一个Contextual任务的用例，随机从其他轮次中抽取一个
                返回 D_1 + [SEP] + ... + [SEP] token

                list_token_turn:
                    [[R_t + ; + U_t + [SEP]],[R_t + ; + U_t + [SEP]],...]
                list_dialogue:
                    [
                        {
                            "dialogue_idx": "MUL0012.json",
                            "domains": [],
                            "dialogue": []
                        },
                        ...
                    ]
                dialogueIdx: int 当前的对话第几个

                return: 
                    [R_t + ; + U_t + [SEP] + R_t + ; + U_t + [SEP]], str
            '''
            changeContextual =  (np.random.rand() < self.config.contextualChangeProb) # 是否改变Contextual
            

            if changeContextual: # 如果需要更改的话

                swapDialogueIdx = np.random.randint(0, len(list_dialogue)) # 决定我从哪一组对话中抽一句
                while swapDialogueIdx == dialogueIdx:
                    swapDialogueIdx = np.random.randint(0, len(list_dialogue))
                swapTurnIdx = np.random.randint(0, len(list_dialogue[swapDialogueIdx]['dialogue'])) # 从这组对话中抽取哪一轮次

                turnIdx = np.random.randint(0, len(list_token_turn)) # 替换当前对话的哪一轮次

                list_token_turn[turnIdx] = getDialogueTurnToken(self.config.tokenizer, list_dialogue[swapDialogueIdx]['dialogue'][swapTurnIdx]) # 替换对应的对话为新的对话 

            # 拼接生成最终的输入
            token_input = []
            for token_turn in list_token_turn:
                token_input += token_turn

            # 生成 str_label
            str_label = 'uncontextual' if changeContextual else 'contextual'

            return token_input, str_label
            
        

        datasetFilePath = self.config.train_path if self.isTrain else self.config.test_path 
        with open(datasetFilePath, 'r', encoding='utf-8') as in_: # 读取数据集
            list_dialogue = json.load(in_)
        
        list_datas = []
        for multiSample in range(self.config.teacherMultiSample): # 重采样扩充数据
            for dialogueIdx, dict_dialogue in tqdm(enumerate(list_dialogue), 'load dataset '):
                '''
                    {
                        "dialogue_idx": "MUL0012.json",
                        "domains": [],
                        "dialogue": []
                    }
                
                '''

                if len(dict_dialogue['dialogue']) <= 1:
                    self.config.logger.warning('{}-{}，只有一轮,丢弃这个数据'.format(dialogueIdx, dict_dialogue['dialogue_idx']))
                    continue

                list_token_turn = [] # 每个turn的文本idx [turnNum_split]
                seq_len = 0 # 当前列表中的序列长度
                for dialogueTurn in dict_dialogue['dialogue']: # 准备要用的轮次
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
                    token_turn = getDialogueTurnToken(self.config.tokenizer, dialogueTurn)
                    seq_len += len(token_turn)

                    if seq_len > self.config.pad_size: # 如果已经大于最大长度了那就不要后面的了
                        break 
                    list_token_turn.append(token_turn) # 分词 构建 idx

                if self.isSequential:
                    token_input, str_label = getSequentialSample(list_token_turn) # 如果是生成 Sequential 任务
                else:
                    token_input, str_label = getContextualSample(list_token_turn, list_dialogue, dialogueIdx) # 如果是生成 Contextual 任务

                token_input = [self.config.tokenizer.cls_token] + token_input # [CLS] + D_0 + [SEP] + ... + [SEP]
                idx_token_input = self.config.tokenizer.convert_tokens_to_ids(token_input)


                # padding, 并且生成attention mask
                attentionMask = [1] * len(idx_token_input)
                if len(idx_token_input) <= self.config.pad_size: # 做padding
                    pad_len = self.config.pad_size-len(idx_token_input)
                    idx_token_input += ([self.config.tokenizer.pad_token_id] * pad_len)
                    attentionMask += ([0] * pad_len)
                else: # 做截断
                    idx_token_input = idx_token_input[:self.config.pad_size]
                    attentionMask = attentionMask[:self.config.pad_size]
                

                
                list_datas.append([idx_token_input, attentionMask, str_label])

        return list_datas

def getDataloader(config:DiCoSConfig, isTrain:bool, isSequential:bool):
    return DataLoader(
        dataset = MultiWoZ_teacher(config, isTrain, isSequential),
        batch_size = config.batch_size,
        shuffle = config.shuffle,
        drop_last = config.drop_last,
        num_workers = config.num_workers
    )

if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig.json')
    config.setLogger()

    config.logger.info("load sequential train set......")
    sequential_trainset = getDataloader(config, False, True)
    config.logger.info("load contextual test set......")
    contextual_trainset = getDataloader(config, False, False)

    for epoch in range(config.num_epochs):
        config.logger.info("------Epoch: {}/{}------".format(epoch, config.num_epochs))
        for i, (data1, data2) in enumerate(zip(sequential_trainset, contextual_trainset)):
            pass
