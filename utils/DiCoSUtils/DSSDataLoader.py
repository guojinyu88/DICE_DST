from typing import List
from torch.utils.data import DataLoader
from configs.DiCoSConfig import DiCoSConfig
from utils.DiCoSUtils.DiCoSMultiWoZDataUtils import SentencesInstences
import torch
from utils.multiWoZUtils import MultiWoZSchema
from utils.DiCoSUtils.DiCoSMultiWoZDataUtils import MultiWoZDataset

def dss_collate_fn(datas:List[List[SentencesInstences]], config:DiCoSConfig, buildSUpervise:bool=True):
    '''
        对层层封装进行拆包，拆包成模型可以直接输入的格式
        [
            [ # 封装一个样本我们需要用到的所有内容
                turnid, # 当前轮的id
                current_turn, # 当前轮的输入 SentencesInstences
                list_sentencesInstence_history.copy(), # 用于对话选择过程
                list_select_history, # 用于选择好历史对话之后的SVG阶段
                graph, # 存储了四种边
                current_state # 当前阶段的状态真值 dict 用于验证acc
            ],
            ......
        ]

        return:
        sentencesTokens,  存放用于DSS 选择阶段的 token_idx [batchSize, seqLen]
        attentionMask, attentionMask [batchSize, seqLen]
        segmentEmbedding, segmentEmbedding [batchSize, seqLen]
        slotPosition, 存放slot标签的位置 [batchSize, slotType]
        mask_sentences, 标记哪些地方是一个句子哪些地方是其他的token [batchSize, seqLen] 1为句子 0 为其他
        update_target, 是否要更新slot的监督信号 [batchSize, slotType] 
        cata_target, 分类的监督信号 [batchSize, slotType] 0表示继承 其他表示更新
        cate_mask,  0表示不需要监督 1 表示需要监督 [batchSize, slotType]
        noncate_start, 抽取开始位置得监督信号 [batchSize, slotType] 0表示继承 其他表示更新
        noncate_end, 抽取结束位置得监督信号 [batchSize, slotType] 0表示继承 其他表示更新
        noncate_mask  0表示不需要监督 1 表示需要监督 [batchSize, slotType]

    '''

    # 模型输入
    list_turn_id = [] # 存放当前对话的id
    list_sentencesTokens = [] # 存放用于DSS 选择阶段的 token_idx [batchSize, seqLen]
    list_attentionMask = [] # attentionMask [batchSize, seqLen]
    list_segmentEmbedding = [] # segmentEmbedding [batchSize, seqLen]
    list_slotPosition = [] # 存放slot标签的位置 [batchSize, slotType]
    list_mask_sentences = [] # 标记哪些地方是一个句子哪些地方是其他的token [batchSize, seqLen] 1为句子 0 为其他

    # 监督信号
    list_update = [] # 是否要更新slot的监督信号 [batchSize, slotType]

    list_cata_target = [] # 分类的监督信号 [batchSize, slotType] 0表示继承 其他表示更新
    list_cate_mask = [] # 0表示不需要监督 1 表示需要监督 [batchSize, slotType]

    list_noncate_start = [] # 抽取开始位置得监督信号 [batchSize, slotType] 0表示继承 其他表示更新
    list_noncate_end = [] # 抽取结束位置得监督信号 [batchSize, slotType] 0表示继承 其他表示更新
    list_noncate_mask = [] # 0表示不需要监督 1 表示需要监督 [batchSize, slotType]

    for data in datas:
        # 模型输入
        list_turn_id.append(data[0])
        list_sentencesTokens.append(data[1].tokens_idx)
        list_attentionMask.append(data[1].attentionMask)
        list_segmentEmbedding.append(data[1].segmentEmbedding)
        list_slotPosition.append(data[1].slotTokenPosition)
        list_mask_sentences.append(data[1].sentencesMask)
        if buildSUpervise: # 如果构建监督信号的话（默认构建）
            # 监督信号的构建
            list_update.append(data[1].supervised_update)

            # 分类监督信号
            cata_target = []
            cate_mask = []
            for slotName, target in zip(data[1].schema.list_slotName, data[1].supervised_cata):
                if target == None: # 分类继承 + 抽取槽
                    if data[1].schema.isCatagorical(slotName): # 分类继承
                        cata_target.append(0)
                        cate_mask.append(1)
                    else: # 抽取槽
                        cata_target.append(1e9)
                        cate_mask.append(0)
                else: # 分类 需要更新的
                    cata_target.append(target)
                    cate_mask.append(1)

            list_cata_target.append(cata_target)
            list_cate_mask.append(cate_mask)

            # 抽取监督信号
            noncate_start = []
            noncate_end = []
            noncate_mask = []
            for start, end in data[1].supervised_noncate:
                if start == None: # 继承+分类槽抽取不到的：ultimate认为不需要更新                
                    noncate_start.append(0)
                    noncate_end.append(0)
                    noncate_mask.append(1)
                elif start == -1: # 抽取槽抽取不到：ultimate认为不需要更新
                    noncate_start.append(0)
                    noncate_end.append(0)
                    noncate_mask.append(1)
                else: # 可以抽取到值：ultimate认为需要更新
                    noncate_start.append(start)
                    noncate_end.append(end)
                    noncate_mask.append(1)
            
            list_noncate_start.append(noncate_start)
            list_noncate_end.append(noncate_end)
            list_noncate_mask.append(noncate_mask)


    sentencesTokens = torch.tensor(list_sentencesTokens, dtype=torch.long) # [batchSize, seqLen]
    attentionMask = torch.tensor(list_attentionMask, dtype=torch.long) # [batchSize, seqLen]
    segmentEmbedding = torch.tensor(list_segmentEmbedding, dtype=torch.long) # [batchSize, seqLen]
    slotPosition = torch.tensor(list_slotPosition, dtype=torch.long) # [batchSize, slotType]
    mask_sentences = torch.tensor(list_mask_sentences, dtype=torch.int8) # [batchSize, seqLen]

    update_target = torch.tensor(list_update, dtype=torch.long) # [batchSize, slotType]
    cata_target = torch.tensor(list_cata_target, dtype=torch.long) # [batchSize, slotType]
    cate_mask = torch.tensor(list_cate_mask, dtype=torch.int8) # [batchSize, slotType]
    noncate_start = torch.tensor(list_noncate_start, dtype=torch.long) # [batchSize, slotType]
    noncate_end = torch.tensor(list_noncate_end, dtype=torch.long) # [batchSize, slotType]
    noncate_mask = torch.tensor(list_noncate_mask, dtype=torch.int8) # [batchSize, slotType]

    return sentencesTokens, attentionMask, segmentEmbedding, slotPosition, mask_sentences, update_target, cata_target, cate_mask, noncate_start, noncate_end, noncate_mask


def loadDataSet_DSS(config:DiCoSConfig, isTrain:bool, schema:MultiWoZSchema):
    return DataLoader(
        dataset = MultiWoZDataset(config, isTrain, schema),
        batch_size = config.batch_size,
        shuffle = config.shuffle,
        drop_last = config.drop_last,
        collate_fn = lambda datas: dss_collate_fn(datas, config),
        num_workers = config.num_workers
    )
        