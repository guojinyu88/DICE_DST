from functools import total_ordering
from imghdr import tests
from typing import Tuple
import torch
from configs.DiCoSConfig import DiCoSConfig
from models.DiCoS import DiCoS
from models.Loss import DiCoSGeneratorLoss
from utils.DiCoSUtils.DiCoSDataLoader import loadDataSet_DiCoS
from utils.multiWoZUtils import MultiWoZSchema
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def to_device(params:Tuple[torch.Tensor], config:DiCoSConfig):
    list_params_device = []
    for param in params:
        if type(param) == torch.Tensor:
            list_params_device.append(param.to(config.device))
        else:
            list_params_device.append(param)
    return list_params_device

def generate_state(schema:MultiWoZSchema, startProb:torch.Tensor, endProb:torch.Tensor, slotValueProb:torch.Tensor, selectedHistory_sentencesTokens:torch.Tensor, update_slot:torch.Tensor, last_state:dict):
    '''
    schema:MultiWoZSchema, 
    startProb:torch.Tensor, 
    endProb:torch.Tensor, 
    slotValueProb:torch.Tensor, 
    selectedHistory_sentencesTokens:torch.Tensor, 
    update_slot:torch.Tensor, 
    last_state:dict
    
    return:
    current_state_pred: dict
    '''
    pass


def train_DiCoS(config:DiCoSConfig, schema:MultiWoZSchema, model:nn.DataParallel):

    # 加载数据集
    config.logger.info("load train set......")
    trainset = loadDataSet_DiCoS(config, True, schema)
    config.logger.info("load test set......")
    testset = loadDataSet_DiCoS(config, False, schema)

    model.train()

    list_param = [
        {'params': model.module.encoder.parameters(), 'lr':config.encoder_learning_rate},
        {'params': [param for name, param in model.module.named_parameters() if 'encoder' not in name and 'bias' not in name], 'lr':config.basic_learning_rate, 'weight_decay':0.01},
        {'params': [param for name, param in model.module.named_parameters() if 'encoder' not in name and 'bias' in name], 'lr':config.basic_learning_rate, 'weight_decay':0}
    ]
    optimizer = AdamW(list_param, config.encoder_learning_rate)

    optimizer_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = int(config.num_epochs * len(trainset) * config.rate_warmup_steps), 
        num_training_steps = config.num_epochs * len(trainset),
        
    )

    loss_f = DiCoSGeneratorLoss(config)

    # 100 batch 统计量
    list_train_loss_100 = []
    list_train_cls_pred = []
    list_train_cls_target = []
    list_train_isExtRight = []

    # 全局统计量
    total_batch = 0  # 记录进行到多少batch
    best_test_acc = 0 # 记录最高准确率
    last_improve = 0  # 记录上次验证集acc上升的batch数
    stop = False 
    for epoch in range(100):
        for i, data in enumerate(trainset):
            '''
                update_slot: torch.Tensor, [batchSize, slotType] 0继承 1更新

                current_sentencesTokens, [batchSize, seqLen] 当前轮对话token
                current_attentionMask, [batchSize, seqLen] 当前轮对话token
                current_segmentEmbedding, [batchSize, seqLen] 当前轮对话token
                current_slotPosition, [batchSize, slotType] 当前轮对话 [slot] 在哪
                current_valuePosition, [batchSize, slotType] 当前轮对话 [value] 在哪

                history_sentencesTokens, [batchSize, maxHistoryNum, seqLen] 历史对话token
                history_attentionMask, [batchSize, maxHistoryNum, seqLen] 历史对话token
                history_segmentEmbedding, [batchSize, maxHistoryNum, seqLen] 历史对话token
                history_mask_sentences, [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
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
            data = to_device(data, config)
            update_slot, current_sentencesTokens, current_attentionMask, current_segmentEmbedding, current_slotPosition, current_valuePosition,\
            history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, \
            updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, \
            selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding, selectedHistory_slotPosition, selectedHistory_mask_sentences, \
            cata_target, cate_mask, noncate_start, noncate_end, noncate_mask, \
            list_last_state, list_current_state = data

            score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens = model.forward(current_sentencesTokens, current_attentionMask, current_segmentEmbedding, 
                                                                                        current_slotPosition, current_valuePosition,  # 当前轮信息
                                                                                        history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, # 历史轮信息
                                                                                        updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, # 四个边
                                                                                        selectedHistory_sentencesTokens,selectedHistory_attentionMask, selectedHistory_segmentEmbedding, 
                                                                                        selectedHistory_slotPosition, selectedHistory_mask_sentences)# 当前轮+·被选择对话的信息
            
            '''
                score,  [batchSize, slotType, max_historyNum]
                selectedScore, [updateSlotNum]           每个更新slot选择当前对话的分数
                startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
                endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
                slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果
                selectedHistory_sentencesTokens [updateSlotNum, seqLen], 从哪一句话中抽取出来的
            '''

            loss, slotValueProb, cata_target, startProb, noncate_start, endProb, noncate_end = loss_f.forward(score, update_slot, startProb, endProb, slotValueProb,
                                                                                                                cata_target, cate_mask, noncate_start, noncate_end, noncate_mask)

            '''
                slotValueProb, [needSup_cls, maxSlotValue]
                cata_target,  [needSup_cls] 需要监督的分类的
                startProb,  [needSup_ext, seqLen] 
                noncate_start,  [needSup_ext] 需要监督的抽取的
                endProb,  [needSup_ext, seqLen]
                noncate_end [needSup_ext]
            '''
            list_train_loss_100.append(loss.detach().cpu())
            # 分类
            _, cls_pred = slotValueProb.max(dim=-1) # [needSup_cls]
            list_train_cls_pred += cls_pred.detach().cpu().tolist()
            list_train_cls_target += cata_target.detach().cpu().tolist()

            # 抽取
            _, startPosition = startProb.max(dim=-1) # [needSup_ext]
            _, endPosition = endProb.max(dim=-1) # [needSup_ext]
            isStartRight = torch.eq(startPosition, noncate_start) # [needSup_ext]
            isEndRight = torch.eq(endPosition, noncate_end) # [needSup_ext]
            isExtRight = isStartRight * isEndRight # 逻辑 ‘和’ 运算

            list_train_isExtRight += isExtRight.int().detach().cpu().tolist() # 1 为正确 0 为错误

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # 保存感兴趣的梯度信息
            select_grad = model.module.multiPrespectiveSelector.hiddenState2score.weight.grad.clone().detach()
            cls_grad = model.module.SVG.fcForEachSlot.grad.clone().detach()
            start_grad = model.module.SVG.fc_start.weight.grad.clone().detach()
            end_grad = model.module.SVG.fc_end.weight.grad.clone().detach()
            
            optimizer.step()
            optimizer.zero_grad()
            if (total_batch+10) < config.num_epochs * len(trainset): # 防止学习率为0
                optimizer_scheduler.step()

            if (total_batch) % 100 == 0:
                config.logger.info("epoch: {}/{}, step: {}/{}, training......, mean loss of 100 batch is {}".format(epoch+1, config.num_epochs, i+1, len(trainset), np.array(list_train_loss_100).mean()))
                dict_loss = {
                    'total loss': np.array(list_train_loss_100).mean()
                }
                config.tbWriter.add_scalars('loss/train', dict_loss, global_step=total_batch)
                dict_cls = {
                    'cls acc': accuracy_score(list_train_cls_target, list_train_cls_pred)
                }
                config.tbWriter.add_scalars('cls/train', dict_cls, global_step=total_batch)
                dict_ext = {
                    'ext acc': np.array(list_train_isExtRight).sum() / len(list_train_isExtRight)
                }
                config.tbWriter.add_scalars('ext/train', dict_ext, global_step=total_batch)
                
                # 模型权重
                config.tbWriter.add_histogram('weight/cls', model.module.SVG.fcForEachSlot, global_step=total_batch)
                config.tbWriter.add_histogram('weight/startExt', model.module.SVG.fc_start.weight, global_step=total_batch)
                config.tbWriter.add_histogram('weight/endExt', model.module.SVG.fc_end.weight, global_step=total_batch)
                config.tbWriter.add_histogram('weight/selector', model.module.multiPrespectiveSelector.hiddenState2score.weight, global_step=total_batch)

                # 分数分布
                config.tbWriter.add_histogram('score/score', score.clone().detach(),  global_step=total_batch)
                config.tbWriter.add_histogram('score/selectedScore', selectedScore.clone().detach(),  global_step=total_batch)

                # 梯度分布
                config.tbWriter.add_histogram('grad/cls', cls_grad, global_step=total_batch)
                config.tbWriter.add_histogram('grad/startExt', start_grad, global_step=total_batch)
                config.tbWriter.add_histogram('grad/endExt', end_grad, global_step=total_batch)
                config.tbWriter.add_histogram('grad/selector', select_grad,  global_step=total_batch)
                
                # 重置
                list_train_loss_100 = []
                list_train_cls_pred = []
                list_train_cls_target = []
                list_train_isExtRight = []
            
            if total_batch % config.eval_step == 0:
                config.logger.warning('Iter:{}/{}, strat evaluate on testset'.format(i, len(trainset)))
                model.eval()
                totalLoss, test_acc = eval_DiCoS(config, schema, model, testset, total_batch)


                if test_acc >= best_test_acc:
                    best_test_acc = test_acc
                    torch.save(model.module.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                msg = 'epoch: {}/{}, step: {}/{}, test mean loss is {}, mean acc is {} {}'.format(epoch+1, config.num_epochs, i+1, len(trainset), totalLoss, test_acc, improve)
                config.logger.info(msg)
                model.train()

            total_batch += 1
            if (total_batch - last_improve) > config.require_improvement:
                config.logger.info("No optimization for a long time, auto-stopping...")
                stop = True
                break
            
        if stop:
            break

    config.logger.info("******FINISH TRAINING******")
 



            


@torch.no_grad()
def eval_DiCoS(config:DiCoSConfig, schema:MultiWoZSchema, model:DiCoS, testset, total_batch:int=None):
    '''
        gold last state + gold sup 的情况下eval
    
    '''
    loss_f = DiCoSGeneratorLoss(config)
    list_test_loss_100 = []
    list_test_cls_pred = []
    list_test_cls_target = []
    list_test_isExtRight = []

    for i, data in enumerate(testset):
        '''
            update_slot: torch.Tensor, [batchSize, slotType] 0继承 1更新

            current_sentencesTokens, [batchSize, seqLen] 当前轮对话token
            current_attentionMask, [batchSize, seqLen] 当前轮对话token
            current_segmentEmbedding, [batchSize, seqLen] 当前轮对话token
            current_slotPosition, [batchSize, slotType] 当前轮对话 [slot] 在哪
            current_valuePosition, [batchSize, slotType] 当前轮对话 [value] 在哪

            history_sentencesTokens, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_attentionMask, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_segmentEmbedding, [batchSize, maxHistoryNum, seqLen] 历史对话token
            history_mask_sentences, [batchSize, max_historyNum, sqeLen] 真句子的地方填写1, slot的地方填写0, padding 的位置填写0
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
        data = to_device(data, config)
        update_slot, current_sentencesTokens, current_attentionMask, current_segmentEmbedding, current_slotPosition, current_valuePosition,\
        history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, \
        updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, \
        selectedHistory_sentencesTokens, selectedHistory_attentionMask, selectedHistory_segmentEmbedding, selectedHistory_slotPosition, selectedHistory_mask_sentences, \
        cata_target, cate_mask, noncate_start, noncate_end, noncate_mask, \
        list_last_state, list_current_state = data

        score, selectedScore, startProb, endProb, slotValueProb, selectedHistory_sentencesTokens = model.forward(current_sentencesTokens, current_attentionMask, current_segmentEmbedding, 
                                                                                    current_slotPosition, current_valuePosition,  # 当前轮信息
                                                                                    history_sentencesTokens, history_attentionMask, history_segmentEmbedding, history_mask_sentences, # 历史轮信息
                                                                                    updateSlot2current, updateSlot2allSlot, slot2lastUpdateTurn, slot_domain_connect, # 四个边
                                                                                    selectedHistory_sentencesTokens,selectedHistory_attentionMask, selectedHistory_segmentEmbedding, 
                                                                                    selectedHistory_slotPosition, selectedHistory_mask_sentences)# 当前轮+·被选择对话的信息
        '''
            score,  [batchSize, slotType, max_historyNum]
            selectedScore, [updateSlotNum]           每个更新slot选择当前对话的分数
            startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
            endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
            slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果
            selectedHistory_sentencesTokens [updateSlotNum, seqLen], 从哪一句话中抽取出来的
        '''

        loss, slotValueProb, cata_target, startProb, noncate_start, endProb, noncate_end = loss_f.forward(score, update_slot, startProb, endProb, slotValueProb,
                                                                                                            cata_target, cate_mask, noncate_start, noncate_end, noncate_mask)

        '''
            slotValueProb, [needSup_cls, maxSlotValue]
            cata_target,  [needSup_cls] 需要监督的分类的
            startProb,  [needSup_ext, seqLen] 
            noncate_start,  [needSup_ext] 需要监督的抽取的
            endProb,  [needSup_ext, seqLen]
            noncate_end [needSup_ext]
        '''
        list_test_loss_100.append(loss.detach().cpu())
        # 分类
        _, cls_pred = slotValueProb.max(dim=-1) # [needSup_cls]
        list_test_cls_pred += cls_pred.detach().cpu().tolist()
        list_test_cls_target += cata_target.detach().cpu().tolist()

        # 抽取
        _, startPosition = startProb.max(dim=-1) # [needSup_ext]
        _, endPosition = endProb.max(dim=-1) # [needSup_ext]
        isStartRight = torch.eq(startPosition, noncate_start) # [needSup_ext]
        isEndRight = torch.eq(endPosition, noncate_end) # [needSup_ext]
        isExtRight = isStartRight * isEndRight # 逻辑 ‘和’ 运算

        list_test_isExtRight += isExtRight.int().detach().cpu().tolist() # 1 为正确 0 为错误

    dict_loss = {
        'total loss': np.array(list_test_loss_100).mean()
    }
    dict_cls = {
        'cls acc': accuracy_score(list_test_cls_target, list_test_cls_pred)
    }
    dict_ext = {
        'ext acc': np.array(list_test_isExtRight).sum() / len(list_test_isExtRight)
    }
    if total_batch != None: # 如果不指定total_batch，则不记录tensorboard
        config.tbWriter.add_scalars('cls/test', dict_cls, global_step=total_batch)
        config.tbWriter.add_scalars('loss/test', dict_loss, global_step=total_batch)
        config.tbWriter.add_scalars('ext/test', dict_ext, global_step=total_batch)

    mean_acc = (dict_cls['cls acc'] + dict_ext['ext acc']) / 2

    return dict_loss['total loss'], mean_acc