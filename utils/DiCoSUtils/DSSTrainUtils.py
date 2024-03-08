import sys
sys.path.append('./')

import torch
from configs.DiCoSConfig import DiCoSConfig
from models.DiCoS import DSS
from models.Loss import MaskedNLLLoss
from utils.DiCoSUtils.DSSDataLoader import loadDataSet_DSS
from utils.multiWoZUtils import MultiWoZSchema
import torch.nn as nn
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def train_DSS(config:DiCoSConfig, schema:MultiWoZSchema, model:nn.DataParallel):
    
    # 加载数据集
    config.logger.info("load train set......")
    trainset = loadDataSet_DSS(config, True, schema)
    config.logger.info("load test set......")
    testset = loadDataSet_DSS(config, False, schema)

    model.train() # 模型训练模式

    list_param = [
        {'params': model.module.encoder.parameters(), 'lr':config.encoder_learning_rate},
        {'params': [param for name, param in model.module.named_parameters() if 'encoder' not in name and 'bias' not in name], 'lr':config.basic_learning_rate, 'weight_decay':0.15},
        {'params': [param for name, param in model.module.named_parameters() if 'encoder' not in name and 'bias' in name], 'lr':config.basic_learning_rate, 'weight_decay':0}
    ]
    optimizer = Adam(list_param, config.encoder_learning_rate)
    optimizer_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = int(config.num_epochs * len(trainset) * config.rate_warmup_steps), 
        num_training_steps = config.num_epochs * len(trainset),
        
    )

    loss_maskCE = MaskedNLLLoss()
    weight = torch.tensor([5,1], dtype=torch.float32, device=config.device, requires_grad=False)
    loss_maskCE_balance = MaskedNLLLoss()


    # 100 batch 统计量
    list_train_loss_100 = []
    list_train_preLoss_100 = []
    list_train_ultClsLoss_100 = []
    list_train_ultExtLoss_100 = []
    list_train_update_pred_100 = []
    list_train_update_target_100 = []
    list_train_cate_pred_100 = []
    list_train_cate_target_100 = []
    list_train_noncate_pred_start_100 = []
    list_train_noncate_target_start_100 = []
    list_train_noncate_pred_end_100 = []
    list_train_noncate_target_end_100 = []

    # 全局统计量
    total_batch = 0  # 记录进行到多少batch
    best_test_acc = 0 # 记录最高准确率
    last_improve = 0  # 记录上次验证集acc上升的batch数
    stop = False 
    for epoch in range(100):
        for i, data in enumerate(trainset):
            '''
                data:

                sentencesTokens,  存放用于DSS 选择阶段的 token_idx [batchSize, seqLen]
                attentionMask, attentionMask [batchSize, seqLen]
                segmentEmbedding, segmentEmbedding [batchSize, seqLen]
                slotPosition, 存放slot标签的位置 [batchSize, slotType]
                mask_sentences, 标记哪些地方是一个句子哪些地方是其他的token [batchSize, seqLen] 1为句子 0 为其他
                update_target, 是否要更新slot的监督信号 [batchSize, slotType] 
                cata_target, 分类的监督信号 [batchSize, slotType]
                cate_mask,  0表示不需要监督 1 表示需要监督 [batchSize, slotType]
                noncate_start, 抽取开始位置得监督信号 [batchSize, slotType]
                noncate_end, 抽取结束位置得监督信号 [batchSize, slotType]
                noncate_mask  0表示不需要监督 1 表示需要监督 [batchSize, slotType]
            '''
            sentencesTokens, attentionMask, segmentEmbedding, slotPosition, mask_sentences, update_target, cata_target, cate_mask, noncate_start, noncate_end, noncate_mask = data
            
            # 移动设备
            sentencesTokens = sentencesTokens.to(config.device) 
            attentionMask = attentionMask.to(config.device) 
            segmentEmbedding = segmentEmbedding.to(config.device) 
            slotPosition = slotPosition.to(config.device) 
            mask_sentences = mask_sentences.to(config.device) 
            update_target = update_target.to(config.device)
            cata_target = cata_target.to(config.device)
            cate_mask = cate_mask.to(config.device)
            noncate_start = noncate_start.to(config.device)
            noncate_end = noncate_end.to(config.device)
            noncate_mask = noncate_mask.to(config.device)

            prob_update, pre_score, startProb, endProb, slotValueProb, extUltScore, clsUltScore = model(sentencesTokens, attentionMask, segmentEmbedding, slotPosition, mask_sentences)
            '''
                prob_update, [batchSize, slotType, 2]
                pre_score, [batchSize, slotType]
                startProb, [batchSize, slotType, seqLen] 各个词为起始值的概率
                endProb,  [batchSize, slotType, seqLen] 各个词为末尾词的概率
                slotValueProb [batchSize, slotType, maxSlotValue] 各slot的分类概率值
                extUltScore [batchSize, slotType] 抽取的话得出的分数
                clsUltScore [batchSize, slotType] 分类的话得出的分数
            '''
            # Preliminary loss
            prob_update = torch.reshape(prob_update, [-1, len(config.dict_update)]) # [batchSize*slotType, 2]
            update_target = torch.reshape(update_target, [-1]) # [batchSize*slotType]
            mask_update = torch.ones(update_target.shape, dtype=torch.int8, device=config.device)

            pre_loss, _, _ = loss_maskCE_balance(prob_update, update_target, mask_update)
            list_train_preLoss_100.append(pre_loss.detach().cpu())

            # Ultimate loss

            # cls
            slotValueProb = torch.reshape(slotValueProb, [-1, config.maxSlotValue]) # [batchSize*slotType, maxSlotValue]
            cata_target = torch.reshape(cata_target, [-1]) # [batchSize*slotType]
            cate_mask = torch.reshape(cate_mask, [-1]) # [batchSize*slotType]

            ultCls_loss, mask_slotValueProb, mask_cata_target = loss_maskCE(slotValueProb, cata_target, cate_mask)
            list_train_ultClsLoss_100.append(ultCls_loss.detach().cpu())

            # ext
            noncate_mask = torch.reshape(noncate_mask, [-1]) # [batchSize*slotType]

            startProb = torch.reshape(startProb, [-1, config.pad_size]) # [batchSize*slotType, seqLen]
            noncate_start = torch.reshape(noncate_start, [-1]) # [batchSize*slotType]
            endProb = torch.reshape(endProb, [-1, config.pad_size]) # [batchSize*slotType, seqLen]
            noncate_end = torch.reshape(noncate_end, [-1]) # [batchSize*slotType]

            ultExt_start_loss, mask_startProb, mask_noncate_start = loss_maskCE(startProb, noncate_start, noncate_mask)
            ultExt_end_loss, mask_endProb, mask_noncate_end = loss_maskCE(endProb, noncate_end, noncate_mask)
            ultExt_loss = 0.5*ultExt_start_loss + 0.5*ultExt_end_loss
            list_train_ultExtLoss_100.append(ultExt_loss.detach().cpu())
            
            # 总loss
            loss = pre_loss + 0.55*ultCls_loss + 0.2*ultExt_loss
            list_train_loss_100.append(loss.detach().cpu())
            
            # 计算梯度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # 保存关心的梯度
            pre_fc_grad = model.module.preliminarySelector.fc.weight.grad.clone().detach()
            ult_CLSfc_grad = model.module.ultimateSelector.generator.fcForEachSlot.grad.clone().detach()

             # 更新参数
            optimizer.step()
            if (total_batch+10) < config.num_epochs * len(trainset): # 防止学习率为0
                optimizer_scheduler.step()
            optimizer.zero_grad()

            # 统计预测结果
            _, pred_update = prob_update.max(-1) # [batchSize*slotType]
            list_train_update_pred_100 += pred_update.detach().cpu().tolist()
            list_train_update_target_100 += update_target.detach().cpu().tolist()

            _, pred_slotValueProb = mask_slotValueProb.max(-1) # [maskNum]
            list_train_cate_pred_100 += pred_slotValueProb.detach().cpu().tolist()
            list_train_cate_target_100 += mask_cata_target.detach().cpu().tolist()

            _, pred_startProb = mask_startProb.max(-1) # [maskNum]
            list_train_noncate_pred_start_100 += pred_startProb.detach().cpu().tolist()
            list_train_noncate_target_start_100 += mask_noncate_start.detach().cpu().tolist()

            _, pred_endProb = mask_endProb.max(-1) # [maskNum]
            list_train_noncate_pred_end_100 += pred_endProb.detach().cpu().tolist()
            list_train_noncate_target_end_100 += mask_noncate_end.detach().cpu().tolist()

            if (total_batch) % 100 == 0:

                config.logger.info("epoch: {}/{}, step: {}/{}, training......, mean loss of 100 batch is {}".format(epoch+1, config.num_epochs, i+1, len(trainset), np.array(list_train_loss_100).mean()))
                dict_loss = {
                    "total loss": np.array(list_train_loss_100).mean(),
                    "Preliminary loss": np.array(list_train_preLoss_100).mean(),
                    "Ultimate CLS loss": np.array(list_train_ultClsLoss_100).mean(),
                    "Ultimate EXT loss": np.array(list_train_ultExtLoss_100).mean()
                }
                config.tbWriter.add_scalars("loss/train", dict_loss, global_step=total_batch)

                # preliminary 的参数分布
                config.tbWriter.add_histogram('weight/Preliminary Linear', model.module.preliminarySelector.fc.weight, global_step=total_batch)
                config.tbWriter.add_histogram('grad/Preliminary Linear', pre_fc_grad, global_step=total_batch)

                dict_acc_update = {
                    "total acc": accuracy_score(list_train_update_target_100, list_train_update_pred_100),
                    "update acc": precision_score(list_train_update_target_100, list_train_update_pred_100, average=None)[config.dict_update['update']],
                    "inherit acc": precision_score(list_train_update_target_100, list_train_update_pred_100, average=None)[config.dict_update['inherit']]
                }
                config.tbWriter.add_scalars("pre/train/acc", dict_acc_update, global_step=total_batch)

                dict_recall_update = {
                    "update recall": recall_score(list_train_update_target_100, list_train_update_pred_100, average=None)[config.dict_update['update']],
                    "inherit recall": recall_score(list_train_update_target_100, list_train_update_pred_100, average=None)[config.dict_update['inherit']]
                }
                config.tbWriter.add_scalars("pre/train/recall", dict_recall_update, global_step=total_batch)

                config.tbWriter.add_scalar('ultcls/train/Acc', accuracy_score(list_train_cate_target_100, list_train_cate_pred_100), total_batch)

                # Ultimate cls 的参数分布
                config.tbWriter.add_histogram('weight/Ultimate cls Linear', model.module.ultimateSelector.generator.fcForEachSlot, global_step=total_batch)
                config.tbWriter.add_histogram('grad/Ultimate cls Linear', ult_CLSfc_grad, global_step=total_batch)

                dict_acc_ultExt = {
                    "start acc": accuracy_score(list_train_noncate_target_start_100, list_train_noncate_pred_start_100),
                    "end acc": accuracy_score(list_train_noncate_target_end_100, list_train_noncate_pred_end_100)
                }
                config.tbWriter.add_scalars("ultext/train/Acc", dict_acc_ultExt, global_step=total_batch)

                list_train_loss_100 = []
                list_train_preLoss_100 = []
                list_train_ultClsLoss_100 = []
                list_train_ultExtLoss_100 = []
                list_train_update_pred_100 = []
                list_train_update_target_100 = []
                list_train_cate_pred_100 = []
                list_train_cate_target_100 = []
                list_train_noncate_pred_start_100 = []
                list_train_noncate_target_start_100 = []
                list_train_noncate_pred_end_100 = []
                list_train_noncate_target_end_100 = []


            if (total_batch) % config.eval_step == 0:
                config.logger.warning('Iter:{}/{}, strat evaluate on testset'.format(i, len(trainset)))
                model.eval()
                totalLoss, preLoss, ultClsLoss, ultExtLoss, preAcc, preAcc_update, preAcc_inherit, preRecall_update, preRecall_inherit, preF1, preF1_update, preF1_inherit, ultAcc_cata, ultCls_F1_update, ultAcc_start, ultAcc_end, ultExt_F1_update, F1_joint, F1_joint_update, F1_joint_inherit = eval_DSS(config, schema, model, testset, total_batch)
                dict_loss = {
                    "total loss": totalLoss,
                    "Preliminary loss": preLoss,
                    "Ultimate CLS loss": ultClsLoss,
                    "Ultimate EXT loss": ultExtLoss
                }
                config.tbWriter.add_scalars("loss/test", dict_loss, global_step=total_batch)

                dict_acc_update = {
                    "total acc": preAcc,
                    "update acc": preAcc_update,
                    "inherit acc": preAcc_inherit
                }
                config.tbWriter.add_scalars("pre/test/acc", dict_acc_update, global_step=total_batch)

                dict_recall_pre = {
                    "update recall": preRecall_update,
                    "inherit recall": preRecall_inherit
                }
                config.tbWriter.add_scalars('pre/test/recall', dict_recall_pre, global_step=total_batch)

                dict_f1_pre_update = {
                    "total f1": preF1,
                    "update f1": preF1_update,
                    "inherit f1": preF1_inherit
                }
                config.tbWriter.add_scalars("pre/test/f1", dict_f1_pre_update, global_step=total_batch)

                config.tbWriter.add_scalar('ultcls/test/Acc', ultAcc_cata, global_step=total_batch)

                dict_acc_ultExt = {
                    "start acc": ultAcc_start,
                    "end acc": ultAcc_end
                }
                config.tbWriter.add_scalars("ultext/test/Acc", dict_acc_ultExt, global_step=total_batch)

                dict_f1_ult_update = {
                    "ult cls update f1": ultCls_F1_update,
                    "ult ext update f1": ultExt_F1_update
                }
                config.tbWriter.add_scalars("ult/test/update", dict_f1_ult_update, global_step=total_batch)


                dict_f1_joint_update = {
                    "total f1": F1_joint,
                    "update f1": F1_joint_update,
                    "inherit f1": F1_joint_inherit
                }
                config.tbWriter.add_scalars("joint/test/f1", dict_f1_joint_update, global_step=total_batch)

                if preF1_update >= best_test_acc:
                    best_test_acc = preF1_update
                    torch.save(model.module.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                msg = 'epoch: {}/{}, step: {}/{}, test mean loss is {}, pre update f1 is {} {}'.format(epoch+1, config.num_epochs, i+1, len(trainset), totalLoss, preF1_update, improve)
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
def eval_DSS(config:DiCoSConfig, schema:MultiWoZSchema, model:DSS, testSet, total_batch:int=None):
    '''
        gold input 得情况下进行 eval
    
    '''
    loss_f = MaskedNLLLoss()
    weight = torch.tensor([5,1], dtype=torch.float32, device=config.device, requires_grad=False)
    loss_maskCE_balance = MaskedNLLLoss()

    list_test_loss = []
    list_test_preLoss = []
    list_test_ultClsLoss = []
    list_test_ultExtLoss = []
    list_test_update_pre_pred = []
    list_test_update_target = []
    list_test_cate_pred = []
    list_test_cate_target = []
    list_test_update_cate_pred = []
    list_test_noncate_pred_start = []
    list_test_noncate_target_start = []
    list_test_noncate_pred_end = []
    list_test_noncate_target_end = []
    list_test_update_noncate_pred = []
    list_test_update_joint_pred = []
    list_test_pre_score = []
    list_test_ext_score = []
    list_test_cls_score = []
    list_test_score = []
    for i, data in enumerate(testSet):
        sentencesTokens, attentionMask, segmentEmbedding, slotPosition, mask_sentences, update_target, cata_target, cate_mask, noncate_start, noncate_end, noncate_mask = data
         # 移动设备
        sentencesTokens = sentencesTokens.to(config.device) 
        attentionMask = attentionMask.to(config.device) 
        segmentEmbedding = segmentEmbedding.to(config.device) 
        slotPosition = slotPosition.to(config.device) 
        mask_sentences = mask_sentences.to(config.device) 
        update_target = update_target.to(config.device)
        cata_target = cata_target.to(config.device)
        cate_mask = cate_mask.to(config.device)
        noncate_start = noncate_start.to(config.device)
        noncate_end = noncate_end.to(config.device)
        noncate_mask = noncate_mask.to(config.device)

        prob_update, pre_score, startProb, endProb, slotValueProb, extUltScore, clsUltScore = model(sentencesTokens, attentionMask, segmentEmbedding, slotPosition, mask_sentences)
        '''
                prob_update, [batchSize, slotType, 2]
                pre_score, [batchSize, slotType]
                startProb, [batchSize, slotType, seqLen] 各个词为起始值的概率
                endProb,  [batchSize, slotType, seqLen] 各个词为末尾词的概率
                slotValueProb [batchSize, slotType, maxSlotValue] 各slot的分类概率值
                extUltScore [batchSize, slotType] 抽取的话得出的分数
                clsUltScore [batchSize, slotType] 分类的话得出的分数
        '''
        # Preliminary loss

        prob_update = torch.reshape(prob_update, [-1, len(config.dict_update)]) # [batchSize*slotType, 2]
        update_target = torch.reshape(update_target, [-1]) # [batchSize*slotType]
        mask_update = torch.ones(update_target.shape, dtype=torch.int8, device=config.device)

        pre_loss, _, _ = loss_maskCE_balance(prob_update, update_target, mask_update)
        list_test_preLoss.append(pre_loss.detach().cpu())

        # Ultimate loss

        # cls
        slotValueProb = torch.reshape(slotValueProb, [-1, config.maxSlotValue]) # [batchSize*slotType, maxSlotValue]
        cata_target = torch.reshape(cata_target, [-1]) # [batchSize*slotType]
        cate_mask = torch.reshape(cate_mask, [-1]) # [batchSize*slotType]

        ultCls_loss, mask_slotValueProb, mask_cata_target = loss_f(slotValueProb, cata_target, cate_mask)
        list_test_ultClsLoss.append(ultCls_loss.detach().cpu())

        # ext
        noncate_mask = torch.reshape(noncate_mask, [-1]) # [batchSize*slotType]

        startProb = torch.reshape(startProb, [-1, config.pad_size]) # [batchSize*slotType, seqLen]
        noncate_start = torch.reshape(noncate_start, [-1]) # [batchSize*slotType]
        endProb = torch.reshape(endProb, [-1, config.pad_size]) # [batchSize*slotType, seqLen]
        noncate_end = torch.reshape(noncate_end, [-1]) # [batchSize*slotType]

        ultExt_start_loss, mask_startProb, mask_noncate_start = loss_f(startProb, noncate_start, noncate_mask)
        ultExt_end_loss, mask_endProb, mask_noncate_end = loss_f(endProb, noncate_end, noncate_mask)
        ultExt_loss = 0.5*ultExt_start_loss + 0.5*ultExt_end_loss
        list_test_ultExtLoss.append(ultExt_loss.detach().cpu())
            
        # 总loss
        loss = pre_loss + 0.55*ultCls_loss + 0.2*ultExt_loss
        list_test_loss.append(loss.detach().cpu())

        # 统计预测结果
        _, pred_update = prob_update.max(-1) # [batchSize*slotType]
        list_test_update_pre_pred += pred_update.detach().cpu().tolist()
        list_test_update_target += update_target.detach().cpu().tolist()

        _, pred_slotValueProb = mask_slotValueProb.max(-1) # [maskNum]
        list_test_cate_pred += pred_slotValueProb.detach().cpu().tolist()
        list_test_cate_target += mask_cata_target.detach().cpu().tolist()

        # 凭借分类能达到的准确率
        cate_update = torch.gt(torch.tensor(0, dtype=torch.float32, device=config.device), clsUltScore).int().reshape([-1]) # [batchSize*slotType]
        list_test_update_cate_pred += cate_update.detach().cpu().tolist()

        _, pred_startProb = mask_startProb.max(-1) # [maskNum]
        list_test_noncate_pred_start += pred_startProb.detach().cpu().tolist()
        list_test_noncate_target_start += mask_noncate_start.detach().cpu().tolist()

        _, pred_endProb = mask_endProb.max(-1) # [maskNum]
        list_test_noncate_pred_end += pred_endProb.detach().cpu().tolist()
        list_test_noncate_target_end += mask_noncate_end.detach().cpu().tolist()

        # 凭借抽取能达到的准确率
        noncate_update = torch.gt(torch.tensor(0, dtype=torch.float32, device=config.device), extUltScore).int().reshape([-1]) # [batchSize*slotType]
        list_test_update_noncate_pred += noncate_update.detach().cpu().tolist()

        # 计算joint 预测结果
        cls_ext_mask = schema.mask_cateSlot.unsqueeze(0).expand_as(extUltScore) # 1 为 分类
        extUltScore = torch.masked_fill(extUltScore, cls_ext_mask==1, 0) # 分类填充为0
        clsUltScore = torch.masked_fill(clsUltScore, cls_ext_mask==0, 0) # 抽取填充为0
        score = pre_score + extUltScore + clsUltScore # [batchSize, slotType]
        joint_update = torch.gt(torch.tensor(0, dtype=torch.float32, device=config.device), score).int().reshape([-1]) # [batchSize*slotType]
        list_test_update_joint_pred += joint_update.detach().cpu().tolist()

        # 保存分数
        list_test_pre_score += torch.reshape(pre_score, [-1]).detach().cpu().tolist() # [batchSize*slotType]
        list_test_cls_score += torch.reshape(clsUltScore, [-1]).detach().cpu().tolist() # [batchSize*slotType]
        list_test_ext_score += torch.reshape(extUltScore, [-1]).detach().cpu().tolist() # [batchSize*slotType]
        list_test_score += torch.reshape(score, [-1]).detach().cpu().tolist() # [batchSize*slotType]
    
    # 计算评价指标
    totalLoss = np.array(list_test_loss).mean()
    preLoss = np.array(list_test_preLoss).mean()
    ultClsLoss = np.array(list_test_ultClsLoss).mean()
    ultExtLoss = np.array(list_test_ultExtLoss).mean()

    preAcc = accuracy_score(list_test_update_target, list_test_update_pre_pred) 
    preAcc_update = precision_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['update']]
    preAcc_inherit = precision_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['inherit']]

    preRecall_update = recall_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['update']]
    preRecall_inherit = recall_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['inherit']]

    preF1 = f1_score(list_test_update_target, list_test_update_pre_pred)
    preF1_update = f1_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['update']]
    preF1_inherit = f1_score(list_test_update_target, list_test_update_pre_pred, average=None)[config.dict_update['inherit']]

    ultAcc_cata = accuracy_score(list_test_cate_target, list_test_cate_pred)
    ultAcc_start = accuracy_score(list_test_noncate_target_start, list_test_noncate_pred_start)
    ultAcc_end = accuracy_score(list_test_noncate_target_end, list_test_noncate_pred_end)

    ultCls_F1_update = f1_score(list_test_update_target, list_test_update_cate_pred, average=None)[config.dict_update['update']]
    ultExt_F1_update = f1_score(list_test_update_target, list_test_update_noncate_pred, average=None)[config.dict_update['update']]

    F1_joint = f1_score(list_test_update_target, list_test_update_joint_pred)
    F1_joint_update = f1_score(list_test_update_target, list_test_update_joint_pred, average=None)[config.dict_update['update']]
    F1_joint_inherit = f1_score(list_test_update_target, list_test_update_joint_pred, average=None)[config.dict_update['inherit']]

    # 分数分布
    if total_batch != None: # 如果不指定total_batch，则不记录tensorboard
        config.tbWriter.add_histogram('score/pre', np.array(list_test_pre_score), total_batch)
        config.tbWriter.add_histogram('score/cls', np.array(list_test_cls_score), total_batch)
        config.tbWriter.add_histogram('score/ext', np.array(list_test_ext_score), total_batch)
        config.tbWriter.add_histogram('score/total', np.array(list_test_score), total_batch)


    return totalLoss, preLoss, ultClsLoss, ultExtLoss, preAcc, preAcc_update, preAcc_inherit, preRecall_update, preRecall_inherit, preF1, preF1_update, preF1_inherit, ultAcc_cata, ultCls_F1_update, ultAcc_start, ultAcc_end, ultExt_F1_update, F1_joint, F1_joint_update, F1_joint_inherit