from statistics import mode
import sys
sys.path.append('./')

import torch
import torch.nn as nn
from configs.DiCoSConfig import DiCoSConfig
from models.Teacher import Teacher
from utils.teacherUtils.teacherDataUtils import getDataloader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import numpy as np

def train_teacher(config:DiCoSConfig, model:nn.DataParallel):

    # 加载数据集
    config.logger.info("load sequential train set......")
    sequential_trainset = getDataloader(config, True, True)
    config.logger.info("load sequential test set......")
    sequential_testset = getDataloader(config, False, True)
    config.logger.info("load contextual train set......")
    contextual_trainset = getDataloader(config, True, False)
    config.logger.info("load contextual test set......")
    contextual_testset = getDataloader(config, False, False)

    start_time = time.time() # 记录开始训练的时间
    config.logger.info("{}, start train......".format(start_time))
    
    model.train() # 模型训练模式
    parameters = [
        {'params': model.module.encoder.parameters(), 'lr': config.encoder_learning_rate},
        {'params': [param for name, param in model.module.named_parameters() if 'encoder' not in name], 'lr': config.basic_learning_rate},
    ]

    # 创建参数优化器
    optimizer = AdamW(parameters, config.encoder_learning_rate)
    optimizer_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = int(config.num_epochs * len(sequential_trainset) * config.rate_warmup_steps), 
        num_training_steps = config.num_epochs * len(sequential_trainset)
    )

    loss_f = nn.CrossEntropyLoss()
    
    total_batch = 0  # 记录进行到多少batch
    best_test_meanAcc = 0 # 记录最高准确率
    last_improve = 0  # 记录上次验证集acc上升的batch数
    stop = False
    list_train_loss_sequential = [] # 保存每100个batch的平均loss
    list_train_loss_contextual = [] # 保存每100个batch的平均loss
    correct_sequential = 0 # 保存每100个batch对了多少
    correct_contextual = 0 # 保存每100个batch对了多少
    for epoch in range(config.num_epochs):
        config.logger.info("------Epoch: {}/{}------".format(epoch, config.num_epochs))
        for i, (sequential_data, contextual_data) in enumerate(zip(sequential_trainset, contextual_trainset)):
            # 构建输入
            sentencesTokens = torch.cat([sequential_data['sentencesTokens'], contextual_data['sentencesTokens']], dim=0)
            attentionMask = torch.cat([sequential_data['attentionMask'], contextual_data['attentionMask']], dim=0)

            # 转移到对应设备
            sentencesTokens = sentencesTokens.to(config.device)
            attentionMask = attentionMask.to(config.device)

            # 得到输出
            logit_sequential, logit_contextual = model(sentencesTokens, attentionMask) # [batchSize*2, 2], [batchSize*2, 2]
            
            # 取出两个任务各自的分类结果
            logit_sequential = logit_sequential[:int(sentencesTokens.shape[0]/2)] # [batchSize, 2]
            logit_contextual = logit_contextual[int(sentencesTokens.shape[0]/2):] # [batchSize, 2]

            # 把label放到对应的设备上
            target_sequential = sequential_data['label'].to(config.device) # [batchSize]
            target_contextual = contextual_data['label'].to(config.device) # [batchSize]

            # 计算损失
            loss_sequential = loss_f(logit_sequential, target_sequential)
            loss_contextual = loss_f(logit_contextual, target_contextual)

            # 保存损失
            list_train_loss_sequential.append(loss_sequential.cpu().detach().numpy())
            list_train_loss_contextual.append(loss_contextual.cpu().detach().numpy())

            # 计算梯度
            loss_total = loss_contextual + loss_sequential
            loss_total.backward()

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            optimizer_scheduler.step()

            # 统计我们预测对了多少
            _, pred_sequential = torch.max(logit_sequential, dim=-1) # [batchSize]
            _, pred_contextual = torch.max(logit_contextual, dim=-1) # [batchSize]
            correct_sequential += torch.eq(pred_sequential, target_sequential).sum().cpu()
            correct_contextual += torch.eq(pred_contextual, target_contextual).sum().cpu()

            if total_batch % 100 == 0:
                config.logger.info('epoch: {}/{}, step: {}/{}, training......, sequential mean loss is {}, sequential acc is {}, contextual mean loss is {}, contextual acc is {}'.format(epoch+1, config.num_epochs, i+1, len(sequential_trainset), np.array(list_train_loss_sequential).mean(), correct_sequential/(100*config.batch_size), np.array(list_train_loss_contextual).mean(), correct_contextual/(100*config.batch_size)))
                config.tbWriter.add_scalar('sequential/train/loss', np.array(list_train_loss_sequential).mean(), total_batch)
                config.tbWriter.add_scalar('contextual/train/loss', np.array(list_train_loss_contextual).mean(), total_batch)
                config.tbWriter.add_scalar('sequential/train/acc', correct_sequential/(100*config.batch_size), total_batch)
                config.tbWriter.add_scalar('contextual/train/acc', correct_contextual/(100*config.batch_size), total_batch)
                list_train_loss_sequential = [] # 保存每100个batch的平均loss
                list_train_loss_contextual = [] # 保存每100个batch的平均loss
                correct_sequential = 0 # 保存每100个batch对了多少
                correct_contextual = 0 # 保存每100个batch对了多少

                
            if total_batch % config.eval_step == 0:
                config.logger.warning('Iter:{}/{}, strat evaluate on testset'.format(i, len(sequential_trainset)))

                model.eval()

                test_loss_sequential, test_acc_sequential, test_loss_contextual, test_acc_contextual = eval_teacher(config, model, sequential_testset, contextual_testset)

                mean_acc = (test_acc_contextual + test_acc_sequential) / 2

                if mean_acc >= best_test_meanAcc:
                    best_test_meanAcc = mean_acc
                    torch.save(model.module.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                msg = 'epoch: {}/{}, step: {}/{}, sequential mean loss is {}, sequential acc is {}, contextual mean loss is {}, contextual acc is {}. {}'.format(epoch+1, config.num_epochs, i+1, len(sequential_trainset),test_loss_sequential, test_acc_sequential,test_loss_contextual, test_acc_contextual, improve)
                config.logger.warning(msg)
                config.tbWriter.add_scalar('sequential/test/loss', test_loss_sequential, total_batch)
                config.tbWriter.add_scalar('contextual/test/loss', test_loss_contextual, total_batch)
                config.tbWriter.add_scalar('sequential/test/acc', test_acc_sequential, total_batch)
                config.tbWriter.add_scalar('contextual/test/acc', test_acc_contextual, total_batch)

                model.train()

            total_batch += 1

            if (total_batch - last_improve) > config.require_improvement:
                config.logger.info("No optimization for a long time, auto-stopping...")
                stop = True
                break
        
        
        if stop:
            break

    config.logger.info("******FINISH TRAINING******")
            

def eval_teacher(config:DiCoSConfig, model:Teacher, sequential_testset, contextual_testset):

    loss_f = nn.CrossEntropyLoss()

    list_test_loss_sequential = [] # 保存每个batch的平均loss
    list_test_loss_contextual = [] # 保存每个batch的平均loss
    correct_sequential = 0 # 保存此epoch到目前为止对了多少
    correct_contextual = 0 # 保存此epoch到目前为止对了多少
    
    for i, (sequential_data, contextual_data) in enumerate(zip(sequential_testset, contextual_testset)):

        # 构建输入
        sentencesTokens = torch.cat([sequential_data['sentencesTokens'], contextual_data['sentencesTokens']], dim=0)
        attentionMask = torch.cat([sequential_data['attentionMask'], contextual_data['attentionMask']], dim=0)

        # 转移到对应设备
        sentencesTokens = sentencesTokens.to(config.device)
        attentionMask = attentionMask.to(config.device)

        # 得到输出
        logit_sequential, logit_contextual = model(sentencesTokens, attentionMask) # [batchSize*2, 2], [batchSize*2, 2]

        # 取出两个任务各自的分类结果
        logit_sequential = logit_sequential[:int(sentencesTokens.shape[0]/2)] # [batchSize, 2]
        logit_contextual = logit_contextual[int(sentencesTokens.shape[0]/2):] # [batchSize, 2]

        # 把label放到对应的设备上
        target_sequential = sequential_data['label'].to(config.device) # [batchSize]
        target_contextual = contextual_data['label'].to(config.device) # [batchSize]

        # 计算损失
        loss_sequential = loss_f(logit_sequential, target_sequential)
        loss_contextual = loss_f(logit_contextual, target_contextual)

        # 保存损失
        list_test_loss_sequential.append(loss_sequential.cpu().detach().numpy())
        list_test_loss_contextual.append(loss_contextual.cpu().detach().numpy())

        # 统计我们预测对了多少
        _, pred_sequential = torch.max(logit_sequential, dim=-1) # [batchSize]
        _, pred_contextual = torch.max(logit_contextual, dim=-1) # [batchSize]
        correct_sequential += torch.eq(pred_sequential, target_sequential).sum().cpu()
        correct_contextual += torch.eq(pred_contextual, target_contextual).sum().cpu()

    return np.array(list_test_loss_sequential).mean(), correct_sequential/(config.batch_size*len(sequential_testset)), np.array(list_test_loss_contextual).mean(), correct_contextual/(config.batch_size*len(contextual_testset))
