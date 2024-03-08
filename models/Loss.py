from distutils.command.config import config
import sys

from configs.DiCoSConfig import DiCoSConfig
sys.path.append('./')

import torch
import torch.nn as nn

class MaskedNLLLoss(nn.Module):
    '''
        mask的地方 不进行监督
    '''

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        
        self.crossEntropy = nn.NLLLoss(weight) #  计算公式： avg(- pred)

    def forward(self, pred_y:torch.Tensor, target_y:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        '''
            pred_y: [batchSize, classNum]  要求值为过完softmax之后的(概率值)
            target_y: [batchSize]
            mask: [batchSize] 1的样本计算loss 0的样本不参与计算loss


            计算过程: (概率值) -[mask]-> (概率值) -[log]-> (概率值的对数) -[NLLLOSS]-> (loss值)

            return:
            loss
            mask之后的pred_y
            mask之后的target_y
        
        '''
        mask_pred_y = pred_y.masked_select(mask.unsqueeze(-1).expand_as(pred_y)==1).reshape([-1, pred_y.shape[1]]) # [maskNum, classNum]
        mask_target_y = target_y.masked_select(mask==1) # [maskNum]
        log_mask_pred_y = torch.log(mask_pred_y+1e-5) # 防止x取值为0，导致梯度nan
        if mask_pred_y.shape[0] > 0: # 有需要监督的样本，才计算loss
            return self.crossEntropy(log_mask_pred_y, mask_target_y), mask_pred_y, mask_target_y
        else: # 如果没有需要监督的东西，则loss直接返回0，否则两个空张量做交叉熵会导致loss nan
            return torch.tensor(0, dtype=torch.float32, requires_grad=True, device=mask_pred_y.device), mask_pred_y, mask_target_y

class DiCoSGeneratorLoss(nn.Module):
    '''
        用来监督dicos选出来的对话生成的slot准不准的loss
    '''
    def __init__(self, config:DiCoSConfig) -> None:
        super(DiCoSGeneratorLoss, self).__init__()
        self.config = config
        self.maskNLLLoss = MaskedNLLLoss()
        # self.error = 0

    def forward(self, score:torch.Tensor, update_slot:torch.Tensor, startProb:torch.Tensor, endProb:torch.Tensor, slotValueProb:torch.Tensor,
                cata_target:torch.Tensor, cate_mask:torch.Tensor, noncate_start:torch.Tensor, noncate_end:torch.Tensor, noncate_mask:torch.Tensor):
        '''
            score,  [batchSize, slotType, max_historyNum]
            update_slot, [batchSize, slotType] 0继承 1更新
            startProb, [updateSlotNum, slotType, seqLen] 各个词为起始词的概率              每个updateSlotNum仅仅包含一个slot的预测结果
            endProb,  [updateSlotNum, slotType, seqLen] 各个词为末尾词的概率               每个updateSlotNum仅仅包含一个slot的预测结果
            slotValueProb [updateSlotNum, slotType, maxSlotValue] 各slot的分类概率值       每个updateSlotNum仅仅包含一个slot的预测结果

            cata_target, 分类的监督信号 [batchSize, maxHistoryNum, slotType] 0表示删除槽值 其他表示更新
            cate_mask,  0表示不需要监督 1 表示需要监督 [batchSize, maxHistoryNum, slotType]
            noncate_start, 抽取开始位置得监督信号 [batchSize, maxHistoryNum, slotType]
            noncate_end, 抽取结束位置得监督信号 [batchSize, maxHistoryNum, slotType]
            noncate_mask  0代表不需要监督，1代表需要监督，2代表需要监督但是在当前句子找不到 [batchSize, maxHistoryNum, slotType]

            抽取方法可以监督更多的slot

            return:
            loss: 标量
            slotValueProb, [needSup_cls, maxSlotValue]
            cata_target,  [needSup_cls] 需要监督的分类的
            startProb,  [needSup_ext, seqLen] 
            noncate_start,  [needSup_ext] 需要监督的抽取的
            endProb,  [needSup_ext, seqLen]
            noncate_end [needSup_ext]
        '''
        # 每个slot选择的句子 用什么监督信号
        _, selectedHistoryIdx = score.max(-1) # [batchSize, slotType] 选择第几轮对话
        selectedHistoryIdx_slotType = selectedHistoryIdx.unsqueeze(-1).repeat([1, 1, self.config.slotTypeNum]) # [batchSize, slotType] -> [batchSize, slotType, slotType]

        # 维度意义： [i,j,k] -> 第i个sample的第j个slot选择的句子在分类第k个slot的时候使用的监督信号
        cata_target = cata_target.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]
        cate_mask = cate_mask.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]
        noncate_start = noncate_start.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]
        noncate_end = noncate_end.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]
        noncate_mask = noncate_mask.gather(dim=1, index=selectedHistoryIdx_slotType) # [batchSize, slotType, slotType]

        # 对无意义的监督信号进行mask
        slotmask = torch.eye(self.config.slotTypeNum, self.config.slotTypeNum, device=self.config.device) # [slotType, slotType] 对角矩阵 因为在第j个slot选择的句子中不需要监督其他slot
        slotmask = slotmask.unsqueeze(0).expand_as(cate_mask)
        cate_mask = cate_mask*slotmask
        noncate_mask = noncate_mask*slotmask

        # 仅仅取出要更新的slot的监督信号来
        update_slot = update_slot.unsqueeze(-1).expand_as(cata_target) # [batchSize, slotType] -> [batchSize, slotType, slotType]
        cata_target = cata_target.masked_select(update_slot==self.config.dict_update['update']).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]
        cate_mask = cate_mask.masked_select(update_slot==self.config.dict_update['update']).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]
        noncate_start = noncate_start.masked_select(update_slot==self.config.dict_update['update']).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]
        noncate_end = noncate_end.masked_select(update_slot==self.config.dict_update['update']).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]
        noncate_mask = noncate_mask.masked_select(update_slot==self.config.dict_update['update']).reshape([-1, self.config.slotTypeNum]) # [updateSlotNum, slotType]


        # 拉平
        cate_mask = cate_mask.reshape([-1])
        noncate_mask = noncate_mask.reshape([-1])

        slotValueProb = slotValueProb.reshape([-1, self.config.maxSlotValue])
        cata_target = cata_target.reshape([-1])

        startProb = startProb.reshape([-1, self.config.pad_size])
        noncate_start = noncate_start.reshape([-1])

        endProb = endProb.reshape([-1, self.config.pad_size])
        noncate_end = noncate_end.reshape([-1])

        # 计算loss
        cls_loss, slotValueProb, cata_target = self.maskNLLLoss.forward(slotValueProb, cata_target, cate_mask)
        start_loss, startProb, noncate_start = self.maskNLLLoss.forward(startProb, noncate_start, noncate_mask)
        end_loss, endProb, noncate_end = self.maskNLLLoss.forward(endProb, noncate_end, noncate_mask)

        # 惩罚 noncate_mask == 2
        

        loss = cls_loss + start_loss + end_loss
        # if torch.isnan(loss): # 检测nan
        #     if torch.isnan(cls_loss):
        #         self.config.tbWriter.add_histogram('error/cls/target', cata_target, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/cls/prob', slotValueProb, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/cls/pred', torch.max(slotValueProb, dim=-1).indices, global_step=self.error)
        #     if torch.isnan(start_loss):
        #         self.config.tbWriter.add_histogram('error/start/target', noncate_start, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/start/prob', startProb, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/start/pred', torch.max(startProb, dim=-1).indices, global_step=self.error)
        #     if torch.isnan(end_loss):
        #         self.config.tbWriter.add_histogram('error/end/target', noncate_end, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/end/prob', endProb, global_step=self.error)
        #         self.config.tbWriter.add_histogram('error/end/pred', torch.max(endProb, dim=-1).indices, global_step=self.error)

        #     self.error += 1
        

        return loss, slotValueProb, cata_target, startProb, noncate_start, endProb, noncate_end

if __name__ == '__main__':
    pass
