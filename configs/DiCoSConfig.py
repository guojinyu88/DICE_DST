from distutils.command.build_scripts import first_line_re
import sys
sys.path.append('./')

from logging import Logger
from transformers import BertTokenizer, AlbertTokenizer
import torch
import json
import os
from pathlib import Path
from utils.logger import get_logger
import time
from torch.utils.tensorboard import SummaryWriter


class DiCoSConfig():
    def __init__(self):
        # basic param
        self.exp_purpose = None
        self.train_path = None
        self.test_path = None
        self.save_path = None
        self.log_path = None
        self.tensorBoard_path = None
        self.script_path = None
        self.require_improvement = None
        self.num_epochs = None
        self.batch_size = None
        self.pad_size = None
        self.eval_step = None
        self.basic_learning_rate = None
        self.encoder_learning_rate = None
        self.bert_path = None
        self.hidden_size = None

        self.logger = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        # 优化器参数
        self.rate_warmup_steps = None

        # dataLoader 参数
        self.shuffle = None
        self.drop_last = None
        self.num_workers = None
        self.teacherMultiSample = None

        # 附加参数
        self.schema_path = None # schema的地址
        self.testset_update_pred_path = None # 测试集预分类的分类文件
        self.slotTypeNum = None # 我们关注得slot数目
        self.dict_update = None # 分类依据
        self.maxSlotValue = None # 分类情况下最多的一个slot有多少种答案
        self.domain = None # 我们关注的领域
        self.num_multiHead = None # 多头头数
        self.num_relationType = None # 关系数目
        self.num_GNNLayer = None # 图神经网络的层数
        self.track_slots = None # 我们关注哪些槽值
        self.mergeClassifier = None # 是否使用融合的classifier来对分类槽值进行预测

        # teacher 的参数
        self.dict_sequential = None # teacher 任务：判断是否是按照顺序的一组对话
        self.sequentialChangeProb = None #  teacher 任务：多大概率构造负例
        self.dict_contextual = None #  teacher 任务：判断是否所有的turn来自于同一组对话
        self.contextualChangeProb = None  # teacher 任务：多大概率构造负例




    def load_from_file(self, path:str):
        with open(path, 'r', encoding='utf-8') as in_:
            dict_config = json.load(in_)

        self.startTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) # 训练开始时间

        self.exp_purpose = dict_config['exp_purpose'] # 训练解释
        self.train_path = dict_config['train_path'] # 训练集
        self.test_path = dict_config['test_path'] # 测试集

        path = Path(dict_config['save_path']+str(self.startTime)+'/') # 实验记录保存地址
        if not path.exists():
            os.makedirs(path)

        self.save_path = str(path.joinpath('best.pth'))  # 模型训练结果
        self.log_path = str(path.joinpath('logs.log')) # 日志保存地址

        os.makedirs(path.joinpath('tensorboard/')) # 开辟文件夹
        self.tensorBoard_path = str(path.joinpath('tensorboard/')) # tensorboard 保存

        os.makedirs(path.joinpath('script/')) # 开辟文件夹
        self.script_path = str(path.joinpath('script/')) # 训练脚本保存地址

        self.require_improvement = dict_config['require_improvement'] # early stop
        self.num_epochs = dict_config['num_epochs']
        self.batch_size = dict_config['batch_size'] 
        self.pad_size = dict_config['pad_size'] # padding 之后的序列长度
        self.eval_step = dict_config['eval_step']
        self.basic_learning_rate = dict_config['basic_learning_rate']
        self.encoder_learning_rate = dict_config['encoder_learning_rate']
        self.bert_path = dict_config['bert_path']
        self.hidden_size = dict_config['hidden_size'] 

        self.tokenizer = AlbertTokenizer.from_pretrained(self.bert_path) # 构建tokenizer
        self.tokenizer.add_special_tokens(dict_config['special_tokens']) # 加入special token

        self.rate_warmup_steps = dict_config['rate_warmup_steps']

        self.shuffle = dict_config['shuffle']
        self.drop_last = dict_config['drop_last']
        self.num_workers = dict_config['num_workers']
        self.teacherMultiSample = dict_config['teacherMultiSample']

        self.schema_path = dict_config['schema_path']
        self.testset_update_pred_path = dict_config['testset_update_pred_path']
        self.slotTypeNum = dict_config['slot_type_num']
        self.dict_update = dict_config['update']
        self.maxSlotValue = dict_config['maxSlotValue']
        self.domain = dict_config['domain']
        self.num_multiHead = dict_config['num_multiHead']
        self.num_relationType = dict_config['num_relationType']
        self.num_GNNLayer = dict_config['num_GNNLayer']
        self.mergeClassifier = dict_config['mergeClassifier']

        self.dict_sequential = dict_config['sequential']
        self.sequentialChangeProb = dict_config['sequentialChangeProb']
        self.dict_contextual = dict_config['contextual']
        self.contextualChangeProb = dict_config['contextualChangeProb']

        self.track_slots = dict_config['track_slots']


    def setLogger(self, logger: Logger = None):
        
        self.tbWriter = SummaryWriter(self.tensorBoard_path)
        self.tbWriter.add_hparams({key:str(value) for key, value in self.__dict__.items()},{}) # 记录超参数
        if logger == None:
            self.logger = get_logger(str(self.startTime), self.log_path) # 获取日志器
        else:
            self.logger = logger

    def toLogFile(self):
        # 记录训练使用的超参数 
        self.logger.info("******HYPER-PARAMETERS******")
        for key in self.__dict__.keys():
            self.logger.info("{}: {}".format(key, self.__dict__[key]))
        self.logger.info("****************************")
    
    def saveScript(self):
        # 保存训练使用的脚本
        copyList = ['main','utils','models','configs']
        for file in copyList:
            os.system('cp -r ./{} {}'.format(file, self.script_path))
        

if __name__ == '__main__':
    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    config.saveScript()