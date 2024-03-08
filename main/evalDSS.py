from statistics import mode
import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from models.DiCoS import DSS
from utils.DiCoSUtils.DSSTrainUtils import eval_DSS
from utils.DiCoSUtils.DSSDataLoader import loadDataSet_DSS

import torch.nn as nn
import torch
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DSS.json')

    schema = MultiWoZSchema(config)

    model = DSS(config, schema)
    model.load_state_dict(torch.load('./savedModels/DSS/2022-05-24-23-32-24.pth'))
    model.to(config.device)
    model.eval()
    
    model = nn.DataParallel(model)
    testset = loadDataSet_DSS(config, False, schema)
    
    _, _, _, _, _, preAcc_update, _, preRecall_update, _, _, preF1_update, _, _, _, _, _, _, _, F1_joint_update, _ = eval_DSS(config, schema, model, testset)
    print(preAcc_update, preRecall_update, preF1_update, F1_joint_update)
