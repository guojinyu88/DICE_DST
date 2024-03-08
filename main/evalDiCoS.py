import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from models.DiCoS import DiCoS
from utils.DiCoSUtils.DiCoSTrainUtils import eval_DiCoS
from utils.DiCoSUtils.DiCoSDataLoader import loadDataSet_DiCoS

import torch.nn as nn
import torch
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if __name__ == '__main__':

    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')

    schema = MultiWoZSchema(config)

    model = DiCoS(config, schema)
    model.load_state_dict(torch.load('./savedModels/DiCoS/2022-05-26-11-09-24.pth'))
    model.to(config.device)
    model.eval()
    
    model = nn.DataParallel(model)
    testset = loadDataSet_DiCoS(config, False, schema)
    totoalLoss, mean_acc = eval_DiCoS(config, schema, model, testset)

    print(totoalLoss, mean_acc)
