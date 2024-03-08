import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from models.DiCoS import DiCoS, DSS
from utils.DiCoSUtils.DiCoSEvalUtils import eval_multiWoZ

import torch.nn as nn
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DiCoS.json')
    # config.device = torch.device('cpu')
    schema = MultiWoZSchema(config)

    dss = DSS(config, schema)
    dss.load_state_dict(torch.load('./savedModels/DSS/2022-05-24-23-32-24.pth', map_location='cpu'))

    dicos = DiCoS(config, schema)
    dicos.load_state_dict(torch.load('./savedModels/DiCoS/2022-05-25-09-14-57.pth', map_location='cpu'))
    
    dss.to(config.device)
    dicos.to(config.device)

    # dss = nn.DataParallel(dss)
    # dicos = nn.DataParallel(dicos)

    jga = eval_multiWoZ(dss, dicos, config, schema)

    print(jga)