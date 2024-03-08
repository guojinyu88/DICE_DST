import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from models.DiCoS import DSS
from utils.DiCoSUtils.DSSTrainUtils import train_DSS
import torch.nn as nn
import torch
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig_DSS.json')
    config.setLogger()
    config.toLogFile()
    config.saveScript()

    schema = MultiWoZSchema(config)

    model = DSS(config, schema)
    model.to(config.device)

    model = nn.DataParallel(model)

    train_DSS(config, schema, model)




