import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from utils.multiWoZUtils import MultiWoZSchema
from models.Distillation import Distillation
from utils.DiCoSUtils.DistillationTrainUtils import train_DiCoS
import torch.nn as nn
import torch
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = DiCoSConfig()
    config.load_from_file('./configs/DistillationConfig.json')
    config.setLogger()
    config.toLogFile()
    config.saveScript()

    schema = MultiWoZSchema(config)

    model = Distillation(config, schema)
    # 加载teacher的参数
    model.teacher.load_state_dict(torch.load('./exp/teacher/2022-07-20-20-56-46/best.pth'))
    model.to(config.device)
    model.teacher.requires_grad_(False)

    model = nn.DataParallel(model)

    train_DiCoS(config, schema, model)
