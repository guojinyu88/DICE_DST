import sys
sys.path.append('./')

import os
from configs.DiCoSConfig import DiCoSConfig
from models.Teacher import Teacher
from utils.teacherUtils.teacherTrainUtils import train_teacher
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
    config.load_from_file('./configs/DiCoSConfig.json')
    config.setLogger()
    config.toLogFile()

    model = Teacher(config)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./savedModels/DiCoS/teacher/2022-04-22-21-12-56.pth', map_location='cpu'))
    model.to(config.device)
    # 记录模型结构
    # init_input = [torch.zeros([2, config.pad_size], dtype=torch.long, device=config.device), torch.zeros([2, config.pad_size], dtype=torch.long, device=config.device), torch.zeros([2, config.pad_size], dtype=torch.long, device=config.device)]
    # config.tbWriter.add_graph(model, init_input)
    

    # train_teacher(config, model)


