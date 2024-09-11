from os.path import join

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data import spec
from engine import Engine
from options.specularitynet.train_options import TrainOptions
a = TrainOptions()
a.initialize()
opt = a.parse()  # 获取参数的命名空间

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

dataset_wild = spec.TestDataset('/mnt/data3/SpecularityNet-PSD-main/new_test/', imgsize='small')
dataloader_wild = DataLoader(dataset_wild, 1, num_workers=opt.nThreads, shuffle=not opt.serial_batches, drop_last=False)

engine = Engine(opt)

engine.test(dataloader_wild, savedir=join('./results', 'Keren2'))
