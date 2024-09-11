import torch
import os.path
from os.path import join
import numpy as np
import cv2
import math

cv2.setNumThreads(0)  # 将opencv的多线程设置关闭
cv2.ocl.setUseOpenCL(False)  # 设置禁止使用opencl
'''
一个用来表示数据集的抽象类，其他所有的数据集都应该是这个类的子类，并且需要重写__len__和__getitem__
'''


class SpecDataset(torch.utils.data.Dataset):  # 继承torch.utils.data.Dataset类
    def __init__(self, opt, data_dir, dirA='input', dirB='gt', img_size=None):
        # 初始化，传入参数opt(opt是解析后得到的argparse.Namespace对象)，文件路径：data_dir，高光文件夹：dirA，diffuse文件夹dirB
        # 图片大小：img_size
        super(SpecDataset, self).__init__()  # 继承父类init，同时重写init
        self.opt = opt
        self.data_dir = data_dir
        self.dirA = dirA
        self.dirB = dirB
        self.fnsA = sorted(os.listdir(join(data_dir, dirA)))  # 得到经过排序后的数据集图片名称的列表
        # print(self.fnsA)
        self.fnsB = sorted(os.listdir(join(data_dir, dirB)))  # GT和非GT名字一致就可以省略行，改用下一行
        #self.fnsB = self.fnsA
        self.img_size = img_size  # 设置图片尺寸
        # np.random.seed(0)
        print('Load {} items in {} ...'.format(len(self.fnsA), data_dir))  # 输入加载了多少数据集

    def __getitem__(self, index):  # 定义__getitem__魔法函数
        fnA = self.fnsA[index]  # 根据index得到图片名称
        fnB = self.fnsB[index]  # 同上
        t_img = cv2.imread(join(self.data_dir, self.dirB, fnB))  # 读入GT图片
        m_img = cv2.imread(join(self.data_dir, self.dirA, fnA))  # 读入高光图片
        # print(self.data_dir)
        #if np.random.rand() < self.opt.fliplr:  # 利用np.random.rand()随机得到一个0-1浮点数，和fliplr比较，判断图片是否左右翻转
        #    t_img = cv2.flip(t_img, 1)
        #    m_img = cv2.flip(m_img, 1)
        #if np.random.rand() < self.opt.flipud:  # 利用np.random.rand()随机得到一个0-1浮点数，和fliplr比较，判断图片是否上下翻转
        #    t_img = cv2.flip(t_img, 0)
        #    m_img = cv2.flip(m_img, 0)
        if self.img_size == 'middle':  # 设置图片的尺寸
            size = (512, 512)
        elif self.img_size == 'small':
            #size = (384, 256)
            size = (512,512)
        else:
            # opencv 读入的图象是(w,h,3),
            # size = (m_img.shape[1],m_img.shape[0])(h,w,c)
            if m_img.shape[0] < m_img.shape[1]:  # 如果照片的h小于w
                size = (int(256 * m_img.shape[1] / m_img.shape[0]), 256)  # 指定size的h为256
            else:  # 如果照片的h大于w
                size = (256, int(256 * m_img.shape[0] / m_img.shape[1]))  # 指定size的w为256
        #if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) and not self.img_size is None:
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0] and t_img.shape[0] == size[1] and t_img.shape[1] == size[0]) and not self.img_size is None:
            # 如果图片的维度和size()不一致并且设置了img_size，对图片进行resize()
            scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))  #
            for i in range(0, scale):  # 对图像进行三次下采样得到(260, 390, 3)的图片
                m_img = cv2.pyrDown(m_img)
                t_img = cv2.pyrDown(t_img)

            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) or not (
                    t_img.shape[0] == size[1] and t_img.shape[1] == size[0]):
                # 使用线性插值的方式对图片resize()
                m_img = cv2.resize(m_img, size, cv2.INTER_AREA)
                t_img = cv2.resize(t_img, size, cv2.INTER_AREA)
        # 因为opencv读入图片格式是BGR，所以需要转换为RGB
        # openCV中读入的图像数据是以(h，w，c)的顺序构建数据的。并且数据的类型都为uint8.通道顺序为BGR!!!
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        #if t_img is None or t_img.size == 0 or not isinstance(t_img, np.ndarray):
        #    print("Error: Image is empty, not loaded correctly, or not a valid ndarray.")
        #else:
        #    t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
        t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)

        M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
        T = np.transpose(np.float32(t_img) / 255.0, (2, 0, 1))
        delta = M - T
        mask = 0.3 * delta[0] + 0.59 * delta[1] + 0.11 * delta[2]
        mask = np.float32(mask > 0.707 * mask.max())
        if self.opt.noise:
            M = M + np.random.normal(0, 2 / 255.0, M.shape).astype(np.float32)
            # T = T+np.random.normal(0,1/255.0,T.shape).astype(np.float32)
        data = {'input': M, 'target_t': T, 'fn': fnA[:-4], 'mask': mask}
        return data

    def __len__(self):
        return len(self.fnsA)


class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, opt, datadir, imgsize=None, groups=95, idxs=12, idxd=1, idxis=[7],
                 name="group-{:04d}-idx-{:02d}.png", freq=-1, any_valid=True):
        super(GroupDataset, self).__init__()
        self.opt = opt
        self.datadir = datadir
        self.imgsize = imgsize
        self.freq = freq

        if isinstance(groups, int):
            self.groups = list(range(1, groups + 1))
        else:
            self.groups = list(groups)
        self.idxd = idxd
        if isinstance(idxs, int):
            assert (not idxd in idxis) and idxd <= idxs
            self.idxs = []
            for idx in range(1, idxs + 1):
                if (not idx == idxd) and (not idx in idxis):
                    self.idxs.append(idx)
        else:
            self.idxs = list(idxs)
        self.name = name
        self.build(groups, any_valid)
        print('Load {} items in {} ...'.format(len(self.pairs), datadir))
        if self.freq > 0 and self.freq < 1:
            print('Select {} items ...'.format(int(len(self.pairs) * self.freq)))

    def build(self, groups, any_valid):
        self.pairs = []
        if any_valid:
            for g in self.groups:
                if os.path.exists(os.path.join(self.datadir, self.name.format(g, self.idxd))):
                    for idx in self.idxs:
                        if os.path.exists(os.path.join(self.datadir, self.name.format(g, idx))):
                            self.pairs.append(
                                {'input': self.name.format(g, idx), 'target': self.name.format(g, self.idxd)})
        else:
            for g in self.groups:
                if not os.path.exists(os.path.join(self.datadir, self.name.format(g, self.idxd))):
                    continue
                group_pairs = []
                valid = True
                for idx in self.idxs:
                    if os.path.exists(os.path.join(self.datadir, self.name.format(g, idx))):
                        group_pairs.append(
                            {'input': self.name.format(g, idx), 'target': self.name.format(g, self.idxd)})
                    else:
                        valid = False
                        break
                if valid:
                    self.pairs += group_pairs

    def __getitem__(self, index):
        if 0 < self.freq < 1:
            index = np.random.randint(len(self.pairs))
        fnA = self.pairs[index]['input']
        fnB = self.pairs[index]['target']
        m_img = cv2.imread(join(self.datadir, fnA))
        t_img = cv2.imread(join(self.datadir, fnB))
        # print(self.imgsize)
        if np.random.rand() < self.opt.fliplr:
            t_img = cv2.flip(t_img, 1)
            m_img = cv2.flip(m_img, 1)
        if np.random.rand() < self.opt.flipud:
            t_img = cv2.flip(t_img, 0)
            m_img = cv2.flip(m_img, 0)
        if self.imgsize == 'middle':
            size = (768, 512)
        elif self.imgsize == 'small':
            size = (384, 256)
        else:
            # size = (m_img.shape[1],m_img.shape[0])
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(256 * m_img.shape[1] / m_img.shape[0]), 256)
            else:
                size = (256, int(256 * m_img.shape[0] / m_img.shape[1]))
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
            for i in range(0, scale):
                m_img = cv2.pyrDown(m_img)
                t_img = cv2.pyrDown(t_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) or not (
                    t_img.shape[0] == size[1] and t_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img, size, cv2.INTER_AREA)
                t_img = cv2.resize(t_img, size, cv2.INTER_AREA)
        t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)

        M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
        T = np.transpose(np.float32(t_img) / 255.0, (2, 0, 1))
        delta = M - T
        mask = 0.3 * delta[0] + 0.59 * delta[1] + 0.11 * delta[2]
        mask = np.float32(mask > 0.507 * mask.max())
        if self.opt.noise:
            M = M + np.random.normal(0, 2 / 255.0, M.shape).astype(np.float32)
            # T = T+np.random.normal(0,1/255.0,T.shape).astype(np.float32)
        data = {'input': M, 'target_t': T, 'fn': fnA[:-4], 'mask': mask}
        return data

    def __len__(self):
        if self.freq > 0 and self.freq < 1:
            return int(len(self.pairs) * self.freq)
        else:
            return len(self.pairs)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, imgsize='small'):
        super(TestDataset, self).__init__()
        self.datadir = datadir
        self.fns = sorted(os.listdir(datadir))
        self.imgsize = imgsize
        print('Load {} items in {} ...'.format(len(self.fns), datadir))

    def __getitem__(self, index):
        fn = self.fns[index]
        m_img = cv2.imread(join(self.datadir, fn))

        # print(self.imgsize)
        assert self.imgsize in ['middle', 'small', 'origin']
        if self.imgsize == 'middle':
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(512 * m_img.shape[1] / m_img.shape[0]), 512)
            else:
                size = (512, int(512 * m_img.shape[0] / m_img.shape[1]))
        else:
            if m_img.shape[0] < m_img.shape[1]:
                size = (int(256 * m_img.shape[1] / m_img.shape[0]), 256)
            else:
                size = (256, int(256 * m_img.shape[0] / m_img.shape[1]))
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
            for i in range(0, scale):
                m_img = cv2.pyrDown(m_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img, size, cv2.INTER_AREA)

        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
        data = {'input': M, 'target_t': torch.zeros([1, 0]), 'fn': fn[:-4], 'mask': torch.zeros([1, 0])}
        return data

    def __len__(self):
        return len(self.fns)
