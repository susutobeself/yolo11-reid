import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms
import math
from utils.resnet50Model import ft_net
from utils import fuse_all_conv_bn
import yaml
import os

class FeatureExtractor:
    # ms='1' 单尺度特征提取（1.0倍缩放）
    def __init__(self, config_path, which_epoch='last', gpu_ids='0', ms='1'):
        # 解析配置文件
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        self.opt = argparse.Namespace()
        for key, value in config.items():
            setattr(self.opt, key, value)
        self.opt.which_epoch = which_epoch
        self.opt.gpu_ids = gpu_ids
        self.opt.ms = ms

        str_ids = self.opt.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)

        str_ms = self.opt.ms.split(',')
        self.ms = []
        for s in str_ms:
            s_f = float(s)
            self.ms.append(math.sqrt(s_f))

        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])
            cudnn.benchmark = True

        # 加载模型结构
        model_structure = ft_net(self.opt.nclasses, stride=self.opt.stride, ibn=self.opt.ibn,
                                     linear_num=self.opt.linear_num)


        self.model = self.load_network(model_structure)

        # Remove the final fc layer and classifier layer
        self.model.classifier.classifier = nn.Sequential()

        # Change to test mode
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = fuse_all_conv_bn(self.model)

        # 定义图像预处理转换
        self.h, self.w = 256, 128

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def load_network(self, network):
        save_path = os.path.join('./models', 'net_%s.pth' % self.opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            if torch.cuda.get_device_capability()[0] > 6 and len(self.opt.gpu_ids) == 1 and int(
                    torch.__version__[0]) > 1:
                torch.set_float32_matmul_precision('high')
                network = torch.compile(network, mode="default", dynamic=True)
            network.load_state_dict(torch.load(save_path))
        return network

    def fliplr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(self, img):
        if self.opt.linear_num <= 0:
            self.opt.linear_num = 2048

        img = self.data_transforms(img).unsqueeze(0)
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, self.opt.linear_num).zero_().cuda()


        for i in range(2):
            if (i == 1):
                img = self.fliplr(img)
            input_img = Variable(img.cuda())
            for scale in self.ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                          align_corners=False)
                with torch.no_grad():
                    outputs = self.model(input_img)
                ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        return ff.cpu().numpy().flatten()

