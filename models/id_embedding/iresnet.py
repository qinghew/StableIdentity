import torch
from torch import nn

__all__ = ["iresnet18", "iresnet34", "iresnet50", "iresnet100", "iresnet200"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
        fc_scale = 7 * 7,
    ):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.fc_scale = fc_scale
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05,),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            # print(x.shape)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        model_dir = {
            'iresnet18': './weights/r18-backbone.pth',
            'iresnet34': './weights/r34-backbone.pth',
            'iresnet50': './weights/r50-backbone.pth',
            'iresnet100': './weights/r100-backbone.pth',
        }
        pre_trained_weights = torch.load(model_dir[arch], map_location=torch.device('cpu'))

        tmp_dict = {}
        for key in pre_trained_weights:
            # if 'features' in key or 'fc' in key:
            #     print('skip %s' % key)
            #     continue
            tmp_dict[key] = pre_trained_weights[key]

        # get 'iresnet' model layers which don't exist in 'arcxx' and insert to tmp
        model_dict = model.state_dict()
        for key in model_dict:
            if key not in tmp_dict:
                tmp_dict[key] = model_dict[key]

        model.load_state_dict(tmp_dict, strict=False)
        print("load pre-trained iresnet from %s" % model_dir[arch])

    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet18", IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet34", IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet50", IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs
    )


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet100", IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs
    )


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet(
        "iresnet200", IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs
    )


@torch.no_grad()
def identification(folder: str = './images', target_idx: int = 0):
    import os
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import kornia
    import numpy as np

    os.makedirs('crop', exist_ok=True)
    img_list = os.listdir(folder)
    img_list.sort()
    n = len(img_list)
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_matrix = torch.tensor(
            [[[1.07695457, -0.03625215, -1.56352194],
               [0.03625215, 1.07695457, -5.32134629]]],
            requires_grad=False).float().cuda()

    fid_model = iresnet50(pretrained=True).cuda().eval()

    def save_tensor_to_img(tensor: torch.Tensor, path: str, scale=255):
        tensor = tensor.permute(0, 2, 3, 1)[0]  # in [0,1]
        tensor = tensor.clamp(0, 1)
        tensor = tensor * scale
        tensor_np = tensor.cpu().numpy().astype(np.uint8)
        if tensor_np.shape[-1] == 1:  # channel dim
            tensor_np = tensor_np.repeat(3, axis=-1)
        tensor_img = Image.fromarray(tensor_np)
        tensor_img.save(path)

    feats = torch.zeros((n, 512), dtype=torch.float32).cuda()
    for idx, img_path in enumerate(img_list):
        img_pil = Image.open(os.path.join(folder, img_path)).convert('RGB')
        img_tensor = trans(img_pil).unsqueeze(0).cuda()

        # img_tensor = kornia.geometry.transform.warp_affine(img_tensor, trans_matrix, (256, 256))
        save_tensor_to_img(img_tensor / 2 + 0.5, path=os.path.join('./crop', img_path))
        img_tensor = F.interpolate(img_tensor, size=112, mode="bilinear", align_corners=True)  # to 112

        feat = fid_model(img_tensor)
        feats[idx] = feat

    target_feat = feats[target_idx].unsqueeze(0)
    cosine_sim = F.cosine_similarity(target_feat, feats, 1)
    print(cosine_sim.shape)

    print('====== similarity with %s ======' % img_list[target_idx])
    for idx in range(n):
        print('[%d] %s = %.2f' % (idx, img_list[idx], float(cosine_sim[idx].cpu())))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="arcface")
    parser.add_argument("-i", "--target_idx", type=int, default=0)
    args = parser.parse_args()

    identification(target_idx=args.target_idx)

