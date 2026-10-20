# Copyright (c) 2024 XiaoyiQin, Yuke Lin, Shuai Wang
# Modified for Asymmetric Dual LoRA Disentanglement (Full Stage 1-4)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers
from wespeaker.models.samresnet import SimAMBasicBlock

import logging

# =============================================================================
# 1. 核心组件: 非对称双流解耦 LoRA 卷积层
# =============================================================================
class DisentangledLoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, spk_rank=16, lang_rank=4, alpha=16, enable_lang=True):
        super().__init__()

        # 1. 基础卷积 (W_base)
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)

        self.enable_lang = enable_lang
        self.spk_rank = spk_rank
        self.lang_rank = lang_rank

        # Alpha scaling
        self.spk_scale = alpha / spk_rank
        self.lang_scale = alpha / lang_rank if lang_rank > 0 else 1.0

        # 2. Speaker LoRA Branch (B_s * A_s) - 始终存在
        self.spk_lora_A = nn.Parameter(torch.zeros(spk_rank, in_channels, kernel_size, kernel_size))
        self.spk_lora_B = nn.Parameter(torch.zeros(out_channels, spk_rank, 1, 1))

        # 3. Language LoRA Branch (B_l * A_l) - 现在所有 Stage 都启用 (如果 enable_lang=True)
        if self.enable_lang:
            self.lang_lora_A = nn.Parameter(torch.zeros(lang_rank, in_channels, kernel_size, kernel_size))
            self.lang_lora_B = nn.Parameter(torch.zeros(out_channels, lang_rank, 1, 1))
        else:
            self.register_parameter('lang_lora_A', None)
            self.register_parameter('lang_lora_B', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Speaker Init
        nn.init.kaiming_uniform_(self.spk_lora_A, a=math.sqrt(5))
        nn.init.normal_(self.spk_lora_B, mean=0.0, std=1e-4)

        # Language Init (if enabled)
        if self.enable_lang:
            nn.init.kaiming_uniform_(self.lang_lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lang_lora_B, mean=0.0, std=1e-4)

    def forward(self, x, task='spk'):
        # Base output
        base_out = self.base_conv(x)

        if task == 'spk' or task == 'inference':
            # Speaker LoRA: 始终激活
            lora_out = F.conv2d(x, self.spk_lora_A, stride=self.base_conv.stride,
                                padding=self.base_conv.padding)
            lora_out = F.conv2d(lora_out, self.spk_lora_B)
            return base_out + (lora_out * self.spk_scale)

        elif task == 'lang':
            if self.enable_lang:
                # Language LoRA
                lora_out = F.conv2d(x, self.lang_lora_A, stride=self.base_conv.stride,
                                    padding=self.base_conv.padding)
                lora_out = F.conv2d(lora_out, self.lang_lora_B)
                return base_out + (lora_out * self.lang_scale)
            else:
                return base_out

        else:
            return base_out

# =============================================================================
# 2. 辅助类: 支持 Task 的容器
# =============================================================================
class TaskSequential(nn.Sequential):
    def forward(self, x, task='spk'):
        for module in self:
            if isinstance(module, (LoRASimAMBasicBlock, TaskSequential)):
                x = module(x, task=task)
            else:
                x = module(x)
        return x

class LoRASimAMBasicBlock(SimAMBasicBlock):
    def __init__(self, in_planes, planes, stride=1, block_id=1,
                 spk_rank=16, lang_rank=4, enable_lang=True):
        nn.Module.__init__(self)

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.lambda_p = torch.tensor(1e-4)
        self.expansion = 1

        self.conv1 = DisentangledLoRAConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            spk_rank=spk_rank, lang_rank=lang_rank, enable_lang=enable_lang
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = DisentangledLoRAConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            spk_rank=spk_rank, lang_rank=lang_rank, enable_lang=enable_lang
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Shortcut 保持普通卷积
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, task='spk'):
        out = self.conv1(x, task=task)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, task=task)
        out = self.bn2(out)

        out = self.SimAM(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out

# =============================================================================
# 3. 主模型: Asymmetric Dual LoRA ResNet (ALL STAGES)
# =============================================================================
class DualLoRAResNet(nn.Module):
    def __init__(self, block, num_blocks, in_planes=64, in_ch=1,
                 spk_rank=16, lang_rank=4, **kwargs):
        super().__init__()
        self.in_planes = in_planes
        self.spk_rank = spk_rank
        self.lang_rank = lang_rank

        # Stem: 普通卷积 (Input -> Stage 1)
        self.conv1 = nn.Conv2d(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Layer 1: [MODIFIED] 现在使用 LoRA (Spk + Lang)
        self.layer1 = self._make_layer(LoRASimAMBasicBlock, in_planes, num_blocks[0], stride=1,
                                       use_lora=True, enable_lang=True)

        # Layer 2: LoRA (Spk + Lang)
        self.layer2 = self._make_layer(LoRASimAMBasicBlock, in_planes * 2, num_blocks[1], stride=2,
                                       use_lora=True, enable_lang=True)

        # Layer 3: LoRA (Spk + Lang)
        self.layer3 = self._make_layer(LoRASimAMBasicBlock, in_planes * 4, num_blocks[2], stride=2,
                                       use_lora=True, enable_lang=True)

        # Layer 4: [MODIFIED] 现在启用 Lang LoRA (enable_lang=True)
        self.layer4 = self._make_layer(LoRASimAMBasicBlock, in_planes * 8, num_blocks[3], stride=2,
                                       use_lora=True, enable_lang=True)

    def _make_layer(self, block, planes, num_blocks, stride, use_lora=False, enable_lang=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            if use_lora:
                layers.append(block(self.in_planes, planes, stride,
                                    spk_rank=self.spk_rank,
                                    lang_rank=self.lang_rank,
                                    enable_lang=enable_lang))
            else:
                layers.append(block(nn.Conv2d, nn.BatchNorm2d, self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        if use_lora:
            return TaskSequential(*layers)
        else:
            return nn.Sequential(*layers)

    def forward(self, x, task='spk'):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x, task=task) # Layer 1 now accepts task
        x = self.layer2(x, task=task)
        x = self.layer3(x, task=task)
        x = self.layer4(x, task=task)

        return x

# =============================================================================
# 4. 封装类
# =============================================================================

class DualLoRA_SimAM_ResNet_Base(nn.Module):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0.0,
                 spk_rank=16, lang_rank=4, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            raise ValueError("num_blocks must be specified")

        self.front = DualLoRAResNet(LoRASimAMBasicBlock, num_blocks, in_planes=in_planes,
                                    spk_rank=spk_rank, lang_rank=lang_rank)

        self.pooling = pooling_layers.ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(p=float(dropout))
        self.use_dropout = dropout > 0.0

        self._freeze_base_parameters()

    def forward(self, x, task='spk'):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(dim=1)
        x = self.front(x, task=task)
        x = self.pooling(x)
        if self.use_dropout:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x

    def _freeze_base_parameters(self):
        logging.info("Freezing base parameters, enabling LoRA and Bottleneck...")
        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'lora_' in name:
                param.requires_grad = True
            if 'bottleneck' in name:
                param.requires_grad = True

    def load_pretrained_weights(self, pretrained_path):
        logging.info(f"Loading pretrained weights from {pretrained_path}...")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            # [新增修改] 遇到 projection 相关的参数直接跳过，不放入 new_state_dict
            if 'projection' in k:
                continue
            # [MODIFIED] Check all layers (1, 2, 3, 4) for conv weights to remap to base_conv
            if 'conv1.weight' in k or 'conv2.weight' in k:
                # 只要是 layer1, 2, 3, 4 中的卷积，都映射到 base_conv
                if 'layer1' in k or 'layer2' in k or 'layer3' in k or 'layer4' in k:
                    new_k = k.replace('.weight', '.base_conv.weight')
                    new_state_dict[new_k] = v
                    continue
            new_state_dict[k] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        lora_keys = [k for k in missing if 'lora_' in k]
        other_keys = [k for k in missing if 'lora_' not in k]

        logging.info(f"Weights loaded summary: {len(missing)} missing keys.")
        if len(other_keys) > 0:
            logging.warning(f"Unexpected missing keys: {other_keys}")

# 具体实现类：ResNet34
class DualLoRA_SimAM_ResNet34_ASP(DualLoRA_SimAM_ResNet_Base):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0.0, spk_rank=16, lang_rank=4):
        super().__init__(
            in_planes=in_planes,
            embed_dim=embed_dim,
            acoustic_dim=acoustic_dim,
            dropout=dropout,
            spk_rank=spk_rank,
            lang_rank=lang_rank,
            num_blocks=[3, 4, 6, 3]
        )

# 具体实现类：ResNet100
class DualLoRA_SimAM_ResNet100_ASP(DualLoRA_SimAM_ResNet_Base):
    def __init__(self, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0.0, spk_rank=16, lang_rank=4):
        super().__init__(
            in_planes=in_planes,
            embed_dim=embed_dim,
            acoustic_dim=acoustic_dim,
            dropout=dropout,
            spk_rank=spk_rank,
            lang_rank=lang_rank,
            num_blocks=[6, 16, 24, 3]
        )

# =============================================================================
# 5. 接口函数
# =============================================================================

def DualLoRA_SimAM_ResNet34(feat_dim, embed_dim, pooling_func='ASP', dropout=0.0, spk_rank=16, lang_rank=4):
    return DualLoRA_SimAM_ResNet34_ASP(
        in_planes=64,
        embed_dim=embed_dim,
        acoustic_dim=feat_dim,
        dropout=dropout,
        spk_rank=spk_rank,
        lang_rank=lang_rank
    )

def DualLoRA_SimAM_ResNet100(feat_dim, embed_dim, pooling_func='ASP', dropout=0.0, spk_rank=16, lang_rank=4):
    return DualLoRA_SimAM_ResNet100_ASP(
        in_planes=64,
        embed_dim=embed_dim,
        acoustic_dim=feat_dim,
        dropout=dropout,
        spk_rank=spk_rank,
        lang_rank=lang_rank
    )

if __name__ == '__main__':
    # 测试代码
    print("Testing Full-Stage DualLoRA ResNet100...")
    model = DualLoRA_SimAM_ResNet100(feat_dim=80, embed_dim=256, spk_rank=16, lang_rank=4)

    x = torch.randn(2, 200, 80) # [B, T, F]

    # 1. 验证正向传播 (Speaker Task)
    out_spk = model(x, task='spk')
    print(f"Spk Output: {out_spk.shape}")

    # 2. 验证正向传播 (Language Task)
    out_lang = model(x, task='lang')
    print(f"Lang Output: {out_lang.shape}")

    # 3. 打印参数量对比验证
    print("\nLayer Check:")
    for name, m in model.named_modules():
        if isinstance(m, DisentangledLoRAConv2d):
            if "layer4" in name:
                print(f"{name}: Enable Lang? {m.enable_lang} (Should be True)")
            elif "layer1" in name:
                # 随机抽查 layer1
                print(f"{name}: Enable Lang? {m.enable_lang} (Should be True)")
                break