# Copyright (c) 2024 Modified for Asymmetric Dual LoRA ResNet (Full Stage 1-4)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers
from wespeaker.models.resnet import BasicBlock, Bottleneck
import logging

# =============================================================================
# 1. 核心组件
# =============================================================================

class DisentangledLoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, spk_rank=16, lang_rank=4, alpha=16, enable_lang=True):
        super().__init__()
        # 1. Base Convolution
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)

        self.enable_lang = enable_lang
        self.spk_rank = spk_rank
        self.lang_rank = lang_rank

        # Alpha scaling
        self.spk_scale = alpha / spk_rank
        self.lang_scale = alpha / lang_rank if lang_rank > 0 else 1.0

        # 2. Speaker LoRA Branch
        self.spk_lora_A = nn.Parameter(torch.zeros(spk_rank, in_channels, kernel_size, kernel_size))
        self.spk_lora_B = nn.Parameter(torch.zeros(out_channels, spk_rank, 1, 1))

        # 3. Language LoRA Branch
        if self.enable_lang:
            self.lang_lora_A = nn.Parameter(torch.zeros(lang_rank, in_channels, kernel_size, kernel_size))
            self.lang_lora_B = nn.Parameter(torch.zeros(out_channels, lang_rank, 1, 1))
        else:
            self.register_parameter('lang_lora_A', None)
            self.register_parameter('lang_lora_B', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.spk_lora_A, a=math.sqrt(5))
        nn.init.normal_(self.spk_lora_B, mean=0.0, std=1e-4)
        if self.enable_lang:
            nn.init.kaiming_uniform_(self.lang_lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lang_lora_B, mean=0.0, std=1e-4)

    def forward(self, x, task='spk'):
        base_out = self.base_conv(x)

        if task == 'spk' or task == 'inference':
            lora_out = F.conv2d(x, self.spk_lora_A, stride=self.base_conv.stride,
                                padding=self.base_conv.padding)
            lora_out = F.conv2d(lora_out, self.spk_lora_B)
            return base_out + (lora_out * self.spk_scale)
        elif task == 'lang':
            if self.enable_lang:
                lora_out = F.conv2d(x, self.lang_lora_A, stride=self.base_conv.stride,
                                    padding=self.base_conv.padding)
                lora_out = F.conv2d(lora_out, self.lang_lora_B)
                return base_out + (lora_out * self.lang_scale)
            else:
                return base_out
        else:
            return base_out

class TaskSequential(nn.Sequential):
    def forward(self, x, task='spk'):
        for module in self:
            if isinstance(module, (LoRABasicBlock, LoRABottleneck, TaskSequential)):
                x = module(x, task=task)
            else:
                x = module(x)
        return x

# =============================================================================
# 2. LoRA Block 实现
# =============================================================================

class LoRABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, spk_rank=16, lang_rank=4, enable_lang=True):
        super(LoRABasicBlock, self).__init__()
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, task='spk'):
        out = F.relu(self.bn1(self.conv1(x, task=task)))
        out = self.bn2(self.conv2(out, task=task))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LoRABottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, spk_rank=16, lang_rank=4, enable_lang=True):
        super(LoRABottleneck, self).__init__()
        # Conv1: 1x1
        self.conv1 = DisentangledLoRAConv2d(
            in_planes, planes, kernel_size=1, bias=False,
            spk_rank=spk_rank, lang_rank=lang_rank, enable_lang=enable_lang
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Conv2: 3x3
        self.conv2 = DisentangledLoRAConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            spk_rank=spk_rank, lang_rank=lang_rank, enable_lang=enable_lang
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Conv3: 1x1
        self.conv3 = DisentangledLoRAConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False,
            spk_rank=spk_rank, lang_rank=lang_rank, enable_lang=enable_lang
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, task='spk'):
        out = F.relu(self.bn1(self.conv1(x, task=task)))
        out = F.relu(self.bn2(self.conv2(out, task=task)))
        out = self.bn3(self.conv3(out, task=task))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# =============================================================================
# 3. Dual LoRA ResNet 主类 (Full Stage 1-4)
# =============================================================================

class DualLoRAResNet(nn.Module):
    def __init__(self,
                 block_type,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False,
                 spk_rank=16,
                 lang_rank=4):
        super(DualLoRAResNet, self).__init__()

        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.spk_rank = spk_rank
        self.lang_rank = lang_rank

        if block_type.__name__ == 'BasicBlock':
            self.LoRABlock = LoRABasicBlock
            self.BaseBlock = BasicBlock
        elif block_type.__name__ == 'Bottleneck':
            self.LoRABlock = LoRABottleneck
            self.BaseBlock = Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Stem Layer
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        # [MODIFIED] Layer 1: 现在使用 LoRA Block
        self.layer1 = self._make_layer(self.LoRABlock, m_channels, num_blocks[0], stride=1, use_lora=True, enable_lang=True)

        # Layer 2
        self.layer2 = self._make_layer(self.LoRABlock, m_channels * 2, num_blocks[1], stride=2, use_lora=True, enable_lang=True)

        # Layer 3
        self.layer3 = self._make_layer(self.LoRABlock, m_channels * 4, num_blocks[2], stride=2, use_lora=True, enable_lang=True)

        # [MODIFIED] Layer 4: enable_lang=True (原为 False)
        self.layer4 = self._make_layer(self.LoRABlock, m_channels * 8, num_blocks[3], stride=2, use_lora=True, enable_lang=True)

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * self.BaseBlock.expansion
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)

        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

        self._freeze_base_parameters()

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
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        if use_lora:
            return TaskSequential(*layers)
        else:
            return nn.Sequential(*layers)

    def _freeze_base_parameters(self):
        logging.info("Freezing base parameters, enabling LoRA and Head...")
        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'lora_' in name:
                param.requires_grad = True
            # if 'seg_' in name:
            #     param.requires_grad = True
            # if 'pool' in name:
            #     param.requires_grad = True

    def _get_frame_level_feat(self, x, task='spk'):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze_(1)

        out = F.relu(self.bn1(self.conv1(x)))
        
        # [MODIFIED] Layer 1 现在需要传入 task
        out = self.layer1(out, task=task)
        out = self.layer2(out, task=task)
        out = self.layer3(out, task=task)
        out = self.layer4(out, task=task)

        return out

    def forward(self, x, task='spk'):
        out = self._get_frame_level_feat(x, task=task)
        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a

    def load_pretrained_weights(self, pretrained_path):
        import logging
        logging.info(f"Loading pretrained weights from {pretrained_path}...")

        state_dict = torch.load(pretrained_path, map_location='cpu')
        new_state_dict = {}

        for k, v in state_dict.items():
            if 'projection' in k or 'loss' in k:
                continue

            # [MODIFIED] 增加 'layer1' 到检查列表
            if any(x in k for x in ['layer1', 'layer2', 'layer3', 'layer4']):
                if 'downsample' not in k:
                    if 'conv' in k and k.endswith('.weight'):
                        new_k = k.replace('.weight', '.base_conv.weight')
                        new_state_dict[new_k] = v
                        continue

            new_state_dict[k] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        lora_missing = [k for k in missing if 'lora_' in k]
        proj_missing = [k for k in missing if 'projection' in k]
        real_missing = [k for k in missing if 'lora_' not in k and 'projection' not in k]

        logging.info(f"Loaded weights. Ignored {len(proj_missing)} projection keys.")
        logging.info(f"Initialized {len(lora_missing)} LoRA params from scratch.")

        if len(real_missing) > 0:
            logging.warning(f"Warning: Unexpected missing keys in backbone: {real_missing}")

# =============================================================================
# 4. ResNet293 专用接口
# =============================================================================

def DualLoRA_ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False,
                       spk_rank=16, lang_rank=4):
    return DualLoRAResNet(
        Bottleneck,
        [10, 20, 64, 3],
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        two_emb_layer=two_emb_layer,
        spk_rank=spk_rank,
        lang_rank=lang_rank
    )

if __name__ == '__main__':
    # 测试代码
    print("Testing DualLoRA_ResNet293 (Full Stage 1-4)...")
    model = DualLoRA_ResNet293(feat_dim=80, embed_dim=256, spk_rank=16, lang_rank=4)
    model.eval()

    x = torch.randn(2, 200, 80) # (B, T, F)

    _, emb_spk = model(x, task='spk')
    print(f"Spk Embedding: {emb_spk.shape}")

    _, emb_lang = model(x, task='lang')
    print(f"Lang Embedding: {emb_lang.shape}")

    # Check Layer 1 Type
    print(f"\nLayer 1 Type: {type(model.layer1)}")
    print(f"Layer 1 Block 0 Type: {type(model.layer1[0])} (Should be LoRABottleneck)")