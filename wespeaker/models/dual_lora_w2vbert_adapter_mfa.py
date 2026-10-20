# Copyright (c) 2025 Qituan Shangguan (2369144677@qq.com)
# Based on original code from deeplab/3D-Speaker & WeSpeaker
# Modified for Full-Layer Dual LoRA

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers

class DisentangledLoRALinear(nn.Module):
    """
    非对称双流解耦 LoRA Linear 层
    """
    def __init__(self, in_features, out_features,
                 spk_rank=16, lang_rank=4, alpha=16,
                 enable_spk=True, enable_lang=True):
        super().__init__()

        # 1. 基础 Linear (Frozen Base)
        self.base_linear = nn.Linear(in_features, out_features)

        self.enable_spk = enable_spk
        self.enable_lang = enable_lang

        self.spk_scale = alpha / spk_rank if spk_rank > 0 else 1.0
        self.lang_scale = alpha / lang_rank if lang_rank > 0 else 1.0

        # 2. Speaker LoRA Branch
        if self.enable_spk:
            self.spk_lora_A = nn.Parameter(torch.zeros(spk_rank, in_features))
            self.spk_lora_B = nn.Parameter(torch.zeros(out_features, spk_rank))
        else:
            self.register_parameter('spk_lora_A', None)
            self.register_parameter('spk_lora_B', None)

        # 3. Language LoRA Branch
        if self.enable_lang:
            self.lang_lora_A = nn.Parameter(torch.zeros(lang_rank, in_features))
            self.lang_lora_B = nn.Parameter(torch.zeros(out_features, lang_rank))
        else:
            self.register_parameter('lang_lora_A', None)
            self.register_parameter('lang_lora_B', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.base_linear.reset_parameters()
        if self.enable_spk:
            nn.init.kaiming_uniform_(self.spk_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.spk_lora_B)
        if self.enable_lang:
            nn.init.kaiming_uniform_(self.lang_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lang_lora_B)

    def forward(self, x, task='spk'):
        # Base (Frozen)
        base_out = self.base_linear(x)

        # Speaker Task
        if task == 'spk' or task == 'inference':
            if self.enable_spk:
                # x @ A^T @ B^T
                lora_out = F.linear(F.linear(x, self.spk_lora_A), self.spk_lora_B)
                return base_out + (lora_out * self.spk_scale)
            return base_out

        # Language Task
        elif task == 'lang':
            if self.enable_lang:
                lora_out = F.linear(F.linear(x, self.lang_lora_A), self.lang_lora_B)
                return base_out + (lora_out * self.lang_scale)
            return base_out

        else:
            return base_out


class DualLoRA_W2VBert_Adapter_MFA(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int = 256,
        pooling_func: str = "ASP",
        n_mfa_layers: int = 24,
        adapter_dim: int = 128,
        dropout: float = 0.0,
        num_frontend_hidden_layers: int = 24,
        spk_rank: int = 16,
        lang_rank: int = 4,
        # [MODIFIED] 默认范围修改为 1 到 24 (覆盖全层)
        spk_layer_start: int = 1,   # 原: 12 -> 改: 1
        spk_layer_end: int = 24,    # 原: 24
        lang_layer_start: int = 1,  # 原: 6  -> 改: 1
        lang_layer_end: int = 24    # 原: 18 -> 改: 24 (或更高以确保覆盖)
    ):
        super().__init__()

        # 如果 n_mfa_layers 为 -1，自动设为 25 (24层 transformer + 1 input embedding)
        actual_feat_dim = feat_dim
        num_available_states = num_frontend_hidden_layers + 1
        if n_mfa_layers == -1:
            self.n_mfa_layers = num_available_states
        else:
            self.n_mfa_layers = n_mfa_layers

        logging.info(f"Init DualLoRA Adapter. Layers: {self.n_mfa_layers}")
        logging.info(f"Dual Config: Spk[{spk_layer_start}-{spk_layer_end}], Lang[{lang_layer_start}-{lang_layer_end}]")

        self.adapter_layers = nn.ModuleList()

        for i in range(self.n_mfa_layers):
            current_layer_idx = i + 1

            # 判断当前层是否在启用范围内
            enable_spk = (spk_layer_start <= current_layer_idx <= spk_layer_end)
            enable_lang = (lang_layer_start <= current_layer_idx <= lang_layer_end)

            # 构造 Block
            # [0]: DualLoRALinear
            # [1]: LayerNorm
            # [2]: ReLU
            # [3]: Linear
            block = nn.ModuleList([
                DisentangledLoRALinear(
                    actual_feat_dim, adapter_dim,
                    spk_rank=spk_rank, lang_rank=lang_rank,
                    enable_spk=enable_spk, enable_lang=enable_lang
                ),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(True),
                nn.Linear(adapter_dim, adapter_dim)
            ])
            self.adapter_layers.append(block)

        pooling_input_dim = adapter_dim * self.n_mfa_layers
        self.pooling = getattr(pooling_layers, pooling_func)(
            input_dim=pooling_input_dim,
            hidden_dim=adapter_dim,
        )

        pool_out_dim = self.pooling.out_dim
        self.bottleneck = nn.Linear(pool_out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 默认先冻结 Base，等待 load_pretrained_weights 解冻 LoRA
        self._freeze_base_parameters()

    def forward(self, all_hidden_states: tuple, task='spk'):
        # 确保输入状态数量足够
        hidden_states_to_use = all_hidden_states[-self.n_mfa_layers :]

        adapter_outputs = []
        for i in range(self.n_mfa_layers):
            x = hidden_states_to_use[i]
            x = self.adapter_layers[i][0](x, task=task) # DualLoRA
            x = self.adapter_layers[i][1](x) # LN
            x = self.adapter_layers[i][2](x) # ReLU
            x = self.adapter_layers[i][3](x) # Linear
            adapter_outputs.append(x)

        combined_features = torch.cat(adapter_outputs, dim=-1)
        combined_features = combined_features.transpose(1, 2)
        pooled_embedding = self.pooling(combined_features)
        pooled_embedding = self.drop(pooled_embedding)
        embedding = self.bottleneck(pooled_embedding)

        if self.training and hasattr(self, "projection"):
            dummy_loss = 0.0 * self.projection.weight.sum()
            embedding = embedding + dummy_loss

        return torch.tensor(0.0, device=embedding.device), embedding

    def _freeze_base_parameters(self):
        logging.info("Freezing base parameters, enabling LoRA and Bottleneck...")
        for name, param in self.named_parameters():
            param.requires_grad = False

            # 解冻 LoRA
            if 'lora_' in name:
                param.requires_grad = True

    def load_pretrained_weights(self, pretrained_path):
        if not pretrained_path:
            return

        logging.info(f"Loading pretrained weights from {pretrained_path}...")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return

        new_state_dict = {}

        for k, v in state_dict.items():
            # 针对 Adapter 层的特殊映射
            if 'projection' in k:
                continue
            if 'adapter_layers' in k:
                parts = k.split('.')
                # 检查是否是每一层 block 的第 0 个组件 (Linear)
                if len(parts) > 2 and parts[2] == '0':
                    new_k = k.replace('.0.weight', '.0.base_linear.weight')
                    new_k = new_k.replace('.0.bias', '.0.base_linear.bias')
                    new_state_dict[new_k] = v
                    continue

            new_state_dict[k] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        lora_keys = [k for k in missing if 'lora_' in k]
        real_missing = [k for k in missing if 'lora_' not in k]

        logging.info(f"Weights loaded. {len(lora_keys)} LoRA keys initialized from scratch.")
        if len(real_missing) > 0:
            logging.warning(f"Unexpected missing keys: {real_missing}")

# =============================================================================
# 接口函数 (Interface Function)
# =============================================================================

def DualLoRA_W2VBert(feat_dim, embed_dim, pooling_func='ASP', dropout=0.0,
                     spk_rank=16, lang_rank=4,
                     **kwargs):
    """
    Factory function for DualLoRA W2VBert Adapter.
    """
    return DualLoRA_W2VBert_Adapter_MFA(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        dropout=dropout,
        spk_rank=spk_rank,
        lang_rank=lang_rank,
        **kwargs
    )