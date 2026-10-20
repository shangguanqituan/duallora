# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging


# def load_checkpoint(model: torch.nn.Module, path: str):
#     checkpoint = torch.load(path, map_location='cpu')
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
#                                                           strict=False)
#     for key in missing_keys:
#         logging.warning('missing tensor: {}'.format(key))
#     for key in unexpected_keys:
#         logging.warning('unexpected tensor: {}'.format(key))

def load_checkpoint(model: torch.nn.Module, path: str):
    """
    Load a checkpoint and handle potential size mismatch in the projection layer.
    """
    checkpoint = torch.load(path, map_location='cpu')
    current_state_dict = model.state_dict()

    # --- 关键新增逻辑：处理 projection.weight 尺寸不匹配 ---
    proj_key = 'projection.weight'
    if proj_key in checkpoint and proj_key in current_state_dict:
        ckpt_w = checkpoint[proj_key]
        curr_w = current_state_dict[proj_key]

        # 检查尺寸是否不匹配
        if ckpt_w.shape != curr_w.shape:
            logging.warning(f"Size mismatch for {proj_key}: "
                            f"checkpoint has shape {ckpt_w.shape}, "
                            f"current model has shape {curr_w.shape}.")

            ckpt_len = ckpt_w.shape[0]
            curr_len = curr_w.shape[0]

            # 这是从 speed-perturbed 训练 (类别数 * 3)
            # 过渡到 LMFT 训练 (原始类别数) 的情况
            if ckpt_len > curr_len:
                logging.info(f"Loading the first {curr_len} rows from checkpoint's "
                             f"projection layer.")
                # 从 checkpoint 中只取前一部分权重
                checkpoint[proj_key] = ckpt_w[:curr_len, :]

                # 同时处理 bias (如果存在)
                bias_key = 'projection.bias'
                if bias_key in checkpoint and bias_key in current_state_dict:
                    ckpt_b = checkpoint[bias_key]
                    if ckpt_b.shape[0] > curr_len:
                         checkpoint[bias_key] = ckpt_b[:curr_len]

    # 使用 strict=False 加载，它会自动处理其他缺失或多余的键
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)

    # === 这里是你要加打印的地方 ===
    # === Debug 打印 projection 加载情况 ===
    # if any('projection' in k for k in missing_keys):
    #     logging.warning(f"[DEBUG] projection layer NOT loaded from checkpoint! missing: "
    #                     f"{[k for k in missing_keys if 'projection' in k]}")
    # else:
    #     logging.info("[DEBUG] projection layer loaded successfully from checkpoint.")

    # if hasattr(model, 'projection'):
    #     w = model.projection.weight.data
    #     logging.info(f"[DEBUG] projection weight mean={w.mean().item():.6f}, std={w.std().item():.6f}")
    # === 原有 missing/unexpected 打印逻辑 ===


    # 过滤掉我们已经手动处理过的 projection 层的 "size mismatch" 伪错误
    # PyTorch 在 strict=False 时，size mismatch 仍然会报错，所以我们上面的逻辑是必须的
    # 这里的日志只是为了更清晰地展示加载情况
    final_unexpected_keys = [k for k in unexpected_keys if 'projection' not in k]

    for key in missing_keys:
        # 如果缺失的是 projection 层，说明我们是从一个没有该层的模型加载，这是正常的
        if 'projection' not in key:
            logging.warning('missing tensor: {}'.format(key))

    for key in final_unexpected_keys:
        logging.warning('unexpected tensor: {}'.format(key))


def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
