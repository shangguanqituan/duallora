# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torch.nn.functional as F
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度取反！
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

def run_epoch(dataloader, epoch_iter, model, criterion,criterion_lid, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    print("Hard-disabling BatchNorm statistics tracking...")
    for name, module in model.named_modules():
        # [修改点] 将新加的模块加入豁免名单
        # "lid_head", "lang_pooling", "lang_bottleneck" 或者是 "pooling" (如果你希望主任务pooling的BN也更新)
        # 通常建议：新初始化的层必须豁免，预训练的层保持冻结。
        if any(x in name for x in ["lid_head", "lang_pooling", "lang_bottleneck"]):
            module.train()
            module.track_running_stats = True
            continue

        # 剩下的（主要是 Backbone）强制冻结
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            # 1. 强制 eval 模式
            module.eval()
            # 2. 【核心】强制关闭统计量追踪 -> 彻底禁止更新 running_mean/var
            module.track_running_stats = False

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()      # 总 Loss
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    loss_spk_meter = tnt.meter.AverageValueMeter()  # 主任务
    loss_lang_meter = tnt.meter.AverageValueMeter() # 语言任务
    loss_adv_meter = tnt.meter.AverageValueMeter()  # 对抗
    loss_orth_meter = tnt.meter.AverageValueMeter() # 正交

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')

    # 定义 Loss 权重 (写在 yaml 里读进来，这里先给默认值)
    w_spk = configs.get('loss_weights', {}).get('spk', 1.0)
    w_lang = configs.get('loss_weights', {}).get('lang', 0.1)
    w_orth = configs.get('loss_weights', {}).get('orth', 0.1)
    w_adv = configs.get('loss_weights', {}).get('adv', 0.1)

    if epoch <= 1:
        w_spk = 1.0
        w_lang = 1
        w_adv = 0
        w_orth = 0
        logger.info(f"--- Epoch {epoch}: Phase 1 ---")
    elif epoch <= 2:
        w_spk = 1.0
        w_lang = 0.2
        w_adv = 0.2
        w_orth = 0
        logger.info(f"--- Epoch {epoch}: Phase 2 ---")
    else:
        w_spk = 1.0
        w_lang = 0.2
        w_adv = 0.5
        w_orth = 0
        logger.info(f"--- Epoch {epoch}: Phase 3 ---")

    logger.info(f"w_spk: {w_spk:.4f}, w_lang: {w_lang:.4f}, "
                        f"w_adv: {w_adv:.4f}, " # 新增
                        f"w_orth: {w_orth:.4f}")


    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        # 获取语言标签 (如果 dataset 配置正确，这个 key 会存在)
        targets_lang = batch.get('target_lang')
        if targets_lang is not None:
            targets_lang = targets_lang.long().to(device)
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', False):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            # outputs = model(features)  # (embed_a,embed_b) in most cases
            # embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            # outputs = model.module.projection(embeds, targets)
            # if isinstance(outputs, tuple):
            #     outputs, loss = outputs
            # else:
            #     loss = criterion(outputs, targets)

            # 说话人主任务
            emb_spk = model(features, task='spk')

            if isinstance(emb_spk, tuple):
                emb_spk = emb_spk[-1]

            # Speaker Projection & Loss (ArcFace / SubCenter)
            # Projection 层挂在 model.module 上
            outputs_spk = model.module.projection(emb_spk, targets)

            if isinstance(outputs_spk, tuple):
                outputs_spk, loss_spk = outputs_spk
            else:
                loss_spk = criterion(outputs_spk, targets)

            # 语言辅助任务
            loss_lang = torch.tensor(0.0, device=device)
            loss_adv = torch.tensor(0.0, device=device)
            emb_lang = None

            if targets_lang is not None and criterion_lid is not None:
                # 传入 task='lang'，只激活 Language LoRA
                emb_lang = model(features, task='lang')
                if isinstance(emb_lang, tuple):
                    emb_lang = emb_lang[-1]

                # Language Classification
                # 使用我们在 train.py 里挂载的 lid_head
                if hasattr(model.module, 'lid_head'):
                    outputs_lang = model.module.lid_head(emb_lang)
                    loss_lang = criterion_lid(outputs_lang, targets_lang)

                if hasattr(model.module, 'lid_head'):
                    # 梯度反转！alpha=1.0 表示全额反抗
                    emb_spk_reversed = grad_reverse(emb_spk, alpha=1.0)

                    # 喂给同一个 LID Head (共享分类器)
                    outputs_adv = model.module.lid_head(emb_spk_reversed)

                    # 计算 Loss
                    loss_adv = criterion_lid(outputs_adv, targets_lang)

            # 解耦约束
            loss_orth_feat = torch.tensor(0.0, device=device)

            # 特征正交
            if emb_lang is not None:
                # 归一化
                spk_norm = F.normalize(emb_spk, p=2, dim=1)
                lang_norm = F.normalize(emb_lang, p=2, dim=1)
                # 计算余弦相似度 (Batch, )
                cosine = torch.sum(spk_norm * lang_norm, dim=1)
                # 最小化相似度的平方 (逼近0)
                loss_orth_feat = torch.mean(cosine ** 2)

            # 总损失
            loss = (w_spk * loss_spk) + \
                   (w_lang * loss_lang) + \
                   (w_adv * loss_adv) + \
                   (w_orth * loss_orth_feat)

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs_spk.cpu().detach().numpy(), targets.cpu().numpy())

        loss_spk_meter.add(loss_spk.item())
        loss_lang_meter.add(loss_lang.item())
        loss_adv_meter.add(loss_adv.item())
        loss_orth_meter.add(loss_orth_feat.item())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))
            # 详细loss
            logger.info(
                f"Detail-> curr/avg: [SPK] {loss_spk.item():.3f}/{loss_spk_meter.value()[0]:.3f} | "
                f"[LANG] {loss_lang.item():.3f}/{loss_lang_meter.value()[0]:.3f} | "
                f"[ADV] {loss_adv.item():.3f}/{loss_adv_meter.value()[0]:.3f} | "
                f"[ORTH] {loss_orth_feat.item():.3f}/{loss_orth_meter.value()[0]:.3f}"
            )

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
