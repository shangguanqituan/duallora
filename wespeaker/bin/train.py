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

import os
import re
from pprint import pformat

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    lang_set = set()

    if rank == 0:
        logger.info("Generating language labels from train_label keys...")

    # train_utt_spk_list 结构: [['id01/cy/x.wav', 'id01'], ['id02/en/y.wav', 'id02'], ...]
    for columns in train_utt_spk_list:
        if len(columns) > 0:
            # key 就是路径: id010001/cy/cy_30308905.wav
            path_key = columns[0]

            # 解析逻辑：取倒数第二个目录名 (和 processor.py 里的逻辑保持一致)
            try:
                # 兼容 windows/linux
                parts = path_key.replace('\\', '/').split('/')
                if len(parts) >= 2:
                    lang = parts[-2]
                    lang_set.add(lang)
            except:
                pass

    # 生成字典并排序
    sorted_langs = sorted(list(lang_set))
    lang2id_dict = {lang: i for i, lang in enumerate(sorted_langs)}
    num_languages = len(lang2id_dict)

    if rank == 0:
        if num_languages > 0:
            logger.info(f"Detected {num_languages} languages: {lang2id_dict}")
        else:
            logger.warning("No languages detected! Check if 'utt2spk' keys contain path info.")

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            lang2id_dict=lang2id_dict, # 传入这个参数，dataset 才会启动 path_to_lang_id
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))
    # For model_init, only frontend and speaker model are needed !!!
    # if configs['model_init'] is not None:
    #     logger.info('Load initial model from {}'.format(configs['model_init']))
    #     load_checkpoint(model, configs['model_init'])


    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)

    # 手动指定预训练模型路径
    pretrained_path = "/data0/sgqt7/PTMs/w2v_bert2_voxblink_official_LM.pth"
    # /data0/sgqt7/tidybaseline/examples/tidyvocie/exp/w2v_lora_1.365/models/converted_for_duallora.pt
    # /data0/sgqt7/PTMs/samresnet100_voxblink.pt
    # /data0/sgqt7/PTMs/samresnet34_voxblink.pt
    # /data0/sgqt7/PTMs/voxceleb_resnet293_LM.pt
    # /data0/sgqt7/PTMs/samresnet34_company.pt
    # /data0/sgqt7/PTMs/samresnet100_company.pt
    # /data0/sgqt7/PTMs/samresnet100_company_2.pt 40000
    # /data0/sgqt7/PTMs/resnet293_company.pt
    # /data0/sgqt7/PTMs/w2v_company_30000.pt
    # /data0/sgqt7/PTMs/w2v_company_final.pt
    # /data0/sgqt7/PTMs/w2v_bert2_voxblink_official_LM.pth

    if pretrained_path and os.path.exists(pretrained_path) and checkpoint is None:
        logger.info(f'Loading backbone weights from {pretrained_path} for Dual LoRA...')

        # 这里的 model 可能被 DDP 包裹，也可能没有
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # 如果是 DDP，调用 module 里的方法
            if hasattr(model.module, 'load_pretrained_weights'):
                model.module.load_pretrained_weights(pretrained_path)
            else:
                # 回退到普通加载 (如果不是 LoRA 模型)
                load_checkpoint(model, pretrained_path)
        else:
            # 单卡模式
            if hasattr(model, 'load_pretrained_weights'):
                model.load_pretrained_weights(pretrained_path)
            else:
                load_checkpoint(model, pretrained_path)
    elif checkpoint is None:
        logger.info('Train model from scratch ...')

    if num_languages > 0:

        embed_dim = configs['model_args']['embed_dim']
        hidden_dim = 256  # 隐藏层维度，通常设为 256 或 512
        lid_head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_languages)
        )

        # 把它挂载到 model 上，这样 DDP 和 Optimizer 都能自动管理它的参数
        model.add_module("lid_head", lid_head)
        if rank == 0:
            logger.info(f"Initialized LID Head with {num_languages} classes.")

    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        # if frontend_type == 'fbank':
        #     script_model = torch.jit.script(model)
        #     script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()

    # 修改点：如果是单卡训练(world_size=1)，不要使用DDP包装，直接用原始model
    # DDP包装在处理 Sub-Center ArcFace 等复杂Loss时，如果参数被复用，会报 ready twice 错误
    if world_size > 1:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            find_unused_parameters=True,
            broadcast_buffers=False  # 禁止 DDP 自动同步 BN 状态，减少开销并避免冲突
        )
        ddp_model._set_static_graph()
    else:
        # 单卡模式：手动创建一个伪装类，模拟 DDP 的 .module 结构
        # 这样既避免了 DDP 的 ready twice 错误，又满足了 executor.py 对 .module 的调用需求
        class LocalDDP(torch.nn.Module):
            def __init__(self, module):
                super(LocalDDP, self).__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        logger.info("Single GPU detected. Using LocalDDP wrapper to bypass DDP engine.")
        ddp_model = LocalDDP(model)

    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    criterion_lid = torch.nn.CrossEntropyLoss()

    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    scheduler_args = configs['scheduler_args']

    initial_lr = scheduler_args.get('initial_lr')
    min_lr = scheduler_args.get('min_lr')

    configs['optimizer_args']['lr'] = initial_lr if initial_lr is not None else min_lr

    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  criterion_lid,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
