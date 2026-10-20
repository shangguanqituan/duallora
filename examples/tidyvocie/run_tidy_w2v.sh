#!/bin/bash

# Author: Adapted for TidyVoice Challenge (W2V-BERT + LoRA Strategy)

. ./path.sh || exit 1

stage=1
stop_stage=9

# 分布式训练设置
HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2026

# 数据路径 (与 run_tidy.sh 保持一致)
data=data
data_type="shard"  # shard/raw

# ================= 配置区域 =================

# [Stage 1] 跨语言适应 (LoRA, Frozen Encoder)
config_s1=conf/tidy_w2v_stage1_lora.yaml
exp_dir_s1=exp/tidy_w2vbert_lora_s1

# [Stage 2] 大间隔微调 (LMFT, LoRA, Frozen Encoder)
config_s2=conf/tidy_w2v_stage2_lmft.yaml
exp_dir_s2=exp/tidy_w2vbert_lora_s2_lmft

# GPU 设置
gpus="[0,1,2,3,4,5,6,7]"
checkpoint=

# 评测设置
eval_dataset="tidyvoice_dev"
trials="trials.kaldi"
score_norm_method="asnorm"
top_n=300

# ===========================================

. tools/parse_options.sh || exit 1

# -------------------------------------------------------------------
# Stage 1-2: 数据准备 (直接复用 run_tidy.sh 的逻辑)
# -------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ -z "${TIDYVOICE_API_KEY:-}" ]; then
    echo "TIDYVOICE_API_KEY must be exported before data preparation."
    exit 1
  fi
  echo "Stage 1: Prepare TidyVoice datasets ..."
  ./local/prepare_data.sh --stage 1 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Convert data to ${data_type} and LMDB ..."
  # 处理 TidyVoice 训练集和测试集
  for dset in tidyvoice_train $eval_dataset; do
    if [ ! -f "${data}/$dset/wav.scp" ]; then continue; fi

    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 --prefix shards --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done

  # 处理噪音数据 (MUSAN/RIRS)
  if [ -f "${data}/musan/wav.scp" ]; then
    python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  fi
  if [ -f "${data}/rirs/wav.scp" ]; then
    python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
  fi
fi

# -------------------------------------------------------------------
# Stage 3: Stage 1 训练 (LoRA Adaptation)
# -------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Start Stage 1 Training (LoRA Adaptation)..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  # 注意：这里不需要传入 checkpoint，因为 config 中已经指定了 model_init 为 W2V 预训练模型
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=${job_id}_s1 --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config_s1 \
      --exp_dir ${exp_dir_s1} \
      --gpus $gpus \
      --num_avg 1 \
      --data_type "${data_type}" \
      --train_data ${data}/tidyvoice_train/${data_type}.list \
      --train_label ${data}/tidyvoice_train/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb
fi

# -------------------------------------------------------------------
# Stage 4: Stage 1 模型平均、提取特征、评估
# -------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Average and Evaluate Stage 1 Model..."
  avg_model_s1=$exp_dir_s1/models/avg_model.pt
  num_avg_s1=$(yq '.num_avg' $config_s1)

  # 1. 平均模型
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model_s1 \
    --src_path $exp_dir_s1/models \
    --num ${num_avg_s1:-2}

  # 2. [新增] 提取特征 (Extract)
  # 注意：这里需要传入 data_type 和 data 路径
  echo "Extracting embeddings for Stage 1..."
  local/extract_single_dataset.sh \
      --exp_dir $exp_dir_s1 --model_path $avg_model_s1 \
      --nj 8 --gpus $gpus --data_type $data_type \
      --data ${data} --dataset $eval_dataset

  # 3. 打分 (Score)
  echo "Scoring Stage 1..."
  local/score_tidyvoice.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir_s1 \
    --eval_dataset ${eval_dataset} \
    --trials "${trials}"
    # score脚本会自动去 exp_dir 下找 embeddings，不需要传 model_path
fi

# -------------------------------------------------------------------
# Stage 5: 合并 LoRA 权重 (关键步骤)
# -------------------------------------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Merging LoRA weights and Converting keys..."

  # 1. 定义路径
  s1_avg_model=${exp_dir_s1}/models/avg_model.pt
  merged_model=${exp_dir_s1}/models/merged_avg_model.pt
  # 这是转换后，专门给 Stage 2 用的初始化模型
  merged_converted_model=${exp_dir_s1}/models/merged_avg_model_for_s2.pt

  # 2. 先执行 Merge (官方工具)
  # 这一步把 LoRA 权重吸收到 base model，得到干净的权重
  echo "Executing official merge_lora.py..."
  python tools/merge_lora.py \
      --config ${config_s1} \
      --checkpoint_in ${s1_avg_model} \
      --checkpoint_out ${merged_model}

  # 3. 再执行你的转换脚本 (加前缀)
  # 这一步把干净的权重加上 base_model.model 等前缀，以骗过 Stage 2 的加载器
  echo "Running convert_ckpt_for_lora.py to add LoRA prefixes..."
  python tools/convert_ckpt_for_lora.py \
      --input ${merged_model} \
      --output ${merged_converted_model}

  echo "Stage 5 Done. Model ready for Stage 2 init: ${merged_converted_model}"
fi

# -------------------------------------------------------------------
# Stage 6: Stage 2 训练 (LMFT)
# -------------------------------------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Start Stage 2 Training (LMFT)..."

  # 这里的 checkpoint 必须指向 Stage 1 产出的 avg model
  init_checkpoint_s2="${exp_dir_s1}/models/merged_avg_model_for_s2.pt"

  # 如果 conf/tidy_stage2_lmft.yaml 里已经写了 model_init，
  # 这里的 --checkpoint 参数会覆盖 yaml 里的配置，确保接续训练

  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=${job_id}_s2 --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config_s2 \
      --exp_dir ${exp_dir_s2} \
      --gpus $gpus \
      --num_avg 1 \
      --data_type "${data_type}" \
      --train_data ${data}/tidyvoice_train/${data_type}.list \
      --train_label ${data}/tidyvoice_train/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --model_init ${init_checkpoint_s2}
fi

# -------------------------------------------------------------------
# Stage 7: Stage 2 模型评估 (Extract + Score)
# -------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: Evaluate Final Stage 2 Model..."
  avg_model_s2=$exp_dir_s2/models/avg_model.pt

  # 1. 平均模型
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model_s2 \
    --src_path $exp_dir_s2/models \
    --num 1  # LMFT 也是只平均最后 1 个，或者你可以尝试 2-3 个

  # 2. [新增] 提取特征 (Extract)
  # 此时模型处于 LoRA 状态，extract 脚本会根据 yaml 配置加载 LoRA
  echo "Extracting embeddings for Stage 2 (LMFT)..."
  local/extract_single_dataset.sh \
      --exp_dir $exp_dir_s2 --model_path $avg_model_s2 \
      --nj 8 --gpus $gpus --data_type $data_type \
      --data ${data} --dataset $eval_dataset

  # 3. 打分 (Score)
  echo "Scoring Stage 2..."
  local/score_tidyvoice.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir_s2 \
    --eval_dataset ${eval_dataset} \
    --trials "${trials}"
fi

# -------------------------------------------------------------------
# Stage 8: 合并最终模型 (Final Merge)
# -------------------------------------------------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: Merging Final LoRA weights to create the deliverable model..."

  # 输入：Stage 2 的 Average Model (含 LoRA 权重)
  # 配置：Stage 2 的 Config (定义了 LoRA 结构)
  # 输出：Final Merged Model (无 LoRA 依赖的纯权重文件)

  final_merged_model=${exp_dir_s2}/models/final_merged_model.pt

  python tools/merge_lora.py \
      --config ${config_s2} \
      --checkpoint_in ${exp_dir_s2}/models/avg_model.pt \
      --checkpoint_out ${final_merged_model}

  echo "✅ Final TidyVoice Model saved to: ${final_merged_model}"
  echo "You can now use this model for inference without lora config."
fi

# -------------------------------------------------------------------
# Stage 9: 导出 NPY (使用最终合并后的模型)
# -------------------------------------------------------------------
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Stage 9: Exporting final embeddings to NPY..."

  # 这里使用刚刚合并好的 final_merged_model
  final_merged_model=${exp_dir_s2}/models/final_merged_model.pt

  local/extract_embeddings_to_npy.sh \
    --exp_dir ${exp_dir_s2} \
    --model_path ${final_merged_model} \
    --data ${data} \
    --dataset ${eval_dataset} \
    --output_dir ${exp_dir_s2}/embeddings_npy/${eval_dataset} \
    --gpus ${gpus}
fi
