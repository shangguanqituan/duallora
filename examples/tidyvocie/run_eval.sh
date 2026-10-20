#!/bin/bash

# Author: 2025 Aref Farhadipour - University of Zurich
#         (areffarhadi@gmail.com, aref.farhadipour@uzh.ch)
#
# This baseline code is adapted for the TidyVoice dataset
# for the TidyVoice2026 Interspeech Challenge

. ./path.sh || exit 1

stage=5
stop_stage=7


HOST_NODE_ADDR="localhost:29405"
num_nodes=1
job_id=2025

data=data
data_type="shard"  # shard/raw

######## Resnet34 Multi Lingual
exp_dir=exp/fuse

gpus="[0,1]"
num_avg=1
checkpoint=

# ====================== EVAL DATASET SELECTION ======================

eval_dataset="tidyvoice_eval_final" # mini_dev、tidyvoice_dev, tidyvoice_eval_final

# Dataset-specific trial configurations
case $eval_dataset in
  "tidyvoice_eval1")
    trials="trials.kaldi"
    ;;
  "tidyvoice_eval2")
    trials="trials.kaldi"
    ;;
  "tidyvoice_dev")
    trials="trials.kaldi"
    ;;
  "tidyvoice_eval_final")
    trials="tv26_eval-A.txt tv26_eval-U.txt"
    ;;
  "mini_dev")
    trials="trials.kaldi"
    ;;
  *)
    echo "Warning: Unknown dataset $eval_dataset, using default trial file"
    trials="trials.kaldi"
    ;;
esac

echo "Using evaluation dataset: $eval_dataset"
echo "Using trial file: $trials"

score_norm_method="asnorm"  # asnorm/snorm
top_n=300



. tools/parse_options.sh || exit 1

# ===============================================================
# Stage 4: Extract Embeddings
# ===============================================================
# 定义模型路径 (如果变量没定义，默认拼凑一个)
if [ -z "${model_path}" ]; then
  model_path=${exp_dir}/models/avg_model.pt
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Extract embeddings for dataset: $eval_dataset ..."

  # 检查模型是否存在
  if [ ! -f "${model_path}" ]; then
      echo "Error: Model not found at ${model_path}"
      exit 1
  fi

  # 调用官方提取脚本
  # 注意：这里假设你使用的是 shard 模式，如果数据没做 shard，可能需要改参数
  # 根据你开头的 data_type="shard" 设置，这里保持一致

  local/extract_single_dataset.sh \
    --exp_dir $exp_dir \
    --model_path $model_path \
    --nj 1 --gpus $gpus \
    --data_type "${data_type}" \
    --data ${data} \
    --dataset $eval_dataset
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score evaluation dataset: $eval_dataset ..."
  local/score_tidyvoice.sh \
    --stage 1 --stop-stage 1 \
    --data ${data} \
    --exp_dir $exp_dir \
    --eval_dataset $eval_dataset \
    --trials "$trials" \
    --cal_mean True
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm for evaluation dataset: $eval_dataset ..."
  echo "Note: Using tidyvoice_dev as fixed cohort set for score normalization"
  local/score_norm_tidyvoice.sh \
    --stage 1 --stop-stage 2 \
    --score_norm_method $score_norm_method \
    --cohort_set tidyvoice_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --eval_dataset $eval_dataset \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration for evaluation dataset: $eval_dataset ..."
  echo "Note: Using tidyvoice_dev as fixed cohort set for score calibration"
  local/score_calibration.sh \
    --stage 1 --stop-stage 4 \
    --score_norm_method $score_norm_method \
    --calibration_trial "tidyvoice_dev_cali.kaldi" \
    --cohort_set tidyvoice_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi


# ===============================================================
# Stage 8: Final Submission Packaging
# ===============================================================
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: Preparing Final Submission ZIP ..."

  submission_script=/data0/sgqt7/tidychallenge/examples/tidyvocie/local/prepare_submission.py

  # 1. 自动寻找最新的校准分数文件
  # raw_cali_A=$(find ${exp_dir}/scores -name "*cali*tv26_eval-A*.score" | head -n 1)
  raw_cali_A=/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/fuse/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-A.txt.score
  # raw_cali_U=$(find ${exp_dir}/scores -name "*cali*tv26_eval-U*.score" | head -n 1)
  raw_cali_U=/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/fuse/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-U.txt.score

  if [ -z "$raw_cali_A" ] || [ ! -f "$raw_cali_A" ]; then
      echo "Error: Calibrated score file for Trial A not found in ${exp_dir}/scores/"
      exit 1
  fi
  if [ -z "$raw_cali_U" ] || [ ! -f "$raw_cali_U" ]; then
      echo "Error: Calibrated score file for Trial U not found."
      exit 1
  fi

  # 2. 格式化处理：只保留前3列
  final_sub_A="${exp_dir}/scores/final_submission_A.txt"
  final_sub_U="${exp_dir}/scores/final_submission_U.txt"

  echo "Formatting scores to 3 columns..."
  awk '{print $1, $2, $3}' "$raw_cali_A" > "$final_sub_A"
  awk '{print $1, $2, $3}' "$raw_cali_U" > "$final_sub_U"

  # 3. 动态修改 Python 打包脚本
  temp_py="local/prepare_submission_temp.py"
  cp "${submission_script}" "${temp_py}"

  # ================== 核心修改：使用你提供的绝对路径 ==================
  # 直接指定 Score 文件的绝对路径
  abs_score_A=$(readlink -f "$final_sub_A")
  abs_score_U=$(readlink -f "$final_sub_U")

  # 直接指定 Reference (标签) 文件的绝对路径 (你刚刚提供的路径)
  abs_ref_A="/data0/sgqt7/tidychallenge/examples/tidyvocie/data/download_data/TidyVoiceX_Eval_pairs/tv26_eval-A.txt"
  abs_ref_U="/data0/sgqt7/tidychallenge/examples/tidyvocie/data/download_data/TidyVoiceX_Eval_pairs/tv26_eval-U.txt"

  # 再次检查文件是否存在，防止路径敲错
  if [ ! -f "$abs_ref_A" ]; then
      echo "Error: Reference file A not found at: $abs_ref_A"
      exit 1
  fi
  if [ ! -f "$abs_ref_U" ]; then
      echo "Error: Reference file U not found at: $abs_ref_U"
      exit 1
  fi
  # ================================================================

  echo "Configuring submission script..."

  # 使用 sed 替换 Python 脚本中的路径变量
  # 替换 Score 文件路径
  sed -i "s|TV26_EVAL_A_SCORE_FILE = Path(.*)|TV26_EVAL_A_SCORE_FILE = Path(\"${abs_score_A}\")|g" ${temp_py}
  sed -i "s|TV26_EVAL_U_SCORE_FILE = Path(.*)|TV26_EVAL_U_SCORE_FILE = Path(\"${abs_score_U}\")|g" ${temp_py}

  # 替换 Reference 文件路径
  sed -i "s|TV26_EVAL_A_REF_FILE = Path(.*)|TV26_EVAL_A_REF_FILE = Path(\"${abs_ref_A}\")|g" ${temp_py}
  sed -i "s|TV26_EVAL_U_REF_FILE = Path(.*)|TV26_EVAL_U_REF_FILE = Path(\"${abs_ref_U}\")|g" ${temp_py}

  # 4. 运行打包
  echo "Running prepare_submission..."
  python3 "${temp_py}"

  # 5. 清理
  rm "${temp_py}"
  echo "-------------------------------------------------------"
  echo "Stage 8 Completed! Please check the generated ZIP file above."
  echo "-------------------------------------------------------"
fi
