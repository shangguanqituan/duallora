import sys
import numpy as np

# ================= 配置区域 (请修改这里) =================
# 1. 模型路径配置
file_path_1 = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_w2v_0.908_company/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-U.txt.score"
file_path_2 = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_sam100_1.024_company/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-U.txt.score"
file_path_3 = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_res293_company_1.157/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-U.txt.score"

# 2. 输出文件路径
output_path = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/fuse/scores/cali_tidyvoice_dev_asnorm300_tv26_eval-U.txt.score"

# 3. 融合权重 (建议 w1 + w2 + w3 = 1.0)
# 例如: 强模型给 0.5，两个弱模型各给 0.25
w1 = 0.33  # 对应 file_path_1
w2 = 0.33 # 对应 file_path_2
w3 = 0.33 # 对应 file_path_3

# =======================================================

def load_scores(file_path):
    """读取文件，返回 (keys, scores, lines_content)"""
    scores = []
    lines_info = [] # 存储除分数外的其他信息，用于写入
    keys = [] # 用来校验顺序

    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # 格式: enroll_path test_path score label
                if len(parts) < 3:
                    continue

                score = float(parts[2])
                scores.append(score)

                # 存储 key 用于比对: "enroll test"
                key = f"{parts[0]} {parts[1]}"
                keys.append(key)

                # 存储整行信息方便重写
                lines_info.append(parts)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        sys.exit(1)

    return keys, np.array(scores), lines_info

# 1. 加载数据
print("正在加载数据...")
keys_1, scores_1, content_template = load_scores(file_path_1) # 使用第一个文件作为写入模板
keys_2, scores_2, _ = load_scores(file_path_2)
keys_3, scores_3, _ = load_scores(file_path_3)

# 2. 校验三个文件是否对齐
print("正在校验文件对齐情况...")
if not (keys_1 == keys_2 == keys_3):
    print("错误：三个文件的测试对顺序不一致！无法直接按行融合。")
    print(f"文件1数量: {len(keys_1)}, 文件2数量: {len(keys_2)}, 文件3数量: {len(keys_3)}")
    # 简单的长度检查
    if len(keys_1) != len(keys_2) or len(keys_1) != len(keys_3):
        print("提示：文件行数不同。")
    sys.exit(1)
else:
    print(f"校验通过：共 {len(scores_1)} 条测试对。")

# 3. Min-Max 归一化 (关键步骤)
def normalize(s):
    # 防止分母为0
    if s.max() == s.min():
        return s
    return (s - s.min()) / (s.max() - s.min())

print("正在归一化分数...")
norm_1 = normalize(scores_1)
norm_2 = normalize(scores_2)
norm_3 = normalize(scores_3)

# 4. 加权融合
print(f"正在融合 (权重: {w1}, {w2}, {w3})...")
fused_scores = (w1 * norm_1) + (w2 * norm_2) + (w3 * norm_3)

# 5. 写入新文件
print(f"写入结果到 {output_path} ...")
with open(output_path, 'w') as f:
    for i, parts in enumerate(content_template):
        # parts[0]=enroll, parts[1]=test, parts[2]=old_score, parts[3]=label
        # 我们用新的分数替换 parts[2]
        # 注意: 如果 label 存在 parts[3], 否则只写前三列
        label = parts[3] if len(parts) > 3 else ""

        new_line = f"{parts[0]} {parts[1]} {fused_scores[i]:.6f} {label}\n"
        f.write(new_line)

print("完成！")