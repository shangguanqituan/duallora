import sys
import numpy as np

# ================= 配置区域 =================
# 1. 弱模型路径 (EER 3.33)
file_weak_path = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_res293_company_1.157/scores/tv26_eval-U.txt.score"

# 2. 强模型路径 (EER 2.04 Subcenter) - 请替换为实际路径
file_strong_path = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_sam100_1.024_company/scores/tv26_eval-U.txt.score"

# 3. 输出文件路径
output_path = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/fuse/scores/tv26_eval-U-fusion.txt.score"

# 4. 融合权重 (给强模型更高的权重，建议 0.6 ~ 0.9 之间尝试)
weight = 0.5
# final_score = weight * strong + (1-weight) * weak
# ===========================================

def load_scores(file_path):
    """读取文件，返回 (keys, scores, lines_content)"""
    scores = []
    lines_info = [] # 存储除分数外的其他信息，用于写入
    keys = [] # 用来校验顺序

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # 格式: enroll_path test_path score label
            # parts[0]: enroll, parts[1]: test, parts[2]: score, parts[3]: label

            score = float(parts[2])
            scores.append(score)

            # 存储 key 用于比对: "enroll test"
            key = f"{parts[0]} {parts[1]}"
            keys.append(key)

            # 存储整行信息方便重写，只替换分数
            lines_info.append(parts)

    return keys, np.array(scores), lines_info

# 1. 加载数据
print("正在加载数据...")
keys_weak, scores_weak, _ = load_scores(file_weak_path)
keys_strong, scores_strong, content_strong = load_scores(file_strong_path)

# 2. 校验两个文件是否对齐
if keys_weak != keys_strong:
    print("错误：两个文件的测试对顺序不一致！请检查文件生成过程。")
    # 如果顺序不一致，需要写更复杂的字典匹配逻辑，这里假设顺序一致
    sys.exit(1)
else:
    print(f"校验通过：共 {len(scores_weak)} 条测试对。")

# 3. Min-Max 归一化 (关键步骤)
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

print("正在归一化...")
norm_weak = normalize(scores_weak)
norm_strong = normalize(scores_strong)

# 4. 加权融合
print(f"正在融合 (权重: {weight})...")
fused_scores = weight * norm_strong + (1 - weight) * norm_weak

# 5. 写入新文件
print(f"写入结果到 {output_path} ...")
with open(output_path, 'w') as f:
    for i, parts in enumerate(content_strong):
        # parts[0]=enroll, parts[1]=test, parts[2]=old_score, parts[3]=label
        # 我们用新的分数替换 parts[2]
        new_line = f"{parts[0]} {parts[1]} {fused_scores[i]:.6f} {parts[3]}\n"
        f.write(new_line)

print("完成！请使用计算 EER 的脚本测试新的 fused_result.score")