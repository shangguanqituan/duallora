import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# 配色
COLOR_POS = "#2E8B57"  # 绿色（目标）
COLOR_NEG = "#CD5C5C"  # 红色（非目标）
ALPHA_FILL = 0.3
ALPHA_LINE = 0.9

def get_lang(utt_id):
    try:
        return utt_id.split('/')[1]
    except IndexError:
        return "unknown"

def load_worst_case_scores(score_file):
    """
    加载最差情形数据：
    正样本：SS-DL（同说话人-异语言） -> 理想分数应趋近 1
    负样本：DS-SL（异说话人-同语言） -> 理想分数应趋近 -1/0
    """
    pos_scores = []
    neg_scores = []
    
    # 为了演示，如果文件不存在，则生成模拟数据（如不需要可删除）
    import os
    if not os.path.exists(score_file):
        print(f"警告：文件 {score_file} 不存在，已生成模拟数据用于可视化预览。")
        if "baseline" in score_file:
            # 基线：重叠较多
            pos_scores = np.random.normal(0.4, 0.2, 5000)
            neg_scores = np.random.normal(0.3, 0.2, 5000)
        else:
            # 本文方法：分布分离更明显
            pos_scores = np.random.normal(0.7, 0.15, 5000)
            neg_scores = np.random.normal(0.1, 0.15, 5000)
        return np.array(pos_scores), np.array(neg_scores)

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            enroll, test, score, label = parts[0], parts[1], float(parts[2]), parts[3]
            
            lang1 = get_lang(enroll)
            lang2 = get_lang(test)
            
            # 兼容 Kaldi score 格式 (target/nontarget) 和 label (1/0)
            is_same_speaker = (label == 'target' or label == '1')
            is_same_lang = (lang1 == lang2)

            # 最差情形逻辑
            if is_same_speaker and not is_same_lang:  # SS-DL
                pos_scores.append(score)
            elif not is_same_speaker and is_same_lang:  # DS-SL
                neg_scores.append(score)

    return np.array(pos_scores), np.array(neg_scores)

def setup_axis(ax, title):
    """美化坐标轴"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 标题
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_ylabel("密度", fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)

def plot_stacked_density(base_pos, base_neg, ours_pos, ours_neg, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={'hspace': 0.45})

    # 统一 X 轴范围
    all_data = np.concatenate([base_pos, base_neg, ours_pos, ours_neg])
    x_min, x_max = np.percentile(all_data, 0.1), np.percentile(all_data, 99.9)
    x_lim = (x_min - 0.1, x_max + 0.1)

    # ================= 绘制基线 =================
    ax0 = axes[0]
    sns.kdeplot(
        base_neg, ax=ax0, fill=True, color=COLOR_NEG,
        alpha=ALPHA_FILL, linewidth=1.5, label='非目标（异说话人-同语言）'
    )
    sns.kdeplot(
        base_pos, ax=ax0, fill=True, color=COLOR_POS,
        alpha=ALPHA_FILL, linewidth=1.5, label='目标（同说话人-异语言）'
    )
    
    # [顶部标注] 大面积重叠区域
    overlap_x = (np.mean(base_neg) + np.mean(base_pos)) / 2
    y_max_0 = ax0.get_ylim()[1]
    
    ax0.annotate(
        '重叠区域较大',
        xy=(overlap_x, 0.5),
        xytext=(overlap_x, y_max_0 * 0.55),
        arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6, linewidth=1.2),
        fontsize=10, ha='center', color='#333', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )

    setup_axis(ax0, "官方基线 - 最差情形 EER：5.19%")
    ax0.set_xlim(x_lim)
    ax0.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95, edgecolor='gray')

    # ================= 绘制本文方法 =================
    ax1 = axes[1]
    sns.kdeplot(
        ours_neg, ax=ax1, fill=True, color=COLOR_NEG,
        alpha=ALPHA_FILL, linewidth=1.5
    )
    sns.kdeplot(
        ours_pos, ax=ax1, fill=True, color=COLOR_POS,
        alpha=ALPHA_FILL, linewidth=1.5
    )

    # [底部标注] 重叠区域减小
    center_neg = np.median(ours_neg)
    center_pos = np.median(ours_pos)
    mid_point = (center_neg + center_pos) / 2
    y_max_1 = ax1.get_ylim()[1]
    
    ax1.annotate(
        '重叠区域明显减小',
        xy=(mid_point, 0.1),
        xytext=(mid_point, y_max_1 * 0.55),
        arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6, linewidth=1.2),
        fontsize=10, ha='center', color='#2F4F4F', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )

    setup_axis(ax1, "本文方法 - 最差情形 EER：1.62%")
    ax1.set_xlabel("余弦相似度得分", fontsize=12, fontweight='bold')
    ax1.set_xlim(x_lim)

    # 保存
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至：{save_path}")

if __name__ == "__main__":
    # 请确保路径正确
    base_file = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/samresnet34_voxblink_ft_tidy_baseline/scores/trials.kaldi.score"
    ours_file = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_w2v_0.912_off/scores/trials.kaldi.score"

    try:
        b_pos, b_neg = load_worst_case_scores(base_file)
        o_pos, o_neg = load_worst_case_scores(ours_file)
        
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK TC",
                "Noto Sans CJK JP",
                "DejaVu Sans"
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        plot_stacked_density(
            b_pos, b_neg, o_pos, o_neg,
            "score_distribution_comparison.pdf"
        )

        plot_stacked_density(b_pos, b_neg, o_pos, o_neg, "score_distribution_comparison.pdf")
        
    except Exception as e:
        print(f"错误：{e}")