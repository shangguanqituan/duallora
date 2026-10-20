import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# Nature/Science 风格配色
COLOR_POS = "#2E8B57"  # SeaGreen (Target)
COLOR_NEG = "#CD5C5C"  # IndianRed (Non-target)
ALPHA_FILL = 0.3
ALPHA_LINE = 0.9

def get_lang(utt_id):
    try:
        return utt_id.split('/')[1]
    except IndexError:
        return "unknown"

def load_worst_case_scores(score_file):
    """
    加载 Worst Case:
    Positives: SS-DL (同人异语) -> 理想分数应趋近 1
    Negatives: DS-SL (异人同语) -> 理想分数应趋近 -1/0
    """
    pos_scores = []
    neg_scores = []
    
    # 为了演示，如果文件不存在，生成模拟数据（你可以删除这部分）
    import os
    if not os.path.exists(score_file):
        print(f"Warning: File {score_file} not found. Generating dummy data for visual preview.")
        if "baseline" in score_file:
            # Baseline: 很多重叠
            pos_scores = np.random.normal(0.4, 0.2, 5000)
            neg_scores = np.random.normal(0.3, 0.2, 5000)
        else:
            # Ours: 分离
            pos_scores = np.random.normal(0.7, 0.15, 5000)
            neg_scores = np.random.normal(0.1, 0.15, 5000)
        return np.array(pos_scores), np.array(neg_scores)

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            enroll, test, score, label = parts[0], parts[1], float(parts[2]), parts[3]
            
            lang1 = get_lang(enroll)
            lang2 = get_lang(test)
            
            # 兼容 Kaldi score 格式 (target/nontarget) 和 label (1/0)
            is_same_speaker = (label == 'target' or label == '1')
            is_same_lang = (lang1 == lang2)

            # Worst Case Logic
            if is_same_speaker and not is_same_lang: # SS-DL
                pos_scores.append(score)
            elif not is_same_speaker and is_same_lang: # DS-SL
                neg_scores.append(score)

    return np.array(pos_scores), np.array(neg_scores)

def setup_axis(ax, title):
    """美化坐标轴"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 标题字体稍微调大一点点，加粗
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)

def plot_stacked_density(base_pos, base_neg, ours_pos, ours_neg, save_path):
    # [修改点 1] 增大 hspace (0.25 -> 0.45)，拉开上下图距离，防止拥挤
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={'hspace': 0.45})

    # 统一 X 轴范围
    all_data = np.concatenate([base_pos, base_neg, ours_pos, ours_neg])
    x_min, x_max = np.percentile(all_data, 0.1), np.percentile(all_data, 99.9)
    x_lim = (x_min - 0.1, x_max + 0.1)

    # ================= 绘制 Baseline =================
    ax0 = axes[0]
    sns.kdeplot(base_neg, ax=ax0, fill=True, color=COLOR_NEG, 
                alpha=ALPHA_FILL, linewidth=1.5, label='Non-target (Diff Spk - Same Lang)')
    sns.kdeplot(base_pos, ax=ax0, fill=True, color=COLOR_POS, 
                alpha=ALPHA_FILL, linewidth=1.5, label='Target (Same Spk - Diff Lang)')
    
    # [Top Annotation] Large Overlap Area
    overlap_x = (np.mean(base_neg) + np.mean(base_pos)) / 2
    y_max_0 = ax0.get_ylim()[1]
    
    # [修改点 2] 加上 bbox 边框，保持风格统一
    ax0.annotate('Large Overlap Area', 
                 xy=(overlap_x, 0.5),             
                 xytext=(overlap_x, y_max_0 * 0.55), # 稍微再提一点高度
                 arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6, linewidth=1.2),
                 fontsize=10, ha='center', color='#333', fontweight='bold', # 加粗
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)) # 加上边框

    # [修改点 3] 标题简化 + 数值对齐 (5.33 -> 5.19)
    setup_axis(ax0, r"Baseline - Worst Case EER: 5.19%")
    ax0.set_xlim(x_lim)
    ax0.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95, edgecolor='gray')

    # ================= 绘制 Ours =================
    ax1 = axes[1]
    sns.kdeplot(ours_neg, ax=ax1, fill=True, color=COLOR_NEG, 
                alpha=ALPHA_FILL, linewidth=1.5) 
    sns.kdeplot(ours_pos, ax=ax1, fill=True, color=COLOR_POS, 
                alpha=ALPHA_FILL, linewidth=1.5)

    # [Bottom Annotation] Reduced Overlap Area
    center_neg = np.median(ours_neg)
    center_pos = np.median(ours_pos)
    mid_point = (center_neg + center_pos) / 2
    y_max_1 = ax1.get_ylim()[1]
    
    ax1.annotate('Reduced Overlap Area', 
                 xy=(mid_point, 0.1),                
                 xytext=(mid_point, y_max_1 * 0.55),  
                 arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6, linewidth=1.2),
                 fontsize=10, ha='center', color='#2F4F4F', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # [修改点 3] 标题简化
    setup_axis(ax1, r"Ours - Worst Case EER: 1.62%")
    ax1.set_xlabel("Cosine Similarity Score", fontsize=12, fontweight='bold')
    ax1.set_xlim(x_lim)

    # 保存
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # 请确保路径正确
    base_file = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/samresnet34_voxblink_ft_tidy_baseline/scores/trials.kaldi.score"
    ours_file = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_w2v_0.912_off/scores/trials.kaldi.score"

    try:
        b_pos, b_neg = load_worst_case_scores(base_file)
        o_pos, o_neg = load_worst_case_scores(ours_file)
        
        plt.rcParams.update({
            "text.usetex": False, 
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        })

        plot_stacked_density(b_pos, b_neg, o_pos, o_neg, "score_distribution_comparison.pdf")
        
    except Exception as e:
        print(f"Error: {e}")