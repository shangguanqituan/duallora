#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

def get_lang(utt_id):
    """
    从 Utterance ID 中提取语言 ID
    假设格式为: SpeakerID/LanguageID/WavName
    """
    try:
        return utt_id.split('/')[1]
    except IndexError:
        return "unknown"

def compute_eer(y_true, y_score):
    """计算 EER"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def plot_combined_distribution(categories, save_path):
    """将4个场景的分布画在同一张图上"""
    plt.figure(figsize=(12, 8))

    # 收集所有分数以确定 X 轴范围
    all_scores = []
    for scores in categories.values():
        all_scores.extend(scores)

    if not all_scores:
        print("Error: No scores found to plot.")
        return

    # 设置统一的 bin，保证分布图刻度一致
    bins = np.linspace(min(all_scores), max(all_scores), 150)

    # 定义绘图样式
    # SS: Same Speaker (Target), DS: Different Speaker (Non-target)
    # SL: Same Language, DL: Different Language
    plot_configs = [
        {'key': 'SS_SL', 'label': 'Target: Same Lang (SS-SL)', 'color': 'forestgreen', 'ls': '-'},
        {'key': 'SS_DL', 'label': 'Target: Diff Lang (SS-DL)', 'color': 'limegreen', 'ls': '--'},
        {'key': 'DS_SL', 'label': 'Non-target: Same Lang (DS-SL)', 'color': 'firebrick', 'ls': '-'},
        {'key': 'DS_DL', 'label': 'Non-target: Diff Lang (DS-DL)', 'color': 'lightcoral', 'ls': '--'}
    ]

    print("Plotting combined distribution...")
    for config in plot_configs:
        key = config['key']
        scores = categories[key]
        if len(scores) > 0:
            # 绘制线条 (histtype='step') 避免重叠遮挡
            plt.hist(scores, bins=bins, density=True, histtype='step',
                     label=config['label'], color=config['color'],
                     linestyle=config['ls'], linewidth=2)
            # 添加淡淡的填充色，增加可视性
            plt.hist(scores, bins=bins, density=True, histtype='stepfilled',
                     color=config['color'], alpha=0.1)

    plt.title("Score Distributions by Category (TidyVoice)", fontsize=14)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    print(f"  Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_scores(score_file):
    categories = {
        'SS_SL': [], # Same Speaker, Same Language
        'SS_DL': [], # Same Speaker, Diff Language
        'DS_SL': [], # Diff Speaker, Same Language
        'DS_DL': []  # Diff Speaker, Diff Language
    }

    print(f"Loading scores from: {score_file}")
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            enroll, test, score, label = parts[0], parts[1], float(parts[2]), parts[3]

            lang_enroll = get_lang(enroll)
            lang_test = get_lang(test)

            is_same_speaker = (label == 'target')
            is_same_lang = (lang_enroll == lang_test)

            if is_same_speaker:
                if is_same_lang:
                    categories['SS_SL'].append(score)
                else:
                    categories['SS_DL'].append(score)
            else: # Different Speaker
                if is_same_lang:
                    categories['DS_SL'].append(score)
                else:
                    categories['DS_DL'].append(score)

    return categories

def process_results(categories, output_dir):
    # 计算并打印 EER 表格
    scenarios = [
        ("Target: SS-DL vs Non-target: DS-DL", 'SS_DL', 'DS_DL'),
        ("Target: SS-DL vs Non-target: DS-SL", 'SS_DL', 'DS_SL'),
        ("Target: SS-SL vs Non-target: DS-DL", 'SS_SL', 'DS_DL'),
        ("Target: SS-SL vs Non-target: DS-SL", 'SS_SL', 'DS_SL'),
    ]

    print(f"\n{'='*80}")
    print(f"{'Trial Pair Category':<65} | {'EER (%)':<10}")
    print(f"{'-'*80}")

    for name, target_key, nontarget_key in scenarios:
        pos_scores = categories[target_key]
        neg_scores = categories[nontarget_key]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            print(f"{name:<65} | {'N/A':<10}")
            continue

        y_score = pos_scores + neg_scores
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)

        eer = compute_eer(y_true, y_score)
        print(f"{name:<65} | {eer:.2f}")

    # 计算 Overall EER
    all_pos = categories['SS_SL'] + categories['SS_DL']
    all_neg = categories['DS_SL'] + categories['DS_DL']
    if all_pos and all_neg:
        y_score = all_pos + all_neg
        y_true = [1] * len(all_pos) + [0] * len(all_neg)
        eer = compute_eer(y_true, y_score)
        print(f"{'-'*80}")
        print(f"{'Overall (All Targets vs All Non-targets)':<65} | {eer:.2f}")
    print(f"{'='*80}\n")

    # 绘制一张总图
    plot_path = os.path.join(output_dir, "combined_score_distribution.png")
    plot_combined_distribution(categories, plot_path)
    print(f"\nAnalysis complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <score_file>")
        sys.exit(1)

    score_file = sys.argv[1]
    output_dir = os.path.dirname(os.path.abspath(score_file))

    cats = load_scores(score_file)
    process_results(cats, output_dir)