import os
import random
import numpy as np
import kaldiio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import defaultdict

# ================= 配置区域 =================
SCP_PATH = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/debug4/embeddings/tidyvoice_dev/xvector.scp"
DATA_LIST_PATH = "/data0/sgqt7/tidychallenge/examples/tidyvocie/data/tidyvoice_dev/wav.scp"
NUM_SPEAKERS_TO_PLOT = 20
MAX_UTTS_PER_SPK = 100
SEED = 42
# ===========================================

def parse_info_from_key(key_str):
    try:
        parts = key_str.replace('\\', '/').split('/')
        if len(parts) >= 3:
            return parts[0], parts[1] # spk, lang
    except:
        pass
    return "unknown", "unknown"

def load_data_top_speakers(scp_path, list_path):
    print(f"Reading labels from {list_path}...")

    # 1. 构建索引: spk -> lang -> [keys]
    data_index = defaultdict(lambda: defaultdict(list))

    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            key = parts[0]
            spk, lang = parse_info_from_key(key)
            if spk != "unknown":
                data_index[spk][lang].append(key)

    # 2. 筛选候选人
    candidates = []
    for spk, langs_dict in data_index.items():
        if len(langs_dict) >= 2:
            total_utts = sum(len(keys) for keys in langs_dict.values())
            candidates.append((spk, total_utts))

    print(f"Found {len(candidates)} speakers with multiple languages.")

    # 3. 按语音数量倒序排列，取前 20 名
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) > NUM_SPEAKERS_TO_PLOT:
        top_candidates = candidates[:NUM_SPEAKERS_TO_PLOT]
        selected_spks = [x[0] for x in top_candidates]
        print("Selected Top 20 speakers by utterance count:")
        for spk, count in top_candidates:
            print(f" - {spk}: {count} utts")
    else:
        selected_spks = [x[0] for x in candidates]

    print(f"Selected {len(selected_spks)} speakers for visualization.")

    # 4. 读取特征
    print(f"Loading embeddings...")
    needed_keys = set()
    for spk in selected_spks:
        langs = list(data_index[spk].keys())
        limit_per_lang = MAX_UTTS_PER_SPK // len(langs)

        for lang in langs:
            keys = data_index[spk][lang]
            if len(keys) > limit_per_lang:
                selected_keys = random.sample(keys, limit_per_lang)
            else:
                selected_keys = keys
            for k in selected_keys:
                needed_keys.add(k)

    # 5. 从 SCP 读取数据
    X_list = []
    spk_list = []
    lang_list = []

    with kaldiio.ReadHelper(f'scp:{scp_path}') as reader:
        for key, numpy_array in reader:
            if key in needed_keys:
                spk, lang = parse_info_from_key(key)
                X_list.append(numpy_array)
                spk_list.append(spk)
                lang_list.append(lang) # <--- 这里修正了！之前错写成了 spk_list.append(lang)

    return np.array(X_list), np.array(spk_list), np.array(lang_list)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. 加载数据
    X, y_spk, y_lang = load_data_top_speakers(SCP_PATH, DATA_LIST_PATH)

    if len(X) == 0:
        print("No data found.")
        return

    print(f"Running t-SNE on {X.shape[0]} samples...")

    # 2. t-SNE
    curr_perplexity = 30
    if X.shape[0] < 30: curr_perplexity = max(1, X.shape[0] - 1)

    tsne = TSNE(n_components=2, perplexity=curr_perplexity, init='pca', random_state=SEED)
    X_embedded = tsne.fit_transform(X)

    # 3. 全局打乱
    perm = np.random.permutation(len(X))
    X_embedded_shuffled = X_embedded[perm]
    y_spk_shuffled = y_spk[perm]
    y_lang_shuffled = y_lang[perm]

    print("Data shuffled globally for plotting order.")

    # ================= 绘图美化区域 (修改版：竖排) =================

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,  # 稍微调小一点字体适应单栏
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.0
    })

    # [修改点 1]：画布改为 (7, 9) 或 (8, 10)，适应单栏宽度
    # 2 行 1 列
    fig, axes = plt.subplots(2, 1, figsize=(7, 9), dpi=300)

    # ==================== (a) Colored by Speaker ====================
    n_spk = len(set(y_spk_shuffled))
    palette_spk = sns.color_palette("husl", n_spk)

    sns.scatterplot(
        x=X_embedded_shuffled[:,0], y=X_embedded_shuffled[:,1],
        hue=y_spk_shuffled,
        palette=palette_spk,
        s=40,            # 点稍微改小一点
        alpha=0.7,
        linewidth=0,
        ax=axes[0],
        legend=False,
        rasterized=True
    )

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    axes[0].set_title('(a) Colored by Speaker Identity', y=-0.10, fontsize=12, fontweight='bold')

    # ==================== (b) Colored by Language ====================
    unique_langs = list(set(y_lang_shuffled))
    palette_lang = "tab20" if len(unique_langs) > 10 else "Set1"

    sns.scatterplot(
        x=X_embedded_shuffled[:,0], y=X_embedded_shuffled[:,1],
        hue=y_lang_shuffled,
        palette=palette_lang,
        s=40,
        alpha=0.7,
        linewidth=0,
        ax=axes[1],
        rasterized=True
    )

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    axes[1].set_title('(b) Colored by Language', y=-0.10, fontsize=12, fontweight='bold')

    # ==================== 图例优化 (放在下面或侧面) ====================
    # 单栏竖排时，图例如果放右边会挤压图片，建议放在 (b) 图内部的角落，或者下面
    # 这里尝试放在 (b) 图外侧右边 (可能需要微调 bbox_to_anchor)
    axes[1].legend(
        title="Language",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False,
        ncol=1,
        fontsize=9
    )

    # 1. 先自动紧凑布局
    plt.tight_layout()

    # 2. 【关键一步】手动增加垂直间距
    # hspace 默认通常是 0.2 左右，建议设为 0.3 到 0.4
    plt.subplots_adjust(hspace=0.2)

    # 3. 保存
    plt.savefig('plot_tsne.pdf', bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    main()