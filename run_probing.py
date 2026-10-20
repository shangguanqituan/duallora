import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import kaldiio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

# ================= 配置区域 =================
NOADV_SCP = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_res293_1.633_purespk/embeddings/tidyvoice_dev/xvector.scp"
STDADV_SCP = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_res293_1.532_nolang/embeddings/tidyvoice_dev/xvector.scp"
OURS_SCP = "/data0/sgqt7/tidychallenge/examples/tidyvocie/exp/gv4_res293_1.518/embeddings/tidyvoice_dev/xvector.scp"

BATCH_SIZE = 256
EPOCHS = 15       # 增加 epoch 让线性分类器收敛
LR = 0.005        # 线性模型可以适当调大学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# ================= 工具函数 =================

def load_data_from_kaldi(scp_path, max_limit=None):
    print(f"\nLoading data from: {scp_path} ...")
    embeddings, labels_str, speakers = [], [], []
    loader = kaldiio.load_scp(scp_path)
    
    for key, vector in tqdm(loader.items(), desc="Reading vectors"):
        # key example: id014180/en/en_31233331.wav
        try:
            parts = key.split('/')
            if len(parts) >= 2:
                spk_id = parts[0]   # 提取说话人 ID
                lang_tag = parts[1] # 提取语言标签
            else:
                continue

            embeddings.append(vector)
            labels_str.append(lang_tag)
            speakers.append(spk_id)
        except Exception as e:
            continue
            
    embeddings = np.array(embeddings)
    speakers = np.array(speakers)
    
    unique_langs = sorted(list(set(labels_str)))
    lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}
    labels_idx = np.array([lang_to_idx[l] for l in labels_str])
    
    print(f"Loaded {len(embeddings)} utterances.")
    print(f"Feature dim: {embeddings.shape[1]}")
    print(f"Number of languages: {len(unique_langs)}")
    
    return embeddings, labels_idx, speakers, len(unique_langs)

class LinearLanguageProbe(nn.Module):
    """
    纯线性探针 (Linear Probe) 
    这是证明 Disentanglement 的黄金标准：剥离掉表征中线性可分的特征
    """
    def __init__(self, input_dim, num_languages):
        super(LinearLanguageProbe, self).__init__()
        # 去掉所有的隐藏层和激活函数，只保留一层线性映射
        self.lid_head = nn.Linear(input_dim, num_languages)

    def forward(self, x):
        return self.lid_head(x)

def train_and_evaluate(embeddings, labels, speakers, num_langs, title_prefix):
    input_dim = embeddings.shape[1]
    
    # 核心修改：使用 GroupShuffleSplit 确保训练集和测试集的【说话人不重叠】
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(embeddings, labels, groups=speakers))
    
    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 使用纯线性探针
    model = LinearLanguageProbe(input_dim, num_langs).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 加入适量 weight_decay (L2正则化) 防止探针过拟合在少数异常样本上
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    print(f"Training Linear Probe for {title_prefix}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")
            
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_langs)) # 强制固定大小
    
    return cm, acc

def plot_confusion_matrix(cm, acc, title, filename):
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(10, 8))
    # 调整 vmax 为 0.8 或 0.6，如果线性探针准确率掉了很多，这样颜色对比才明显
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=0.8, cbar=True)
    
    plt.title(f"{title}\nLinear LID Acc: {acc*100:.2f}%")
    plt.xlabel('Predicted Language ID')
    plt.ylabel('True Language ID')
    plt.xticks([]) 
    plt.yticks([])
    
    plt.tight_layout()
    # plt.savefig(filename, dpi=300)
    # print(f"Saved figure to {filename}")
    plt.close()

# ================= 主程序 =================

if __name__ == "__main__":
    experiments = [
        {"name": "NoAdv", "scp": NOADV_SCP, "pdf": "cm_noadv_linear.pdf"},
        {"name": "StdAdv", "scp": STDADV_SCP, "pdf": "cm_stdadv_linear.pdf"},
        {"name": "Ours", "scp": OURS_SCP, "pdf": "cm_ours_linear.pdf"}
    ]
    
    results = {}
    
    for exp in experiments:
        name = exp["name"]
        scp_path = exp["scp"]
        pdf_path = exp["pdf"]
        
        if not os.path.exists(scp_path):
            print(f"File not found: {scp_path}. Skipping...")
            continue
            
        emb, lab, speakers, num_langs = load_data_from_kaldi(scp_path)
        cm, acc = train_and_evaluate(emb, lab, speakers, num_langs, name)
        
        plot_title = f"{name}: Linear Probe" 
        plot_confusion_matrix(cm, acc, plot_title, pdf_path)
        results[name] = acc
        print("-" * 50)
        
    print("=" * 50)
    print(f"Final Linear Probe Result Summary (LID Accuracy):")
    for name, acc in results.items():
        print(f"{name:6} Acc: {acc*100:.2f}%")
    print("=" * 50)