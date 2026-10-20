# 1. 定义源目录和目标目录
src_train=data/tidyvoice_train
src_dev=data/tidyvoice_dev
mini_train=data/mini_train
mini_dev=data/mini_dev

mkdir -p $mini_train $mini_dev

# ==========================================
# 制作 mini_train (取前 5 个说话人的数据)
# ==========================================
# 1. 从源 spk2utt 取前 5 行（代表5个说话人）
head -n 10 $src_train/spk2utt > $mini_train/spk2utt

# 2. 根据这 5 个说话人，生成 utt2spk
#    (匹配 spk2utt 中的说话人ID)
awk 'NR==FNR{a[$1];next} $2 in a' $mini_train/spk2utt $src_train/utt2spk > $mini_train/utt2spk

# 3. 根据生成的 utt2spk，生成 wav.scp
#    (匹配 utt2spk 中的语音ID)
awk 'NR==FNR{a[$1];next} $1 in a' $mini_train/utt2spk $src_train/wav.scp > $mini_train/wav.scp

echo "Mini Train ready: $(wc -l < $mini_train/wav.scp) utterances."

# ==========================================
# 制作 mini_dev (取前 2 个说话人的数据用于验证)
# ==========================================
head -n 2 $src_dev/spk2utt > $mini_dev/spk2utt
awk 'NR==FNR{a[$1];next} $2 in a' $mini_dev/spk2utt $src_dev/utt2spk > $mini_dev/utt2spk
awk 'NR==FNR{a[$1];next} $1 in a' $mini_dev/utt2spk $src_dev/wav.scp > $mini_dev/wav.scp

# 对于 Dev 集，还需要 trials 文件 (如果有的话，否则验证步骤会报错)
# 这里做一个假的 mini trials (全匹配) 用于跑通流程
awk '{print $1, $2, "target"}' $mini_dev/utt2spk > $mini_dev/trials/trials.kaldi

echo "Mini Dev ready: $(wc -l < $mini_dev/wav.scp) utterances."