import torch
import collections
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Convert checkpoint keys for LoRA structure")
    parser.add_argument('--input', required=True, help='Path to input checkpoint (.pt/.pth)')
    parser.add_argument('--output', required=True, help='Path to output checkpoint (.pt/.pth)')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    input_ckpt = args.input
    output_ckpt = args.output

    # LoRA 目标模块 (必须与 yaml 中的 target_modules 一致)
    # 建议：如果 target_modules 经常变，也可以考虑做成参数传入
    target_modules = ["linear_q", "linear_v"]

    if not os.path.exists(input_ckpt):
        print(f"Error: Input file not found: {input_ckpt}")
        exit(1)

    print(f"Loading checkpoint from {input_ckpt}...")
    ckpt = torch.load(input_ckpt, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
        wrapper = True
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        wrapper = True
    else:
        state_dict = ckpt
        wrapper = False

    new_state_dict = collections.OrderedDict()

    print("Converting keys for LoRA structure...")
    for k, v in state_dict.items():
        # 1. 处理前缀 (W2V-BERT 被 PEFT 包装了一层)
        if k.startswith("frontend.encoder."):
            # 原始：frontend.encoder.feature_projection...
            # 目标：frontend.encoder.base_model.model.feature_projection...
            suffix = k.replace("frontend.encoder.", "")
            new_k = f"frontend.encoder.base_model.model.{suffix}"
        else:
            new_k = k

        # 2. 处理 LoRA 层的内部结构 (weight -> base_layer.weight)
        # 检查这个 key 是否属于我们要 Apply LoRA 的层
        is_lora_target = any(t in new_k for t in target_modules)

        if is_lora_target:
            # 如果是 weight 或 bias，需要移到 base_layer 下
            if new_k.endswith(".weight"):
                new_k = new_k.replace(".weight", ".base_layer.weight")
                # print(f"  [LoRA Map] {k} -> {new_k}") # 减少日志打印量
            elif new_k.endswith(".bias"):
                new_k = new_k.replace(".bias", ".base_layer.bias")
                # print(f"  [LoRA Map] {k} -> {new_k}")

        new_state_dict[new_k] = v

    # 保存
    if wrapper:
        ckpt['model'] = new_state_dict
        torch.save(ckpt, output_ckpt)
    else:
        torch.save(new_state_dict, output_ckpt)

    print(f"✅ Converted model saved to: {output_ckpt}")

if __name__ == '__main__':
    main()