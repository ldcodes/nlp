import sys

sys.path.append(".")
from enhance_data.eda import *

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="原始数据的输入文件目录")
ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--num_aug", required=False, type=int, help="每条原始语句增强的语句数")
ap.add_argument("--alpha", required=False, type=float, help="每条语句中将会被改变的单词数占比")
args = ap.parse_args()

# 输出文件
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join

    output = join(dirname(args.input), 'eda_' + basename(args.input))

# 每条原始语句增强的语句数
num_aug = 1  # default
if args.num_aug:
    num_aug = args.num_aug

# 每条语句中将会被改变的单词数占比
alpha = 0.7  # default
if args.alpha:
    alpha = args.alpha


def gen_eda(train_orig, output_file, alpha, num_aug):
    writer = open(output_file, 'w', encoding="utf-8")
    lines = open(train_orig, 'r', encoding="utf-8").readlines()

    print("正在使用EDA生成增强语句...")
    # 每一行生成一个元组列表
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')  # 使用[:-1]是把\n去掉了
        label = parts[1]  # 改一下顺序
        sentence = parts[0]  # 改一下顺序
        if label == "4" or label == "9" or label == "2":
            aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
            for aug_sentence in aug_sentences:
                aug_sentence = aug_sentence.replace(" ", "")
                # writer.write(label + "\t" + aug_sentence + '\n')
                writer.write(aug_sentence + "\t" + label + "\n")
        else:
            writer.write(sentence + "\t" + label + "\n")
    writer.close()
    print("已生成增强语句!")
    print(output_file)


if __name__ == "__main__":
    gen_eda(args.input, output, alpha=alpha, num_aug=num_aug)
