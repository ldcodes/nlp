import pandas as pd
import fasttext
import string
import re
import jieba
import sys



data_path = './real/data/'

tiny_mode = False
tiny_pre = 'tiny_'

keyword_num = {
    "财经" : 30,
    "时政" : 50,
    "房产" : 30,
    "科技" : 50,
    "教育" : 50,
    "时尚" : 30,
    "游戏" : 20,
    "家居" : 30,
    "体育" : 50,
    "娱乐" : 30,
}


if tiny_mode == False:
    dataset = pd.read_csv(data_path + 'labeled_data.csv')
    submission = pd.read_csv(data_path + 'submit_example.csv')
    unlabeled_data = pd.read_csv(data_path + 'unlabeled_data.csv')
else:
    dataset = pd.read_csv(data_path + tiny_pre + 'labeled_data.csv')
    submission = pd.read_csv(data_path + 'submit_example.csv')
    unlabeled_data = pd.read_csv(data_path + tiny_pre + 'unlabeled_data.csv')

# test_data = pd.read_csv(data_path + 'test_data.csv')

# 数据处理函数：剔除一些标点符号，然后jieba进行分词
def cal_func(content):

    punc = '～~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}【】'
    punc += '\xa0'
    punc += '\u2003'
    punc += '\u3000'
    punc += '\u3000'
    punc += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    punc += 'abcdefghijklmnopqrstuvwxyz'

    seg_list = jieba.cut(re.sub(r"[%s]+" %punc, "",content), cut_all=False)
    # print(seg_list)
    return " ".join(seg_list)


print('training unlabeled_data ...')
unlabeled_data['content1'] = unlabeled_data.apply(lambda v:cal_func(v['content']),axis=1)

unlabeled_data['content1'].to_csv(data_path+'label_content.txt',sep='\t',
                                index=False, header=False)

# 对数据进行无监督的训练
model = fasttext.train_unsupervised(input=data_path+"label_content.txt")

print('Done!')

# 对每个类别取30个近义词，用于对无标签数据集进行分类
label_str = ['财经','房产','家居','教育','科技','时尚','时政','游戏','娱乐','体育']
label_word = [model.get_nearest_neighbors(label_tmp,keyword_num[label_tmp])
              for label_tmp in label_str]
for label_tmp in label_str:
    print(label_tmp, ':', model.get_nearest_neighbors(label_tmp,keyword_num[label_tmp]), end='\n\n')

# 根据近义词的概率来进行类别的分类
def cal_label(content1):
    prob_list = []
    for i in range(10):
        prob_cur = 0
        len_word = len(label_word[i])
        for j in range(len_word):
            if label_word[i][j][1] in content1:
                prob_cur = label_word[i][j][0]
                break
        prob_list.append(prob_cur)
    if max(prob_list) == 0:
        return 0
    else:
        return label_str[prob_list.index(max(prob_list))]

unlabeled_data['label_content'] = unlabeled_data.apply(lambda v:cal_label(
                                                     v['content1']),axis=1)

# 对于无标签的数据进行分类的结果，30000个数据中14900个无法判断类别：
print (unlabeled_data['label_content'].value_counts())

# 增加游戏类别的数据
game_set = unlabeled_data[unlabeled_data['label_content']=='游戏']
# print(type(game_set), len(game_set))
# print(game_set.columns())
# game_set = game_set.values.tolist()
# print(game_set[0])
# print('\n' * 10)
# print(game_set)

game_set1 = game_set.loc[:, ['id', 'label_content', 'content']]
game_set1.columns = ['id', 'class_label', 'content']
dataset = dataset.append(game_set1, ignore_index=True)

# 增加娱乐类别的数据
joy_set = unlabeled_data[unlabeled_data['label_content']=='娱乐']
joy_set1 = joy_set.loc[:, ['id', 'label_content', 'content']]
joy_set1.columns = ['id', 'class_label', 'content']
dataset = dataset.append(joy_set1, ignore_index=True)

# 增加体育类别的数据
sports_set = unlabeled_data[unlabeled_data['label_content']=='体育']
sports_set1 = sports_set.loc[:, ['id', 'label_content', 'content']]
sports_set1.columns = ['id', 'class_label', 'content']
dataset = dataset.append(sports_set1, ignore_index=True)

dataset.to_csv(data_path+'labeled_data_with_10_classes.csv')

firstline = ""
lines = []
with open(data_path+'labeled_data_with_10_classes.csv', 'r') as fin:
    cnt = 0
    for line in fin:
        cnt += 1
        if cnt == 1:
            firstline = line[1:]
            continue
        pos = line.find(',')
        line = line[pos+1:]
        lines.append(line)
label_count = {}
with open(data_path+'labeled_data_with_10_classes.csv', 'w') as fout:
    fout.write(firstline)
    cnt = 0
    for line in lines:
        pos = line.find(',')
        fout.write(str(cnt) + line[pos:])
        label = line[pos+1:pos+3]
        if label in label_count:

            label_count[label] += 1
        else:
            label_count[label] = 1
        cnt += 1
print(label_count)
