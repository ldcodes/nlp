# coding: UTF-8
import numpy as np
import torch
import time
from utils import build_iterator, get_time_dif
from importlib import import_module
from tqdm import tqdm
from generate_data import cut_para


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
min_length = 64
label2class = {
    "财经" : "高风险",
    "时政" : "高风险",
    "房产" : "中风险",
    "科技" : "中风险",
    "教育" : "低风险",
    "时尚" : "低风险",
    "游戏" : "低风险",
    "家居" : "可公开",
    "体育" : "可公开",
    "娱乐" : "可公开",
}
label2num = {
    "财经" : 0,
    "时政" : 1,
    "房产" : 2,
    "科技" : 3,
    "教育" : 4,
    "时尚" : 5,
    "游戏" : 6,
    "家居" : 7,
    "体育" : 8,
    "娱乐" : 9,
}
num2label = {
    0 : "财经",
    1 : "时政",
    2 : "房产",
    3 : "科技",
    4 : "教育",
    5 : "时尚",
    6 : "游戏",
    7 : "家居",
    8 : "体育",
    9 : "娱乐"
}


class Predict_Baseline():
    """
    第一种预测方法
    不对预测的句子做任何处理
    就直接尾部截断预测

    优点: 快? 因为直接截断，数据量小了很多
    问题: 无法看到篇章的全部信息

    可能会继续做的方法（咕咕咕）:
    1. 把预测的序列变成多个，然后综合每个预测结果做出最终预测
    2. 对篇章关键词抽取 / ... 等可能有用的方法， 然后建图，做谱聚类 (好像很难写...)
    """
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        pass

    def load_dataset(self, path, pad_size):
        contents = []
        config = self.config
        with open(path, 'r', encoding='utf-8') as fin:
            cnt = 0
            for line in tqdm(fin):
                lin = line.strip()
                if not lin:
                    continue
                cnt += 1
                if cnt == 1:
                    continue
                # print(cnt, lin + '\n\n\n')
                pos = lin.find(',')
                id = lin[:pos]
                content = lin[pos + 1:]
                # print('?????????? : ', id, content + '\n\n')

                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(id), seq_len, mask))
                # print('\nlen(contents) : ', str(len(contents))+'\n')
        return contents
    def build_dataset(self, path):
        # 加载数据集
        # [(tokens, int(id), seq_len, mask)]
        config = self.config

        print('\nloading predict set ...')
        predict_data = self.load_dataset(path, config.pad_size)
        print('Done!')
        self.predict_iter = build_iterator(predict_data, config)

    def evaluate(self, model):
        config = self.config
        predict_iter = self.predict_iter
        model.eval()

        predict_all = np.array([], dtype=int)

        with torch.no_grad():
            for texts, ids in tqdm(predict_iter):
                outputs = model(texts)
                # print('outputs : ', outputs)
                ids = ids.data.cpu().numpy()
                predict_label = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predict_label)
        return predict_all

    def predict(self, model):
        config = self.config
        predict_iter = self.predict_iter

        model.load_state_dict(torch.load(config.save_path))
        model.eval()

        start_time = time.time()
        print('prediction ...')
        predict_labels = self.evaluate(model)
        time_dif = get_time_dif(start_time)
        print('Done !')
        print('prediction usage:',time_dif)
        return predict_labels

    def write_csv(self, labels, path):
        with open(path, 'w') as fout:
            cnt = 0

            fout.write('id,class_label,rank_label'+'\n')
            for label in labels:
                fout.write(str(cnt) + ',' + num2label[label] + ',' + label2class[num2label[label]] + '\n')
                cnt += 1

class Predict_Cut_Paras():
    """
    方法二 篇章切割，综合结果预测
    type = 1 表示label投票
    type = 2 表示得分softmax之和
    type = 3 表示得分之和
    others TBD -> ERROR
    """
    def __init__(self, dataset, config, type=1):
        self.dataset = dataset
        self.config = config
        self.type = type
        if type == 1 or type == 2 or type == 3:
            pass
        else:
            raise ValueError

    def load_dataset(self, path, pad_size):
        contents = []
        config = self.config

        # 篇章切割
        print('cut paras ...')
        start_time = time.time()
        with open(path, 'r', encoding='utf-8') as fin:
            cnt = 0
            data = []
            for line in tqdm(fin):
                lin = line.strip()
                if not line:
                    continue
                cnt += 1
                if cnt == 1:
                    continue
                pos = lin.find(',')
                id = lin[:pos]
                content = lin[pos + 1:]
                paras = cut_para(content)
                for para in paras:
                    #if len(para) < min_length:
                    #    continue
                    data.append((int(id), para))
            print('Done!')
            print('\nparas:',len(data))
            print('Time usage:',get_time_dif(start_time))
            print('\n Getting tokens ...')
            for id, content in tqdm(data):

                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(id), seq_len, mask))
                # print('\nlen(contents) : ', str(len(contents))+'\n')
        return contents
    def build_dataset(self, path):
        # 加载数据集
        # [(tokens, int(id), seq_len, mask)]
        config = self.config

        print('\nloading predict set ...')
        predict_data = self.load_dataset(path, config.pad_size)
        print('Done!')
        self.predict_iter = build_iterator(predict_data, config)

    def evaluate(self, model):
        config = self.config
        predict_iter = self.predict_iter
        model.eval()

        predict_all = np.array([], dtype=int)
        id_all = np.array([], dtype=int)
        score_all = np.array([[]], dtype=int)

        with torch.no_grad():
            for texts, ids in tqdm(predict_iter):
                outputs = model(texts)
                # print('outputs : ', outputs)
                ids = ids.data.cpu().numpy()
                predict_label = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predict_label)
                id_all = np.append(id_all, ids)
                score_all = np.append(score_all, outputs.data.cpu().numpy())
        if self.type == 1:
            return predict_all, id_all
        elif self.type == 2:
            return score_all, id_all
        elif self.type == 3:
            return score_all, id_all

    def predict(self, model):
        config = self.config
        model.load_state_dict(torch.load(config.save_path))
        model.eval()

        start_time = time.time()
        print('prediction ...')
        predict_labels, ids = self.evaluate(model)
        time_dif = get_time_dif(start_time)
        print('Done !')
        print('prediction usage:',time_dif)
        return predict_labels, ids

    def softmax(self, score):
        score = np.array(score)
        score = np.exp(score)
        S = np.sum(score)
        score = score / S
        return score

    def write_csv(self, ids, labels, path):
        print('ids:',len(ids))
        print('labels',len(labels))
        print(labels)
        # assert 10 * len(ids) == len(labels)
        cnt = 0
        with open(path, 'w') as fout:
            fout.write('id,class_label,rank_label'+'\n')
            i = 0
            while i < len(ids):
                score = [0] * 10
                if self.type == 1:
                    score = np.array(score)
                else:
                    score = np.array(score, dtype=float)
                if self.type == 1:
                    score[labels[i]] += 1
                    while i+1 < len(ids) and ids[i+1] == ids[i]:
                        i += 1
                        score[labels[i]] += 1
                elif self.type == 2:
                    tmp = labels[10*i:10*(i+1)]
                    score += tmp
                    while i+1 < len(ids) and ids[i+1] == ids[i]:
                        i += 1
                        tmp = labels[10*i:10*(i+1)]
                        tmp = self.softmax(tmp)
                        score += tmp
                elif self.type == 3:
                    tmp = np.array(labels[10*i:10*(i+1)])
                    score += tmp
                    while i+1 < len(ids) and ids[i+1] == ids[i]:
                        i += 1
                        tmp = np.array(labels[10*i:10*(i+1)])
                        score += tmp
                score = list(score)
                label = score.index(max(score))
                fout.write(str(cnt) + ',' + num2label[label] + ',' + label2class[num2label[label]] + '\n')
                cnt += 1
                i += 1
        print('cnt:',cnt)


def baseline_method(x, config, dataset):
    # 准备预测数据集
    start_time = time.time()
    predict_model = Predict_Baseline(dataset=dataset, config=config)
    predict_model.build_dataset(path=dataset + '/data/test_data.csv')
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)
    # 预测并写入文件
    model = x.Model(config).to(config.device)
    predict_labels = predict_model.predict(model=model)
    predict_model.write_csv(labels=predict_labels, path=dataset + '/data/result_baseline.csv')

def cut_paras_method(x, config, dataset, type):
    # 准备预测数据集
    start_time = time.time()
    predict_model = Predict_Cut_Paras(dataset=dataset, config=config, type=type)
    predict_model.build_dataset(path=dataset + '/data/test_data.csv')
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    # 预测并写入文件
    model = x.Model(config).to(config.device)
    predict_labels, ids = predict_model.predict(model=model)
    predict_model.write_csv(ids,predict_labels,path=dataset + '/data/result_cut_paras' + str(type) + '.csv')

def main():
    dataset = 'real'  # 数据集
    model_name = 'bert' # 模型名称

    # 加载模块
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样



    # # predict_baseline 方法
    # baseline_method(x, config, dataset)

    # predict_cut_paras 方法
    cut_paras_method(x, config, dataset, type=3)



if __name__ == '__main__':
    main()
