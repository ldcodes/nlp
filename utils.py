import csv
import os
import random

keys = {
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

def read_labeled_csv(input_filename):
    with open(input_filename, 'r') as fin:
        data = []
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt == 1:
                continue
            pos = line.find(',')
            id = line[0:pos]
            print(id)
            id = int(id)
            class_name = line[pos+1:pos+3]
            text = line[pos+4:]
            data.append((id, class_name, text))
    return data

def read_unlabeled_csv(input_filename):
    with open(input_filename, 'r') as fin:
        data = []
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt == 1:
                continue
            pos = line.find(',')
            id = line[0:pos]
            print(id)
            id = int(id)
            text = line[pos+1:]
            data.append((id, text))
    return data

def write_txt(output_filename, data):
    with open(output_filename, 'w') as fout:
        for z, x, y in data:
            fout.write(x + '\t' + y)

def write_csv_predict(output_filename, data):
    data.sort(key=lambda x:(x[0]))
    with open(output_filename, 'w') as fout:
        fout.write('id,class_label,rank_label' + '\n')
        for x, y in data:
            print(x, y)
            fout.write(str(x) + ',' + y + ',' + keys[y] + '\n')

def write_csv_original(output_filename, data):
    data.sort(key=lambda x:(x[0]))
    with open(output_filename, 'w') as fout:
        fout.write('id,content' + '\n')
        for x, y in data:
            print(x, y)
            fout.write(str(x) + ',' + y)



def main():
    dataset = './data/real'

    # 用labeled_data 生成 train_data和validation_data
    data = read_labeled_csv(dataset+'/labeled_data.csv')
    random.shuffle(data)
    train_percent = 0.7
    train_data = data[:int(train_percent*len(data))]
    validation_data = data[int(train_percent*len(data)):]
    write_txt(dataset+'/train.txt', train_data)
    write_txt(dataset+'/val.txt', validation_data)
    random.shuffle(data)
    write_txt(dataset+'/test.txt', data)

    test_data = read_unlabeled_csv(dataset+'/test_data.csv')
    random.shuffle(test_data)
    test_data_1 = test_data[:100]
    write_csv_original(dataset+'/test_data_1.csv', test_data_1)




if __name__ == '__main__':
    main()