import jieba

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('stopWord/hit_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords


def del_stop_word(sentence):
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
    #for word in sentence:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                #outstr += " "
    return outstr


def  test():
    # 给出文档路径
    filename = "test.txt"
    outfilename = "out.txt"
    inputs = open(filename, 'r', encoding='UTF-8')
    outputs = open(outfilename, 'w', encoding='UTF-8')

    # 将输出结果写入ou.txt中
    for line in inputs:
        line_seg = del_stop_word(line)
        outputs.write(line_seg + '\n')
        print("-------------------正在分词和去停用词-----------")
    outputs.close()
    inputs.close()
    print("删除停用词和分词成功！！！")
