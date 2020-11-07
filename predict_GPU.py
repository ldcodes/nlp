# -*- coding: utf-8 -*-
'''
@author: yaleimeng@sina.com
@license: (C) Copyright 2019
@desc: 这个代码是进行预测的。既可以根据ckpt检查点，也可以根据单个pb模型。
@DateTime: Created on 2019/7/19, at 下午 04:13 by PyCharm
'''
from train_eval import *
from tensorflow.python.estimator.model_fn import EstimatorSpec


class Bert_Class():

    def __init__(self):
        self.graph_path = os.path.join(arg_dic['pb_model_dir'], 'classification_model.pb')
        self.ckpt_tool, self.pbTool = None, None
        self.prepare()

    def classification_model_fn(self, features, mode):
        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_map = {"input_ids": input_ids, "input_mask": input_mask}
        pred_probs = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_prob:0'])

        return EstimatorSpec(mode=mode, predictions={
            'encodes': tf.argmax(pred_probs[0], axis=-1),
            'score': tf.reduce_max(pred_probs[0], axis=-1)})

    def prepare(self):
        tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])
        self.config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])

        if arg_dic['max_seq_length'] > self.config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (arg_dic['max_seq_length'], self.config.max_position_embeddings))

        # tf.gfile.MakeDirs(self.out_dir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=arg_dic['vocab_file'],
                                                    do_lower_case=arg_dic['do_lower_case'])

        self.processor = SelfProcessor()
        self.train_examples = self.processor.get_train_examples(arg_dic['data_dir'])
        global label_list
        label_list = self.processor.get_labels()

        self.run_config = tf.estimator.RunConfig(
            model_dir=arg_dic['output_dir'], save_checkpoints_steps=arg_dic['save_checkpoints_steps'],
            tf_random_seed=None, save_summary_steps=100, session_config=None, keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=10000, log_step_count_steps=100, )

    def predict_on_ckpt(self, sentence):
        if not self.ckpt_tool:
            num_train_steps = int(len(self.train_examples) / arg_dic['train_batch_size'] * arg_dic['num_train_epochs'])
            num_warmup_steps = int(num_train_steps * arg_dic['warmup_proportion'])

            model_fn = model_fn_builder(bert_config=self.config, num_labels=len(label_list),
                                        init_checkpoint=arg_dic['init_checkpoint'], learning_rate=arg_dic['learning_rate'],
                                        num_train=num_train_steps, num_warmup=num_warmup_steps)

            self.ckpt_tool = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config, )
        exam = self.processor.one_example(sentence)  # 待预测的样本列表
        feature = convert_single_example(0, exam, label_list, arg_dic['max_seq_length'], self.tokenizer)

        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=arg_dic['max_seq_length'], is_training=False,
                                            drop_remainder=False)
        result = self.ckpt_tool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        gailv = list(result)[0]["probabilities"].tolist()
        pos = gailv.index(max(gailv))  # 定位到最大概率值索引，
        return label_list, gailv

    def predict_on_pb(self, sentence):
        if not self.pbTool:
            self.pbTool = tf.estimator.Estimator(model_fn=self.classification_model_fn, config=self.run_config, )
        exam = self.processor.one_example(sentence)  # 待预测的样本列表
        print(exam)
        feature = convert_single_example(0, exam, label_list, arg_dic['max_seq_length'], self.tokenizer)
        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=arg_dic['max_seq_length'], is_training=False,
                                            drop_remainder=False)
        result = self.pbTool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        ele = list(result)[0]
        print('类别：{}，置信度：{:.3f}'.format(label_list[ele['encodes']], ele['score']))
        return label_list[ele['encodes']]


def main():
    import utils
    import time

    eps = 0.3

    data = utils.read_unlabeled_csv('./data/real/test_data_1.csv')
    predict = Bert_Class()
    ans_ckpt = []
    ans_pb = []

    # tic = time.clock()
    # for x, y in data:
    #     res = predict.predict_on_pb(y)
    #     print(x, res)
    #     ans_pb.append((x, res))
    # toc = time.clock()
    # print('pb_model time : ', toc - tic)
    # utils.write_csv_predict('./data/real/ans_pb.csv', ans_pb)

    tic = time.clock()
    for x, y in data:
        label_list, gailv = predict.predict_on_ckpt(y)
        print(x, label_list, gailv)
        pos = gailv.index(max(gailv))
        if max(gailv) < eps:
            print(x, y)
        ans_ckpt.append((x, label_list[pos]))
    toc = time.clock()
    print('ckpt_model time : ', toc - tic)
    utils.write_csv_predict('./data/real/ans_ckpt.csv', ans_ckpt)





if __name__ == "__main__":
    main()
    # import time
    #
    # testcase = ['中国高铁动车组通常运行速度是多少？', '梨树种下之后，过几年能结果子？', '青蛙在吃虫子时眼睛是怎样的状态？', '剑与远征中的迷宫遗物是一些非常不错的装备，不过其中的迷宫遗物是有着非常的多的，不少小伙伴都不知道它们的排名是什么，小编在这里总结了所有的迷宫遗物排名，不要错过了哦。遗物排名1，火剑，冰剑套装2，日之石，月之石套装3，巫毒娃娃或者涂毒，这两个是打困难哥布林很好用的东西4，稀有的金色遗物，例如死神镰刀，蛛丝手套，诅咒勾玉，神圣之锤这四个在对付困难第三层怪很好用，可以对付新手讨厌的狮子，抢先手，分裂攻击，神圣攻击都很有用。5，从普通到金色遗物都有的比较好用的有暴击率，暴击伤害，减伤率，能量石，种族克制。选择遗物时按这个排名选，再加上分配等级合理，手动操作控制血量，能量合理再过关，那么我相信每个人都会很轻松的通关困难哥布林和第三层最后，如果实在运气差拿不到排名前列的，如果每周送的三瓶眼泪充足，可以选择去打困难尝试一下，实在打不过可以喝药过，若运气差还没有药的，我建议选择普通难度，毕竟通关才是最该做的']
    # toy = Bert_Class()
    # aaa = time.clock()
    # for t in testcase:
    #     print(toy.predict_on_ckpt(t), t)
    # bbb = time.clock()
    # print('ckpt预测用时：', bbb - aaa)
    #
    # aaa = time.clock()
    # for t in testcase:
    #     toy.predict_on_pb(t)
    # bbb = time.clock()
    # print('pb模型预测用时：', bbb - aaa)
