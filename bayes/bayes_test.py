# -*- coding:utf-8 -*-

import os
from sklearn.externals import joblib
from bayes.bayes_train import load_dataset, get_dataset, split_train_and_test_set, multinamialNB_save_path, bernousNB_save_path

'''
    从文件读取模型并进行分类
'''

test_data = get_dataset()    # 从文本中测试
train_set_tmp, train_label_tmp, test_set, test_label = split_train_and_test_set(test_data, 0.0)

# test_data = load_dataset()     # 从测试集中测试
# train_set_tmp, train_label_tmp, test_set, test_label = split_train_and_test_set(test_data, 0.7)

'''
    分离出来
'''
def divideTestSet(test_set):
    for tset in test_set:
        print(tset)
        # pass

'''
    获取最新的模型
'''
def get_newest_model(model_path):
    if os.path.exists(model_path):
        # 按文件最后修改时间排序，reverse=True表示降序排序
        filelist = sorted(os.listdir(model_path), key=lambda x: os.path.getctime(os.path.join(model_path, x)), reverse=True)
        return os.path.join(model_path, filelist[0])



'''
    测试贝叶斯分类器
'''
def test_bayes(model_file):
    clf = joblib.load(model_file)
    predict = clf.predict(test_set)

    count = 0
    for left, right, tset in zip(predict, test_label, test_set):
        if left == "坐车":
            left = "坐高铁"
        if right == "坐车":
            right = "坐高铁"
        # print(left, "--", right, "--", tset)
        # if left == right:
        #     count += 1
        if left != right:
            print(left, "--", right, "--", tset)
    print(model_file, "准确率：", count / len(test_label))



def main():
    # test_bayes(get_newest_model(multinamialNB_save_path))
    test_bayes(get_newest_model(bernousNB_save_path))
    # print(get_newest_model(multinamialNB_save_path))
    # print(get_newest_model(bernousNB_save_path))
    # divideTestSet(test_set)

if __name__ == '__main__':
    main()