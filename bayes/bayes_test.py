# -*- coding:utf-8 -*-

import os
from sklearn.externals import joblib
from bayes.bayes_train import get_words, isChat

'''
    测试分类器
'''

def test_bayes(model_file):
    clf = joblib.load(model_file)
    while True:
        message = input("请说话：")
        word_list = []
        new_sentences, shinei_area = get_words(message)
        word_list.append(new_sentences)
        # print("word_list:", word_list)
        if isChat(new_sentences) is False:  # 如果不是咨询类
            predict = clf.predict(word_list)
            for left in predict:
                if left == "坐车":
                    left = "坐高铁"
                print(left, "-->", word_list, "-->", message)

def main():
    # newest_model = "./model/bernousNB/bernousNB_1576632950_9512195121951219_0_0.m"
    newest_model = "./model/bernousNB/bernousNB_1593583220_9872746553552492_0_None.m"
    test_bayes(newest_model)

if __name__ == '__main__':
    main()