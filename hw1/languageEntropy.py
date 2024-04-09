# -*- coding: utf-8 -*-

"""
languageEntropy.py
作业1-2: 对中文语料库进行统计，按字和词计算中文的平均信息熵
"""
import math

import jieba
import os
import csv
from util import getFilePathList

# 声明数据文件路径
dataPath = './data/'
listFilePath = os.path.join(dataPath, 'inf.txt')
stopFilePath = os.path.join(dataPath, 'cn_stopwords.txt')
totFilePath = os.path.join(dataPath, 'total.txt')
resFilePath = './languageEntropy_result.csv'


# 用某个具体语料库计算中文信息熵的类
class LanguageEntropy(object):
    def __init__(self, name, stopWordFilePath):
        self.name = name  # 文件名
        self.data = None  # 原始char列表
        self.chars = []  # 单字列表
        self.words = []  # 单词列表
        # 读取stop words
        with open(stopWordFilePath, 'r', encoding='utf-8') as stopWordFile:
            self.stopWords = []
            for line in stopWordFile.readlines():
                self.stopWords.append(line.strip())  # 去掉回车

    # 读取文件，读入data，并进行分词
    def readFile(self, filePath):
        txt = open(filePath, 'r', encoding='gb18030')
        # 按字存储
        rawTxt = txt.read()
        # 去掉无意义词汇
        rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
        rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
        self.data = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
        for ch in self.data:
            if ch not in self.stopWords and (not ch.isspace()):
                self.chars.append(ch)
        # 分词之后，按词存储
        words = jieba.lcut(self.data)
        for word in words:
            if word not in self.stopWords:
                self.words.append(word)

    # 用unigram模型进行词频统计，mode只能为"char"或“word”，表示对字或词进行词频统计
    # 返回一个cntMap，内容为 string->int ,即char或word的计数map
    def unigramFrequencyCount(self, mode):
        cntMap = {}
        unitSeq = None
        if mode == "char":
            unitSeq = self.chars
        elif mode == "word":
            unitSeq = self.words
        # 进行必要的错误处理
        if unitSeq is None:
            raise Exception("Wrong mode for frequency Count!")
        # 按照mode进行计数
        for unit in unitSeq:
            cntMap[unit] = cntMap.get(unit, 0) + 1
        return cntMap

    # 用bigram模型进行词频统计，mode只能为"char"或“word”，表示对字或词进行词频统计
    # 返回一个cntMap，内容为 (string,string)->int ,即2个char或word组成元组的计数map
    def bigramFrequencyCount(self, mode):
        cntMap = {}
        unitSeq = None
        if mode == "char":
            unitSeq = self.chars
        elif mode == "word":
            unitSeq = self.words
        # 进行必要的错误处理
        # bigram要求序列必须大于等于2个符号
        if unitSeq is None or len(unitSeq) <= 1:
            raise Exception("Error in bigram frequency Count!")
        # 按照mode进行计数
        for i in range(len(unitSeq) - 1):
            nowTup = (unitSeq[i], unitSeq[i + 1])
            cntMap[nowTup] = cntMap.get(nowTup, 0) + 1
        return cntMap

    # 用trigram模型进行词频统计，mode只能为"char"或“word”，表示对字或词进行词频统计
    # 返回一个cntMap，内容为 (string, string, string)->int ,即3个char或word组成元组的计数map
    def trigramFrequencyCount(self, mode):
        cntMap = {}
        unitSeq = None
        if mode == "char":
            unitSeq = self.chars
        elif mode == "word":
            unitSeq = self.words
        # 进行必要的错误处理
        # trigram要求序列必须大于等于3个符号
        if unitSeq is None or len(unitSeq) <= 2:
            raise Exception("Error in trigram frequency Count!")
        # 按照mode进行计数
        for i in range(len(unitSeq) - 2):
            nowTup = (unitSeq[i], unitSeq[i + 1], unitSeq[i + 2])
            cntMap[nowTup] = cntMap.get(nowTup, 0) + 1
        return cntMap

    # 按照unigram模型计算平均信息熵
    def unigramEntropy(self, mode):
        singleUnitFreqMap = self.unigramFrequencyCount(mode)
        length = len(self.words if mode == "word" else self.chars)
        entropy = 0
        for item in singleUnitFreqMap.items():
            freqPr = item[1] / length
            entropy += -1.0 * freqPr * math.log(freqPr, 2)
        print("For {}, Unigram Entropy in mode {} is: {}".format(self.name, mode, entropy))
        return entropy

    # 按照bigram模型计算平均信息熵
    def bigramEntropy(self, mode):
        singleUnitFreqMap = self.unigramFrequencyCount(mode)
        bigramFreqMap = self.bigramFrequencyCount(mode)
        length = len(self.words if mode == "word" else self.chars) - 1  # 2字/词序列共有len-1个
        entropy = 0
        for item in bigramFreqMap.items():
            PrUnion = item[1] / length  # 联合概率，即P(x, y)
            PrCond = PrUnion / (singleUnitFreqMap[item[0][0]] / (length + 1))  # 条件概率，
            # 即P(x|y)=P(x, y)/P(y)，注意x是后一个元素，y是前一个
            entropy += -1.0 * PrUnion * math.log(PrCond, 2)
        print("For {}, Bigram Entropy in mode {} is: {}".format(self.name, mode, entropy))
        return entropy

    # 按照trigram模型计算平均信息熵
    def trigramEntropy(self, mode):
        doubleUnitFreqMap = self.bigramFrequencyCount(mode)
        trigramFreqMap = self.trigramFrequencyCount(mode)
        length = len(self.words if mode == "word" else self.chars) - 2  # 2字/词序列共有len-2个
        entropy = 0
        for item in trigramFreqMap.items():
            PrUnion = item[1] / length  # 联合概率，即P(x, y, z)
            PrCond = PrUnion / (doubleUnitFreqMap[(item[0][0], item[0][1])] / (length + 1))  # 条件概率，
            # 即P(x|y, z)=P(x, y, z)/P(y, z)，注意x是后一个元素，y, z是前面两个
            entropy += -1.0 * PrUnion * math.log(PrCond, 2)
        print("For {}, Trigram Entropy in mode {} is: {}".format(self.name, mode, entropy))
        return entropy


if __name__ == '__main__':
    # 读取inf.txt，得到文件列表
    filePathList = list(getFilePathList(listFilePath))
    filePathList.append(totFilePath)  # 加上所有语料库组成的文件
    # 便于结果写入csv文件，提前定义好csv头
    resCsvHeadList = ['FileName', 'Char-Unigram', 'Char-Bigram', 'Char-Trigram',
                                  'Word-Unigram', 'Word-Bigram', 'Word-Trigram']
    # 用多个map组成的list记录最终结果，一个map对应一个语料库的结果，map的键参考HeadList
    lanEntropyResMaps = []
    for filePath in filePathList:
        # 创建对象并读取文件
        entropyCalc = LanguageEntropy(os.path.basename(filePath), stopFilePath)
        entropyCalc.readFile(filePath)
        # 创建结果记录map, 直接分别计算三个模型下的信息熵结果
        resMap = {
            'FileName': entropyCalc.name,
            'Char-Unigram': entropyCalc.unigramEntropy('char'),
            'Char-Bigram': entropyCalc.bigramEntropy('char'),
            'Char-Trigram': entropyCalc.trigramEntropy('char'),

            'Word-Unigram': entropyCalc.unigramEntropy('word'),
            'Word-Bigram': entropyCalc.bigramEntropy('word'),
            'Word-Trigram': entropyCalc.trigramEntropy('word')
        }
        # 将结果记录入list
        lanEntropyResMaps.append(resMap)

    # 计算结束，将结果写入csv文件
    with open(resFilePath, 'w', encoding='gbk', newline='') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=resCsvHeadList)
        writer.writeheader()
        writer.writerows(lanEntropyResMaps)


