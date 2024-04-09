# -*- coding: utf-8 -*-

"""
zipfLaw.py
作业1-1：对中文语料库进行词频统计，验证Zipf's Law
"""

import jieba
import os
import matplotlib.pyplot as plt

from util import getFilePathList

# 声明数据文件路径
dataPath = './data/'
listFilePath = os.path.join(dataPath, 'inf.txt')
totalFilePath = os.path.join(dataPath, 'total.txt')


# 读取txt文件，将词频计数写入wordCntMap
def wordCount(txtFilePath, wordCntMap):
    # 检查单词是否均为中文
    def wordIsChinese(wordToCheck):
        for ch in wordToCheck:
            if '\u4e00' <= ch <= '\u9fff':
                continue
            else:
                return False
        return True

    if not os.path.exists(txtFilePath) or wordCntMap is None:
        raise Exception('File not exists or cntMap is None')
    with open(txtFilePath, 'r', encoding='gb18030') as txtFile:
        rawTxt = txtFile.read()
        # 去掉无意义词汇
        rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
        rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
        txt = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
        wordList = jieba.lcut(txt)  # 精确模式分词
        for word in wordList:
            if wordIsChinese(word):  # 过滤掉所有非中文词（包括标点）
                wordCntMap[word] = wordCntMap.get(word, 0) + 1


# 保存词频统计文件
def saveCntList(cntList, saveFilePath):
    with open(saveFilePath, 'w', encoding='gbk') as saveFile:
        for word, cnt in cntList:
            saveFile.write(word + '\t' + str(cnt) + '\n')


# 对map进行排序，得到排名-词频图（对数坐标便于观察现象）
def showZipf(wordCntMap):
    # 对词频统计map进行排序
    wordCntList = list(wordCntMap.items())
    wordCntList.sort(key=lambda x: x[1], reverse=True)
    # 保存词频统计map到文件
    saveCntList(wordCntList, 'frequency.txt')
    # 进一步得到排名-词频列表
    cntList = list(map(lambda x: x[1], wordCntList))
    rankList = [i for i in range(len(cntList))]
    # 显示并保存统计图
    plt.loglog(rankList, cntList, linewidth='1.5', color='green', label='Result')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf-Law', fontsize=18)
    plt.legend()
    plt.savefig('./zipfLaw.jpg')
    plt.show()


if __name__ == '__main__':
    # 读取inf.txt，得到文件列表
    fileList = getFilePathList(listFilePath)
    # fileList = ['D:\\Desktop\\Learning\\NLP\\NLPhw1\\code\\data\\三十三剑客图.txt']
    # 对所有文件一并进行词频统计，写入map
    cntMap = {}
    for filePath in fileList:
        wordCount(filePath, cntMap)
    # 展示排名-词频统计图，验证zipf‘s law
    showZipf(cntMap)
