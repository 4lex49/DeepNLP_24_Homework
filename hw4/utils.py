# -*- coding: utf-8 -*-
import jieba
import os

import torch
from gensim import corpora
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


# 读取文件，清洗数据，获得词段落序列
def readFile(txtFilePath, stopWordsPath):
    # 读取stopWords
    stopWords = []
    for file in os.listdir(stopWordsPath):
        with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as stopWordFile:
            for line in stopWordFile.readlines():
                stopWords.append(line.strip())  # 去掉回车
    # 读取文件
    txtFilePath = txtFilePath
    corpusFile = open(txtFilePath, 'r', encoding='gb18030')
    if corpusFile.closed:
        raise IOError(txtFilePath + 'File Open error!')
    # 读取内容并删除冗余，转换成段落序列
    rawTxt = corpusFile.read()
    rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
    rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
    txtData = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
    sentenceData = txtData.split('\n')
    # 需要段落序列
    sentences = []
    for sen in sentenceData:
        rawSenWords = list(jieba.cut(sen))
        senWords = [word for word in rawSenWords if not word.isspace() and word not in stopWords]
        if len(senWords) > 0:
            sentences.append(senWords)
    return sentences


def preProcessing(sentences, tokenSizeLimit):
    allInOneSentences = []
    for sen in sentences:
        allInOneSentences.append('<bos>')
        allInOneSentences.extend(sen)
        allInOneSentences.append('<eos>')
    postSentences = []
    i = 0
    while(i + tokenSizeLimit < len(allInOneSentences)):
        postSentences.append(allInOneSentences[i:i+tokenSizeLimit])
        i += tokenSizeLimit
    postSentences.append(allInOneSentences[i:len(allInOneSentences)]
                         + ['<pad>'] * (i+tokenSizeLimit-len(allInOneSentences)))
    return postSentences


def generateDictionary(sentences, otherTokens, saveDictPath):
    dictionary = corpora.Dictionary([otherTokens])
    dictionary.add_documents(sentences)
    dictionary.save(saveDictPath)
    return dictionary


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, sentencesOfWords, dictionary):
        self.dictionary = dictionary
        self.sentencesOfWords = sentencesOfWords

    def __len__(self):
        return len(self.sentencesOfWords)

    def __getitem__(self, idx):
        sen = self.sentencesOfWords[idx]
        nowVec = [self.dictionary.token2id[word] for word in sen]
        return torch.LongTensor(nowVec)


def drawLoss(losses, epochNum, modelName):
    plt.cla()
    x1 = [(i / len(losses) * epochNum) for i in range(len(losses))]
    y1 = losses
    plt.title(modelName + ' Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("./imgs/" + modelName + ".png")
    plt.show()
