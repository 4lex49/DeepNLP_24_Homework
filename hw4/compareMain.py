# -*- coding: utf-8 -*-
import csv
import random
import Seq2SeqModel
from utils import *
import TransformerModel

# 运行参数：
txtFilePath = os.path.join('.\\data\\corpus', '雪山飞狐.txt')  # 测试文本
stopWordsPath = '.\\data\\stopwords'  # 停用词目录
dictPath = '.\\models\\seq2seq.dict'  # 词典保存路径
extraTokens = ['<bos>', '<eos>', '<pad>']  # 添加的token
trainRate = 0.8  # 训练集占比
maxTokens = 512  # 最大token长度
testLength = 5  # 测试初始长度
resultsPath = '.\\results\\'

if __name__ == '__main__':
    # 读取文件
    sentences = readFile(txtFilePath, stopWordsPath)
    # 插入tokens，padding到相同长度
    sentences = preProcessing(sentences, maxTokens)
    # 创建词典
    dictionary = None
    if os.path.exists(dictPath):
        dictionary = corpora.Dictionary.load(dictPath)
    else:
        dictionary = generateDictionary(sentences, extraTokens, dictPath)

    # 划分训练集和测试集
    random.seed(20240616)
    random.shuffle(sentences)
    trainNum = int(trainRate * len(sentences))
    trainDataset = TextDataset(sentences[:trainNum], dictionary)
    testDataset = TextDataset(sentences[trainNum:], dictionary)
    # 首先训练并保存Seq2Seq
    if not os.path.exists(Seq2SeqModel.Seq2SeqArgs.savePath):
        Seq2SeqModel.train(trainDataset, dictionary, testLength, maxTokens)
    resultTups = Seq2SeqModel.test(testDataset, dictionary, testLength, maxTokens)
    with open(os.path.join(resultsPath, 'Seq2Seq.csv'), mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Feed Text', 'Generate Text', 'GroundTruthText'])
        for inputText, outputText, gtText in resultTups:
            writer.writerow([''.join(inputText), ''.join(outputText), ''.join(gtText)])

    if not os.path.exists(TransformerModel.TransformerArgs.savePath):
        TransformerModel.train(trainDataset, dictionary, testLength, maxTokens)
    resultTups = TransformerModel.test(testDataset, dictionary, testLength, maxTokens)
    with open(os.path.join(resultsPath, 'Transformer.csv'), mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Feed Text', 'Generate Text', 'GroundTruthText'])
        for inputText, outputText, gtText in resultTups:
            writer.writerow([''.join(inputText), ''.join(outputText), ''.join(gtText)])

