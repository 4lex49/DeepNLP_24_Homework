# -*- coding:utf-8 -*-

import os
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 声明数据文件路径，依次为数据根目录、语料库目录与停止词目录
dataRootPath = './data/'
corpusFilePath = os.path.join(dataRootPath, 'corpus', 'total.txt')
# corpusFilePath = os.path.join(dataRootPath, 'corpus', 'small.txt')
stopWordsFilePath = os.path.join(dataRootPath, 'stopwords')

# 结果文件目录
resultRootPath = './res/'

# 测试词向量对的距离
testDistWords = [
    ('杨过', '小龙女'),
    ('郭靖', '黄蓉'),
    ('韦小宝', '令狐冲')
]

# 测试词向量关系：a-b+c=?
relationWords = [
    ('杨过', '小龙女', '黄蓉'),
    ('郭靖', '洪七公', '风清扬')
]

# 聚类数量，展示聚类数量，聚类的核心名
clusterNum = 10
showClusters = 6
focusNames = '阿青 石破天 杨过 虚竹 郭靖 张无忌 乔峰 令狐冲 段誉 狄云'.split(' ')

# 测试段落：
testParagraphs = [
    '''所传关外大力擒拿手法，当胸抓来。郭靖左腿微屈，右臂内弯，右掌划了个圆圈，呼的一声，向外推去，正是初学乍练的一招“亢龙有悔”。那大弟子听到掌风劲
    锐，反抓回臂，要挡他这一掌，喀喇一声，手臂已断，身子直飞出六七尺之外，晕了过去。郭靖万料不到这一招竟有偌大威力，一呆之下，拔脚又奔。''',
    '''
    郭靖练到后来，意与神会，发劲收势，渐渐能运用自如，丹田中听一口气，猛力一掌，立即收劲，那松树竟是纹丝不动。郭靖大喜，第二掌照式发招，但力在掌缘，
    只听得格格数声，那棵小松树被他击得弯折了下来。
    '''
]


class WordVectorizer:
    def __init__(self):
        self.sentences = None
        self.word2Vec = None

    # 从语料库中读取并分词（注意要保持段落，用于word2vec训练）
    def readWords(self, stopWordsPath, corpusPath):
        # 读取stop words
        stopWords = []
        for file in os.listdir(stopWordsPath):
            with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as stopWordFile:
                for line in stopWordFile.readlines():
                    stopWords.append(line.strip())  # 去掉回车

        # 读取语料库并分词
        with open(corpusPath, 'r', encoding='gb18030') as corpusFile:
            # 读取内容并删除冗余
            rawTxt = corpusFile.read()
            rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
            rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
            txtData = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
            sentenceData = txtData.split('\n')
            self.sentences = []
            for sen in sentenceData:
                rawSenWords = list(jieba.cut(sen))
                senWords = [word for word in rawSenWords if not word.isspace() and word not in stopWords]
                if len(senWords) > 0:
                    self.sentences.append(senWords)

    # 词向量化，即训练word2vec模型并保存
    def vectorize(self, create=False):
        binFilePath = os.path.join(resultRootPath, 'word2vec.bin')
        if create or not os.path.exists(binFilePath):
            self.word2Vec = Word2Vec(self.sentences, vector_size=100, window=5, workers=3, min_count=5, epochs=50)
            self.word2Vec.save(binFilePath)
        else:
            self.word2Vec = Word2Vec.load(binFilePath)

    # 测试word1与word2向量的距离
    def testWordDist(self, word1, word2):
        if word1 in self.word2Vec.wv.index_to_key and word2 in self.word2Vec.wv.index_to_key:
            similarity = self.word2Vec.wv.similarity(word1, word2)
            print(f'Similarity between {word1} and {word2} is {similarity}')
        else:
            print('Word error!')

    # 测试wordMain-wordMinus+wordPlus=?
    def testRelation(self, wordMain, wordMinus, wordPlus):
        if (wordMain in self.word2Vec.wv.index_to_key and wordMinus in self.word2Vec.wv.index_to_key
                and wordPlus in self.word2Vec.wv.index_to_key):
            simWord = self.word2Vec.wv.most_similar(positive=[wordMain, wordPlus], negative=[wordMinus])[0]
            print(f'{wordMain} - {wordMinus} + {wordPlus} is {simWord}')
        else:
            print('Word error!')

    # 对mainWords提取topk，对提取结果进行聚类
    def clusterWords(self, clusterN, mainWords):
        # 提取TOPk并聚类
        mainContainsWords = [word for word in mainWords if word in self.word2Vec.wv.index_to_key]
        wordWithVecs = []
        for w in mainContainsWords:
            wordWithVecs.extend(self.word2Vec.wv.similar_by_word(w, topn=10))
        clusterWords = [res[0] for res in wordWithVecs]
        vectorOfWords = np.array([self.word2Vec.wv[word] for word in clusterWords])
        kMeansModel = KMeans(n_clusters=clusterN)
        kMeansModel.fit(vectorOfWords)

        # TSNE 降维
        tsne = TSNE(n_components=2, random_state=0)
        tsneResult = tsne.fit_transform(vectorOfWords)
        vector2dWithLabels = np.vstack((tsneResult[:, 0], tsneResult[:, 1], kMeansModel.labels_)).transpose()

        # 接下来进行绘图
        plt.title('聚类分析可视化结果')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 不同类别用不同颜色和样式绘图
        colors = ['b', 'c', 'y', 'r', 'g', 'm', 'k']
        markers = ['.', 'o', 's', 'X', 'P', '*', 'p', 'D', '^']
        for i in range(len(vector2dWithLabels)):
            data = vector2dWithLabels[i]
            if data[2] < showClusters:
                plt.plot(data[0], data[1], color=colors[int(data[2]) % len(colors)],
                         marker=markers[int(data[2]) % len(markers)])
                plt.annotate(clusterWords[i], xy=(data[0], data[1]), xytext=(data[0] + 0.05, data[1] + 0.05),
                             textcoords='offset points')
        # 不显示坐标刻度
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(resultRootPath, 'clusterName.png'))
        plt.show()

    # 计算段落距离
    def calculateParaDist(self, para1, para2):
        words1 = jieba.cut(para1)
        words2 = jieba.cut(para2)
        vec1 = (np.array([self.word2Vec.wv[word] for word in words1 if word in self.word2Vec.wv.index_to_key]).
                mean(axis=0))
        vec2 = (np.array([self.word2Vec.wv[word] for word in words2 if word in self.word2Vec.wv.index_to_key]).
                mean(axis=0))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 测试段落距离
    def testParagraphDist(self, paragraphs):
        print(f"Dist between paragraph in the same friction is {self.calculateParaDist(paragraphs[0], paragraphs[1])}")


if __name__ == '__main__':
    wordVec = WordVectorizer()  # 初始化
    wordVec.readWords(stopWordsFilePath, corpusFilePath)  # 读取并分词
    wordVec.vectorize()  # 训练word2Vec

    for testWordTup in testDistWords:   # 测试单词距离
        wordVec.testWordDist(testWordTup[0], testWordTup[1])

    for testRelationTup in relationWords:  # 测试单词关系
        wordVec.testRelation(testRelationTup[0], testRelationTup[1], testRelationTup[2])

    wordVec.clusterWords(clusterNum, focusNames)  # 聚类分析可视化

    wordVec.testParagraphDist(testParagraphs)  # 测试段落距离
