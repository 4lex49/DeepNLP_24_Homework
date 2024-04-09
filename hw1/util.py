# -*- coding: utf-8 -*-

"""
util.py
辅助工具代码：将多个语料库文件合并为一个
"""

import os

# 声明数据文件路径
dataPath = './data/'
listFilePath = os.path.join(dataPath, 'inf.txt')
totalFilePath = os.path.join(dataPath, 'total.txt')


# 从inf.txt中获得各源文件名，组成list
def getFilePathList(listPath):
    if not os.path.exists(listPath):
        raise Exception('File not exists')
    with open(listPath, 'r', encoding='gbk') as listFile:
        cxt = listFile.readline()
        fileNameList = cxt.split(',')  # inf.txt用','隔开文件
        filePathList = map(lambda fileName: os.path.join(dataPath, fileName + '.txt'), fileNameList)
    return filePathList


if __name__ == '__main__':
    # 读取inf.txt，得到文件列表
    fileList = getFilePathList(listFilePath)
    # 先打开写入文件
    with open(totalFilePath, 'w', encoding='gb18030') as totalFile:
        # 遍历语料库
        for txtFilePath in fileList:
            with open(txtFilePath, 'r', encoding='gb18030') as txtFile:
                txt = txtFile.read()
                # 写入综合文件，并加入换行
                totalFile.write(txt)
                totalFile.write('\n')
                totalFile.flush()
