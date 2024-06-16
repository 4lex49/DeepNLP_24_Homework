# -*- coding: utf-8 -*-
from torch import optim
from torch.utils.data import DataLoader
from utils import *
import torch.nn as nn
from utils import drawLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Seq2SeqArgs:
    embedding_dim = 256
    hidden_dim = 512
    num_epochs = 10
    batch_size = 4
    savePath = '.\\models\\Seq2SeqModel.pth'


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        # src的形状：(batch_size,num_steps,embed_size)
        src_embedded = self.embedding(src)
        # tgt的形状：(batch_size,num_steps,embed_size)
        tgt_embedded = self.embedding(tgt)
        _, (hidden, cell) = self.encoder(src_embedded)
        output, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc_out(output)


def train(trainDataset, dictionary, testLen, maxLen):
    # 创建模型
    vocab_size = len(dictionary)
    model = Seq2SeqModel(vocab_size=vocab_size, embedding_dim=Seq2SeqArgs.embedding_dim,
                         hidden_dim=Seq2SeqArgs.hidden_dim).to(device)

    # 创建数据集和数据加载器
    dataloader = DataLoader(trainDataset, batch_size=Seq2SeqArgs.batch_size, shuffle=True, drop_last=True)

    # 训练Seq2Seq，创建loss和优化器
    lossRecords = []
    criterion = nn.CrossEntropyLoss()  # pad token id
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(Seq2SeqArgs.num_epochs):
        model.train()
        for sentenceIds in dataloader:
            sentenceIds = sentenceIds.to(device)
            srcInput = sentenceIds[:, :testLen]
            optimizer.zero_grad()

            targetInput = sentenceIds[:, testLen:-1]
            targetOutput = sentenceIds[:, testLen+1:]
            outputs = model(srcInput, targetInput)

            targets = torch.zeros_like(outputs).to(device)
            for i in range(targets.size(0)):
                for j in range(targets.size(1)):
                    if targetOutput[i][j] != 2:  # padding id
                        targets[i, j, targetOutput[i][j]] = 1
            loss = criterion(outputs, targets).to(device)

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}/{Seq2SeqArgs.num_epochs}, Loss: {loss.item()}')
            lossRecords.append(loss.item())

    print("Training complete!")
    torch.save(model, Seq2SeqArgs.savePath)
    drawLoss(lossRecords, Seq2SeqArgs.num_epochs, 'Seq2Seq')


def test(testDataset, dictionary, testLength, maxLength):
    model = torch.load(Seq2SeqArgs.savePath).to(device)
    model.eval()
    dataloader = DataLoader(testDataset, batch_size=1)

    resultTups = []
    for groundIds in dataloader:
        testIds = groundIds[:, :testLength].to(device)
        testIds = testIds.to(device)
        generateTextIds = testIds
        for _ in range(maxLength):
            with torch.no_grad():
                outputs = model(testIds, generateTextIds)
                nextTokenId = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(0)
                testIds = torch.cat((testIds, nextTokenId), dim=-1)
                generateTextIds = torch.cat((generateTextIds, nextTokenId), dim=-1)

            if nextTokenId.squeeze(0) == dictionary.token2id['<eos>']:
                break
        inputText = [dictionary[textId.item()] for textId in testIds.squeeze_(0)[:testLength]]
        generateText = [dictionary[textId.item()] for textId in generateTextIds.squeeze_(0)]
        groundText = [dictionary[textId.item()] for textId in groundIds.squeeze_(0)]
        resultTups.append((inputText, generateText, groundText))
        print('Input: ', inputText)
        print('Gen: ', generateText)
        print('GroundTruth: ', groundText)
    return resultTups
