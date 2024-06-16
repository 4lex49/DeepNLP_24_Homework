# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import drawLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerArgs:
    # 模型参数
    embeddingDim = 256
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 1024
    # 训练循环
    num_epochs = 20
    batch_size = 4

    savePath = '.\\models\\TransformerModel.pth'


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embeddingDim, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeddingDim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embeddingDim))
        self.transformer = nn.Transformer(embeddingDim, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, batch_first=True)
        self.fc_out = nn.Linear(embeddingDim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory)
        return self.fc_out(output)


def train(trainDataset, dictionary, testLen, maxLen):
    # 创建模型
    vocab_size = len(dictionary)
    model = TransformerModel(vocab_size=vocab_size, embeddingDim=TransformerArgs.embeddingDim,
                             nhead=TransformerArgs.nhead, num_encoder_layers=TransformerArgs.num_encoder_layers,
                             num_decoder_layers=TransformerArgs.num_decoder_layers,
                             dim_feedforward=TransformerArgs.dim_feedforward,
                             max_seq_length=maxLen).to(device)

    # 创建数据集和数据加载器
    dataloader = DataLoader(trainDataset, batch_size=TransformerArgs.batch_size, shuffle=True, drop_last=True)

    # 训练Transformer，创建loss和优化器
    lossRecords = []
    criterion = nn.CrossEntropyLoss()  # pad token id
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(TransformerArgs.num_epochs):
        model.train()
        for sentenceIds in dataloader:
            optimizer.zero_grad()
            sentenceIds = sentenceIds.to(device)
            src = sentenceIds[:, :testLen]
            targetInput = sentenceIds[:, testLen:-1]
            targetOutput = sentenceIds[:, testLen+1:]
            outputs = model(src, targetInput)

            targets = torch.zeros_like(outputs, device=device)
            for i in range(targets.size(0)):
                for j in range(targets.size(1)):
                    if targetOutput[i][j] != 2:  # padding id
                        targets[i, j, targetOutput[i][j]] = 1
            loss = criterion(outputs, targets).to(device)

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch + 1}/{TransformerArgs.num_epochs}, Loss: {loss.item()}')
            lossRecords.append(loss.item())

    print("Training complete!")
    torch.save(model, TransformerArgs.savePath)
    drawLoss(lossRecords, TransformerArgs.num_epochs, 'Transformer')


def test(testDataset, dictionary, testLength, maxLength):
    model = torch.load(TransformerArgs.savePath).to(device)
    model.eval()
    dataloader = DataLoader(testDataset, batch_size=1)

    resultTups = []
    for groundIds in dataloader:
        testIds = groundIds[:, :testLength].to(device)
        testIds = testIds.to(device)
        generateTextIds = torch.LongTensor([dictionary.token2id['<bos>']]).to(device).unsqueeze_(0)
        for _ in range(maxLength - testLength):
            with torch.no_grad():
                outputs = model(testIds, generateTextIds)
                nextTokenId = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(0)
                generateTextIds = torch.cat((generateTextIds, nextTokenId), dim=-1)

            if nextTokenId.squeeze(0) == dictionary.token2id['<eos>']:
                break
        inputText = [dictionary[textId.item()] for textId in testIds.squeeze_(0)]
        generateText = [dictionary[textId.item()] for textId in generateTextIds.squeeze_(0)]
        groundText = [dictionary[textId.item()] for textId in groundIds.squeeze_(0)]
        resultTups.append((inputText, generateText, groundText))
        print('Input: ', inputText)
        print('Gen: ', generateText)
        print('GroundTruth: ', groundText)
    return resultTups
