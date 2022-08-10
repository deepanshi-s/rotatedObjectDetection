import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from dataset import datasetLoaders
from arch import FPN
from lossFunction import lossFunction
from utils import score_iou
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--configFile", default='configs.yaml', help="Config file with parameters for training")
    
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    args.device = device
    
    with open(args.configFile) as f:
        yamlObject = yaml.load(f)

    #datatype
    trainSamples = yamlObject['trainSamples']
    valSamples = yamlObject['valSamples']
    testSamples = yamlObject['testSamples']
    batchSize = yamlObject['batchSize']
    lr = yamlObject['learningRate']
    epochs = yamlObject['epochs']
    betas = yamlObject['betas']
    deccay = yamlObject['weightDecay']
    
    trainLoader, valLoader, testLoader = datasetLoaders(trainSamples, valSamples, testSamples, batchSize)
    model = FPN()
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=betas, weight_decay=deccay)
    
    model.to(args.device)
    
    writer = SummaryWriter()
    
    lastScore = 0.0
    lastLoss = 99
    trainLastLoss = 99
    for numEpoch in range(1, epochs + 1):
        trainLossEpoch = []
        trainBoxLossEpoch = []
        trainclassLossEpoch = []
        
        valLossEpoch = []
        valBoxLossEpoch = []
        valclassLossEpoch = []
        
        score = []
        #training Step
        model.train()
        for i, batch in enumerate(trainLoader):
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)
            
            totalLoss, boxLoss, classLoss = lossFunction(prediction, label)
            

            loss = totalLoss.mean()
            loss.backward()
            optimizer.step()
            
            trainLossEpoch.append(totalLoss)
            trainBoxLossEpoch.append(boxLoss)
            trainclassLossEpoch.append(classLoss)
            
            
        writer.add_scalar("Loss/total_Loss", torch.mean(torch.cat(trainLossEpoch)), numEpoch)
        writer.add_scalar("Loss/box_Loss", torch.mean(torch.cat(trainBoxLossEpoch)), numEpoch)
        writer.add_scalar("Loss/classification_loss", torch.mean(torch.cat(trainclassLossEpoch)), numEpoch)
                
        #validation step
        with torch.no_grad():
            for i, batch in enumerate(valLoader):
                image, label = batch
                image = image.to(device)
                label = label.to(device)
                
                prediction = model(image)
                
                valLoss, boxLoss, classLoss = lossFunction(prediction, label)
                
                valLossEpoch.append(valLoss)
                valBoxLossEpoch.append(boxLoss)
                valclassLossEpoch.append(classLoss)
                
                iou = score_iou(np.array(prediction[:, 1:].reshape((5)).cpu()), np.array(label[:, 1:].reshape((5)).cpu()))
                score.append(iou)

        score = np.mean(np.array(score))
        writer.add_scalar("ValLoss/total_Loss", torch.mean(torch.cat(valLossEpoch)), numEpoch)
        writer.add_scalar("ValLoss/box_Loss", torch.mean(torch.cat(valBoxLossEpoch)), numEpoch)
        writer.add_scalar("ValLoss/classification_loss", torch.mean(torch.cat(valclassLossEpoch)), numEpoch)    
        writer.add_scalar("metrics/iou", score, numEpoch)
            
        curLoss = torch.mean(torch.cat(valLossEpoch))
        if curLoss < lastLoss:
            torch.save(model.state_dict(), "noUpsample/bestLoss.pickle")
            lastLoss = curLoss
        
        if lastScore < score:
            torch.save(model.state_dict(), "noUpsample/bestScore.pickle")
            lastScore = score
            
        if totalLoss.mean() < trainLastLoss:
            torch.save(model.state_dict(), "noUpsample/bestTrainLoss.pickle")
            trainLastLoss = totalLoss.mean()

        print('Epoch number: %d, trainLoss: %f, valLoss: %f, score: %f, bestScore: %f, lastLoss: %f' %(numEpoch, torch.mean(torch.cat(trainLossEpoch)), torch.mean(torch.cat(valLossEpoch)), score, lastScore, lastLoss)) 
               

if __name__ == '__main__':
    main()
    
