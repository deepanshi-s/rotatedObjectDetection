import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from dataset import datasetLoaders
from arch import FPN
from lossFunction import lossFunction
from utils import score_iou
from torch.utils.tensorboard import SummaryWriter


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
    
    trainLoader, valLoader, testLoader = datasetLoaders(trainSamples, valSamples, testSamples, batchSize)
    model = FPN()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    model.to(args.device)
    
    writer = SummaryWriter()
    
    valLoss = 99
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
            
            #print('line 62', label, prediction, totalLoss, boxLoss, classLoss)
            loss = totalLoss.mean()
            loss.backward()
            optimizer.step()
            
            trainLossEpoch.append(totalLoss)
            trainBoxLossEpoch.append(boxLoss)
            trainclassLossEpoch.append(classLoss)
            
            
        writer.add_scalar("Loss/total_Loss", torch.mean(torch.cat(trainLossEpoch)))
        writer.add_scalar("Loss/box_Loss", torch.mean(torch.cat(trainBoxLossEpoch)))
        writer.add_scalar("Loss/classification_loss", torch.mean(torch.cat(trainclassLossEpoch)))
            
        
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
        writer.add_scalar("ValLoss/total_Loss", torch.mean(torch.cat(valLossEpoch)))
        writer.add_scalar("ValLoss/box_Loss", torch.mean(torch.cat(valBoxLossEpoch)))
        writer.add_scalar("ValLoss/classification_loss", torch.mean(torch.cat(valclassLossEpoch)))    
        writer.add_scalar("metrics/iou", score)
            
        curLoss = torch.mean(torch.cat(valLossEpoch))
        if curLoss < valLoss:
            torch.save(model.state_dict(), "best.pickle")
            valLoss = curLoss
            
        print('Epoch number: %d, trainLoss: %d, valLoss: %d' %(numEpoch, trainLossEpoch, valLossEpoch)) 
                

if __name__ == '__main__':
    main()
    