import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import synthesize_data


class starDataset(Dataset):
    """Star Dataset with labels

    Args:
        n_samples {int} -- items in dataset (default: 50000)
    """
    def __init__(self, n_samples = 50000):
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        image, label = synthesize_data()
        
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label.reshape((label.shape[0])), dtype=torch.float32)
        has_star = (~torch.isnan(label[0])).float().reshape(1)
        label = torch.cat((has_star, label), dim = 0)
        label[[1, 2]] /= 200.
        label[3] /= 2*np.pi
        label[[4,5]] /= 200.
        return image[None], label
   

def datasetLoaders(trainSamples, valSamples, testSamples, batchSize):
    """Loads the train, val and test dataset as torch.utils.data.DataLoader

    Args:
        trainSamples (int): number of train samples
        valSamples (int): number of validation samples
        testSamples (int): number of test samples
        batchSize (int): batch size for train, test and val dataloaders

    Returns:
        torch.utils.data.DataLoader: return train, val and test dataset objects
    """
    trainDataset = starDataset(trainSamples)
    trainLoader = DataLoader(
            dataset=trainDataset,
            batch_size = batchSize,
            num_workers=8,
            shuffle=True  
    )
    
    valDataset = starDataset(valSamples)
    valLoader = DataLoader(
            dataset=valDataset,
            batch_size = 1,
            num_workers=8,
            shuffle=True  
    )
    
    testDataset = starDataset(testSamples)
    testLoader = DataLoader(
            dataset=testDataset,
            batch_size = batchSize,
            num_workers=8,
            shuffle=True  
    )
    
    return trainLoader, valLoader, testLoader
    
    
    
    
        


