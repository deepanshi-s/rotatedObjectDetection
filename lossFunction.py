import torch
import torch.nn.functional as F

def lossFunction(pred, target):
    """Compute loss

    Args:
        pred (tensor batch): p(star), x, y, yaw, w, h
        target (tensor batch):  p(star), x, y, yaw, w, h

    Returns:
        loss: not averaged
    """
    
    assert pred.shape[-1] == 6
    assert target.shape[-1] == 6
    
    indexNoStar = torch.nonzero(target[:, 0] == 0, as_tuple=True)
    
    boxLoss = modulatedLoss(pred[:, 1:], target[:, 1:])
    boxLoss[indexNoStar] = 0
    
    classLoss = torch.nn.BCELoss(reduction='none')(pred[:, 0], target[:, 0])
    totalLoss = 0.5*boxLoss + 0.5*classLoss
    
    return totalLoss, boxLoss, classLoss
    

def modulatedLoss(pred, target):
    """modulated loss fumction: https://arxiv.org/pdf/1911.08299.pdf

    Args:
        pred (tensor batch): predicted bouding box values x, y, yaw, w, h
        target (tensor batch): ground truth bouding box values x, y, yaw, w, h

    Returns:
        loss for each pred-target pair
    """
    assert pred.shape[-1] == 5
    assert target.shape[-1] == 5
    
    x_true, y_true, yaw_true, w_true, h_true = target[:, 0], target[:, 1], target[:, 2], target[:, 3], target[:, 4]
    x_pred, y_pred, yaw_pred, w_pred, h_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    
    loss = torch.abs(x_true - x_pred) + torch.abs(y_true - y_pred) + torch.abs(w_true - w_pred) + torch.abs(h_true - h_pred) + torch.min(
        torch.abs(yaw_true - yaw_pred),
        torch.abs(0.5 - torch.abs(yaw_true - yaw_pred))
    )

    return loss
