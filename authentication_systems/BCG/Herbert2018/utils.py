import torch
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, f1_score
import torch.nn as nn

def create_weight(ytrue, device, num_classes=2): # 40
    w = []
    for y in ytrue:
        if y:
            w.append(num_classes-1)  # higher weight for in class so we dont always predict other
        else:
            w.append(1)

    w = torch.FloatTensor(w).to(device)
    w = w.unsqueeze(1)
    return w


def train_epoch(epoch, model, dataloader, optimizer, criterion, device):
    model.train()

    epoch_loss = 0
    correct = 0
    trues, preds = [], []  # need to actually save all for f1 calc

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        x = x.to(torch.float)
        y = y.to(torch.float)
        y = y.unsqueeze(1)  # bs x 1

        optimizer.zero_grad()
        out = model(x)

        # weight if we want
        weights = create_weight(y, device)
        criterion = nn.BCELoss(weight=weights)
        #criterion = nn.BCELoss()

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = out.round()
        correct += pred.eq(y).sum().item()

        trues.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

    epoch_acc = correct / len(dataloader.dataset)
    f1 = f1_score(trues, preds)

    return epoch_loss, epoch_acc, f1


def test_epoch(epoch, model, dataloader, criterion, device, binary=False):
    model.eval()
    preds = []
    trues = []
    epoch_loss = 0

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)
            x = x.to(torch.float)
            y = y.to(torch.float)
            y = y.unsqueeze(1)  # bs x 1

            out = model(x)
            weights = create_weight(y, device)
            criterion = nn.BCELoss(weight=weights)
            loss = criterion(out, y)
            epoch_loss += loss.item()

            pred = out.round()

            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())

        correct = np.equal(trues, preds).sum()
        epoch_acc = correct / len(dataloader.dataset)
        f1 = f1_score(trues, preds)

        # ALSO get EER cause we can at this point :) 
    
    return epoch_loss, epoch_acc, f1


def eval_eer(model, eval_loader, device):

    # return eer, eer_thresh
    model.eval()
    with torch.no_grad():
        scores, trues = [], []
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            x, y = x.to(torch.float), y.to(torch.float)

            out = model(x)
            scores.extend(out.cpu().tolist())
            trues.extend(y.cpu().tolist())

    fpr, tpr, thresholds = roc_curve(trues, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, eer_threshold
