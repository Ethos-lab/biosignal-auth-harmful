import torch
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, f1_score

def train_epoch(epoch, model, dataloader, optimizer, criterion, device, binary=False):
    model.train()

    epoch_loss = 0
    correct = 0
    trues, preds = [], []  # need to actually save all for f1 calc

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        x = x.to(torch.float)
        y = y.to(torch.float).unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if binary:
            pred = out.round()  # if 1, is me
        else:
            pred = out.argmax(1)
        correct += pred.eq(y).sum().item()

        trues.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

    epoch_acc = correct / len(dataloader.dataset)
    #metric = f1_score(trues, preds)
    metric = np.mean(preds)  # TODO debug

    return epoch_loss, epoch_acc, metric


def test_epoch(epoch, model, dataloader, criterion, device, binary=False):
    model.eval()
    preds = []
    trues = []
    epoch_loss = 0

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)
            x = x.to(torch.float)
            y = y.to(torch.float).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item()

            if binary:
                pred = out.round()
            else:
                pred = out.argmax(1)

            preds.extend(pred.flatten().cpu().tolist())
            trues.extend(y.flatten().cpu().tolist())

        correct = np.equal(trues, preds).sum()
        epoch_acc = correct / len(dataloader.dataset)
        #metric = f1_score(trues, preds)
        metric = np.mean(preds)

    return epoch_loss, epoch_acc, metric


def eval_eer(model, eval_loader, device, binary):

    # return eer, eer_thresh
    model.eval()
    with torch.no_grad():
        scores, trues = [], []
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            x, y = x.to(torch.float), y.to(torch.float)

            out = model(x)
            scores.extend(out.flatten().cpu().tolist())
            trues.extend(y.flatten().cpu().tolist())

    fpr, tpr, thresholds = roc_curve(trues, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    num_correct = 0
    for s, y in zip(scores, trues):
        p = 1 if s >= eer_threshold else 0
        if p == y:  num_correct += 1
    num_correct /= len(scores)  # actually 'fraction_correct'

    return eer, eer_threshold, num_correct
