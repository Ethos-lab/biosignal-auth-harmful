import torch
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval):
    model.train()

    epoch_loss = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)
        x = x.to(torch.float)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    epoch_acc = correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def test_epoch(epoch, model, dataloader, criterion, device):
    model.eval()
    preds = []
    trues = []
    epoch_loss = 0

    with torch.no_grad():
        for x, y in dataloader:

            trues.append(y.numpy())

            x, y = x.to(device), y.to(device)
            x = x.to(torch.float)
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item()

            pred = out.argmax(1)
            preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        correct = np.equal(trues, preds).sum()
        epoch_acc = correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# ==============================================================================


def train_siamese_epoch(epoch, model, dataloader, optimizer, criterion, device, log_interval):
    model.train()

    epoch_loss = 0
    for batch_idx, data in enumerate(dataloader):
        x1, x2, y = data
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        x1, x2 = x1.to(torch.float), x2.to(torch.float)
        y = y.to(torch.float)

        optimizer.zero_grad()
        out = model(x1, x2)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    return epoch_loss


def test_siamese_epoch(epoch, model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for data in dataloader:
            x1, x2, y = data
            x1, x2 = x1.to(device), x2.to(device)
            x1, x2 = x1.to(torch.float), x2.to(torch.float)
            y = y.to(torch.float)
            y = y.to(device)

            out = model(x1, x2)
            loss = criterion(out, y)
            epoch_loss += loss.item()

    return epoch_loss


def eval_eer(model, eval_loader, device):
    # return eer, eer_thresh

    all_scores, all_y = [], []
    for data in eval_loader:
        x1, x2, y = data
        x1, x2 = x1.to(device), x2.to(device)
        x1, x2 = x1.to(torch.float), x2.to(torch.float)

        scores = model(x1, x2).cpu().tolist()  # similarity score 
        all_scores.extend(scores)
        all_y.extend(y)

    fpr, tpr, thresholds = roc_curve(all_y, all_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return eer, eer_threshold
