import torch
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


def train_epoch(epoch, model, dataloader, optimizer, criterion, device):
    model.train()

    epoch_loss = 0
    correct = 0
    pctones = 0
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


def eval_eer(model, eval_loader, device):

    # return eer, eer_thresh
    model.eval()
    with torch.no_grad():
        all_scores, all_y = [], []
        for data in eval_loader:
            x1, x2, y = data
            x1, x2 = x1.to(device), x2.to(device)
            x1, x2 = x1.to(torch.float), x2.to(torch.float)

            feat1 = model.get_template(x1)
            feat2 = model.get_template(x2)
            #rmse = torch.sqrt(torch.mean((feat1-feat2)**2, axis=1))
            rmse = np.linalg.norm(feat2.cpu()-feat1.cpu(), 2, axis=1)
            rmse = np.clip(rmse, 1e-6, rmse.max())
            rmse = 1/rmse

            #all_scores.extend(rmse.cpu().tolist())
            all_scores.extend(rmse.tolist())
            all_y.extend(y.tolist())

    fpr, tpr, thresholds = roc_curve(all_y, all_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return eer, eer_threshold


def predict(model, loader, device, eer_thresh):
    # Get acc at given thresh
    model.eval()
    with torch.no_grad():
        correct = 0
        num_samps = 0
        for data in loader:
            x1, x2, y = data
            x1, x2 = x1.to(device), x2.to(device)
            x1, x2 = x1.float(), x2.float()

            feat1 = model.get_template(x1)
            feat2 = model.get_template(x2)
            #rmse = torch.sqrt(torch.mean((feat1-feat2)**2, axis=1)).cpu().tolist()
            rmse = np.linalg.norm(feat2.cpu()-feat1.cpu(), 2, axis=1)
            rmse = np.clip(rmse, 1e-6, rmse.max())
            rmse = 1/rmse

            for ri, r in enumerate(rmse):
                if r > eer_thresh:
                    pred = 1
                else:
                    pred = 0
                if pred == y[ri]:  correct += 1
                num_samps += 1

    return correct/num_samps  
