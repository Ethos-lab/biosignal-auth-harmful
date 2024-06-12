import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

def to_device_and_float(device, *args):
    ret = []
    for a in args:
        a = a.float().to(device)
        ret.append(a)
    return ret

def create_weight(bid, device):  # NOTE the weights differ depending on the num_classes ratios
    w = []
    for y in bid:
        if y:
            # TODO changed from March from 0.96 / 0.05 to half half
            #w.append(0.95)  # higher weight for my class so we dont always predict other
            w.append(0.5)
        else:
            #w.append(0.05)  # TODO test this
            w.append(0.5)
    w = torch.FloatTensor(w).to(device)
    return w

def train_epoch(epoch, model, dataloader, optimizer, criterion1, criterion2, device, ALPHA=0.0):
    # Remember criterion1 is hr (l1), criterion2 is identity (bce)
    model.train()

    epoch_loss = 0
    correct = 0
    pctones = 0
    for batch_idx, (x, bid, hr) in enumerate(dataloader):

        # turn 1d into 2d bid y, dumb thing
        #bid = torch.stack([bid, ~bid]).T
        bid = bid.unsqueeze(1)
        x, bid, hr = to_device_and_float(device, x, bid, hr)
        
        optimizer.zero_grad()
        out_hr, out_bid = model(x)

        weights = create_weight(bid, device)
        weights = weights.unsqueeze(1)

        weighted_bid_criterion = nn.BCELoss(weight=weights).to(device)
        weighted_bid_loss = weighted_bid_criterion(out_bid, bid)

        weighted_hr_criterion = nn.L1Loss().to(device)
        weighted_hr_loss = weighted_hr_criterion(out_hr, hr)  # weighted by alpha, not by class

        #loss = criterion1(out_hr, hr) + 10*criterion2(out_bid, bid)
        #loss = criterion2(out_bid, bid) + ALPHA*criterion1(out_hr, hr)  # TODO see what kind of scales these are on; ideal if on [0, 1]
        loss = weighted_bid_loss + ALPHA*weighted_hr_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = out_bid.round()
        correct += pred.eq(bid).sum().item()

    epoch_acc = correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def test_epoch(epoch, model, dataloader,  criterion1, criterion2, device, ALPHA=0.0):
    model.eval()
    preds = []
    trues = []
    epoch_loss = 0

    with torch.no_grad():
        for x, bid, hr in dataloader:

            trues.append(bid.numpy())
            bid = bid.unsqueeze(1)
            x, bid, hr = to_device_and_float(device, x, bid, hr)

            out_hr, out_bid = model(x)
            #loss = criterion1(out_hr, hr) + 10*criterion2(out_bid, bid)
            #loss = criterion2(out_bid, bid) + ALPHA*criterion1(out_hr, hr)

            weights = create_weight(bid, device)
            weights = weights.unsqueeze(1)

            weighted_bid_criterion = nn.BCELoss(weight=weights).to(device)
            weighted_bid_loss = weighted_bid_criterion(out_bid, bid)

            weighted_hr_criterion = nn.L1Loss().to(device)
            weighted_hr_loss = weighted_hr_criterion(out_hr, hr)

            loss = weighted_bid_loss + ALPHA*weighted_hr_loss
            
            epoch_loss += loss.item()
            pred = out_bid.round()
            preds.append(pred.detach().cpu().numpy())
        

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)


        correct = np.equal(trues, preds.flatten()).sum()
        epoch_acc = correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def eval_eer(model, eval_loader, device):

    # return eer, eer_thresh
    model.eval()
    with torch.no_grad():
        all_scores, all_y = [], []
        for data in eval_loader:
            x, bid, hr = data
            all_y.extend(bid.numpy().tolist())

            x, bid, hr = to_device_and_float(device, x, bid, hr)
            out_hr, out_bid = model(x)
            all_scores.extend(out_bid.flatten().cpu().numpy().tolist())

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
