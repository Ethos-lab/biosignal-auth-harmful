" Utils for their matching for identify verification 'and periodic re-authentication "
import torch
import numpy as np


def match_real(x1, x2, model, train_threshold=False):  # returns distance
    " Let's assume they're already in batched tensors on device "
    assert not train_threshold  # want the signature to be the same as match_binary 
    with torch.no_grad():
        x1, x2 = torch.FloatTensor(x1), torch.FloatTensor(x2)
        x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)

        feat1 = model.get_features(x1)[0]
        feat2 = model.get_features(x2)[0]

        distance = np.linalg.norm(feat1 - feat2, ord=2)

    return distance

def get_binary_template_thresholds(dataloader, model):

    model.eval()

    device = next(model.parameters()).device
    all_features = []
    for x, y in dataloader:
    
        x, y = x.to(device), y.to(device)
        x, y = x.float(), y.float()
        feats = model.get_features(x)
        all_features.append(feats.detach().numpy())

    all_features = np.concatenate(all_features)
    thresholds = np.median(all_features, 0)

    return thresholds.flatten()

def match_binary(x1, x2, model, median_features):
    """
    Binary features take values in the range [0, 1] and are obtained by quantizing the real 
    features
    """
    #train_threshold = get_binary_template_thresholds(train_loader, model)


    x1, x2 = torch.FloatTensor(x1), torch.FloatTensor(x2)
    x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)

    feat1 = model.get_features(x1)[0].flatten()
    feat2 = model.get_features(x2)[0].flatten()

    numfeats = len(feat1)

    # Now create binary templates: 
    T1 = [True if feat1[i] >= median_features[i] else False for i in range(numfeats)]
    T2 = [True if feat2[i] >= median_features[i] else False for i in range(numfeats)]

    distance = [t1 ^ t2 for (t1, t2) in zip(T1,T2)]
    distance = np.sum(distance)/numfeats
    return distance 

