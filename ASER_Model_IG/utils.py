""" Utilities """
import logging
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import preproc
from IEMOCAPDatasetKFold import IemocapDatasetKFold
from MSPIMPROVDatasetKFold import MspImprovDatasetKFold
from MSPPODCASTDatasetKFold import MspPodcastDatasetKFold
import config


# import torchvision.datasets as dset


'''def get_data(dataset, data_path, cutout_length, validation, features, fold=None):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'iemocap_kfold':
        dset_cls = IemocapDatasetKFold
        n_classes = 4
    elif dataset == 'mspimprov_kfold':
        dset_cls = MspImprovDatasetKFold
        n_classes = 4
    elif dataset == 'msppodcast_kfold':
        dset_cls = MspPodcastDatasetKFold
        n_classes = 4
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform, features=features,
                        fold=fold)

    # assuming shape is NHW or NHWC
    shape = trn_data.data.shape
    if dataset == 'iemocap_kfold' or 'mspimprov_kfold' or 'msppodcast_kfold':
        input_channels = 1 if len(shape) == 4 else 1
        assert shape[2] == shape[3], "not expected shape = {}".format(shape)
        input_size = shape[2]
    else:
        input_channels = 3 if len(shape) == 4 else 1
        assert shape[1] == shape[2], "not expected shape = {}".format(shape)
        input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(
            dset_cls(root=data_path, train=False, download=True, transform=val_transform, features=features, fold=fold))

    return ret'''


def get_data(dataset, data_path, cutout_length, validation, features, fold=None):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'iemocap_kfold':
        dset_cls = IemocapDatasetKFold
        n_classes = 4
    elif dataset == 'mspimprov_kfold':
        dset_cls = MspImprovDatasetKFold
        n_classes = 4
    elif dataset == 'msppodcast_kfold':
        dset_cls = MspPodcastDatasetKFold
        n_classes = 4
    else:
        raise ValueError(dataset)

    # Data transformations (cutout_length may be related to augmentation)
    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    
    # Load training data with specified transformations and features
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform, features=features,
                        fold=fold)

    # Get the shape of the loaded data, assumed to be in the format (batch_size, channels, height, width)
    shape = trn_data.data.shape

    # Handle non-square input (height and width are not equal)
    if dataset in ['iemocap_kfold', 'mspimprov_kfold', 'msppodcast_kfold']:
        input_channels = 1 if len(shape) == 4 else 1  # Set input_channels to 1 if the data has 4 dimensions
        # Remove the assertion since height and width may differ
        height, width = shape[2], shape[3]  # Capture the height and width of the input data
        input_size = (height, width)  # Save input size as a tuple (height, width)
    else:
        input_channels = 3 if len(shape) == 4 else 1  # Assuming RGB data has 3 channels
        height, width = shape[1], shape[2]  # Height and width of the input
        input_size = (height, width)  # Save input size as a tuple (height, width)

    # Return input size, input channels, number of classes, and training data
    ret = [input_size, input_channels, n_classes, trn_data]

    # If validation is enabled, load validation data and append it to the result
    if validation:
        val_data = dset_cls(root=data_path, train=False, download=True, transform=val_transform, features=features, fold=fold)
        ret.append(val_data)

    return ret



def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


def num_parameters(model):
    return sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def precision(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def accuracy(output, target):
    pred = output.permute((0, 1))
    batch_size = pred.size(0)
    target = target.view((batch_size, -1))

    pred_c = pred.argmax(1)
    target_c = target.argmax(1)

    return pred_c.eq(target_c.expand_as(pred_c)).float().sum(0).mul_(1.0 / batch_size)

def weighted_accuracy(output, target, class_weights):
    # Check if output or target are dictionaries, if so extract the tensors
    if isinstance(output, dict):
        output = output['logits']  # Extract the appropriate tensor (e.g., logits)
    if isinstance(target, dict):
        target = target['labels']  # Extract the target tensor (e.g., labels)

    # Flatten tensors to match shapes
    pred = output.permute((0, 1))  # Adjust dimensions if necessary
    batch_size = pred.size(0)
    target = target.view(batch_size, -1)

    # Get predicted classes and true classes
    pred_c = pred.argmax(1)
    target_c = target.argmax(1)

    # Convert class_weights dictionary to a tensor
    if isinstance(class_weights, dict):
        # Assuming class_weights has class indices as keys
        class_weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32, device=target_c.device)

    # Check if class_weights tensor is on the same device as target_c
    class_weights = class_weights.to(target_c.device)

    # Compute correct predictions
    correct = pred_c.eq(target_c.expand_as(pred_c)).float()

    # Validate class_weights: ensure target_c indices are within range
    if target_c.max().item() >= len(class_weights):
        raise ValueError(f"target_c contains class indices ({target_c.max().item()}) greater than the available class weights ({len(class_weights)})")

    # Apply class weights
    weights = class_weights[target_c]  # Assign weights based on true classes

    # Compute weighted correct predictions
    weighted_correct = correct * weights

    # Compute weighted accuracy
    weighted_accuracy = weighted_correct.sum(0).mul_(1.0 / batch_size)
    
    return weighted_accuracy



def scores(output, target, weights):
    pred = output.argmax(1)
    target = target.argmax(1)
    t = target.cpu()
    p = pred.cpu()
    sample_weights = []
    for i in t:
        sample_weights.append(weights[i.item()])
    wa = accuracy_score(t, p, sample_weight=sample_weights)
    f1 = f1_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    recall_s = recall_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    precision_s = precision_score(t, p, average="weighted", zero_division=0.0, sample_weight=sample_weights)
    return wa, f1, recall_s, precision_s

'''def scores(output, target, weights):
    pred = output.argmax(1)

    if len(target.shape) > 1:
        target = target.argmax(1)

    t = target.cpu()
    p = pred.cpu()

    sample_weights = [weights[i.item()] for i in t]

    # Compute weighted accuracy
    wa = accuracy_score(t, p, sample_weight=sample_weights)

    # Compute F1 scores for each class by setting `average=None`
    f1_per_class = f1_score(t, p, average=None, zero_division=0.0, sample_weight=sample_weights)

    # Compute recall and precision per class similarly
    recall_per_class = recall_score(t, p, average=None, zero_division=0.0, sample_weight=sample_weights)
    precision_per_class = precision_score(t, p, average=None, zero_division=0.0, sample_weight=sample_weights)

    return wa, f1_per_class, recall_per_class, precision_per_class'''

#actual
'''def scores(output, target, class_ids_for_name, weights):
    pred = output.argmax(1)  # Get predicted class indices
    target = target.argmax(1)  # Convert one-hot encoded target to class indices if needed

    # Convert indices to labels
    pred_labels = [list(class_ids_for_name.keys())[list(class_ids_for_name.values()).index(p.item())] for p in pred]
    target_labels = [list(class_ids_for_name.keys())[list(class_ids_for_name.values()).index(t.item())] for t in target]

    sample_weights = [weights[class_ids_for_name[label]] for label in target_labels]

    # Compute metrics
    wa = accuracy_score(target_labels, pred_labels, sample_weight=sample_weights)
    f1_per_class = f1_score(target_labels, pred_labels, labels=list(class_ids_for_name.keys()), average=None, zero_division=0.0, sample_weight=sample_weights)
    recall_per_class = recall_score(target_labels, pred_labels, labels=list(class_ids_for_name.keys()), average=None, zero_division=0.0, sample_weight=sample_weights)
    precision_per_class = precision_score(target_labels, pred_labels, labels=list(class_ids_for_name.keys()), average=None, zero_division=0.0, sample_weight=sample_weights)

    return wa, f1_per_class, recall_per_class, precision_per_class'''




def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
