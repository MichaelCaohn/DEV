import random
import torch
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import torch.utils.data as util_data
from data_list import ImageList
import pre_process as prep
import torch.nn as nn
from torch.autograd import Variable



def predict_loss(cls, y_pre): #requires how the loss is calculated for the preduct value and the ground truth value
    """
    Calculate the cross entropy loss for prediction of one picture
    :param y:
    :param y_pre:
    :return:
    """
    cls_torch = np.full(1, cls)
    pre_cls_torch = torch.from_numpy(y_pre.astype(float))
    entropy = nn.CrossEntropyLoss()
    return entropy(pre_cls_torch, cls_torch)


def split_set(source_path, class_num, split = 0.4):
    """
    Split the source list into a list of list of source and a list of list of validation
    :param source_path:
    :param class_num:
    :param split:
    :return:
    """
    source_list = open(source_path).readlines()
    src_list = []
    val_list = []
    for i in range(class_num):
        src_list.append([j for j in source_list if int(j.split(" ")[1].replace("\n", "")) == i])
    for j in range(len(src_list)):
        val = []
        source_len = len(src_list[j])
        val_len = math.ceil(source_len * split)
        for k in range(val_len):
            val.append(src_list[i][-1])
            src_list[i].remove(src_list[i][-1])
        val_list.append(val_len)
    return src_list, val_list

def cross_validation_loss(feature_network, predict_network, src_cls_list, target_path, val_cls_list, class_num, resize_size, crop_size, batch_size, use_gpu):
    """
    Main function for computing the CV loss
    :param feature_network:
    :param predict_network:
    :param src_cls_list:
    :param target_path:
    :param val_cls_list:
    :param class_num:
    :param resize_size:
    :param crop_size:
    :param batch_size:
    :return:
    """

    cross_val_loss = 0

    prep_dict_val = prep.image_train(resize_size=resize_size, crop_size=crop_size)
    # load different class's image
    for cls in range(class_num):

        dsets_val = ImageList(val_cls_list[cls], transform=prep_dict_val)
        dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)

        # prepare validation feature and predicted label for validation
        iter_val = iter(dset_loaders_val)
        val_input, val_labels = iter_val.next()
        if use_gpu:
            val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
        else:
            val_input, val_labels = Variable(val_input), Variable(val_labels)
        val_feature, _ = feature_network(val_input)
        _, pred_label = predict_network(val_input)
        w, h = pred_label.shape
        error = np.zeros(1)
        error[0] = predict_loss(cls, pred_label.reshape(1, w*h)).numpy()
        error = error.reshape(1,1)
        for _ in range(len(val_cls_list[cls]) - 1):
            val_input, val_labels = iter_val.next()
            # val_feature1 = feature_network(val_input)
            val_feature_new, _ = feature_network(val_input)
            val_feature = np.append(val_feature, val_feature_new, axis=0)
            error = np.append(error, [[predict_loss(cls, predict_network(val_input)[1]).numpy()]], axis=0)

        print('The class is {}\n'.format(cls))

        cross_val_loss = cross_val_loss + error.sum()
    return cross_val_loss/class_num
