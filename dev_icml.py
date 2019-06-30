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

def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error # weight correspond to Ntr/Nts, error correspond to validation error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1),rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta

def get_weight(source_feature, target_feature, validation_feature): # 这三个feature根据类别不同，是不一样的. source与target这里需注意一下数据量threshold 2倍的事儿
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape  
    N_t, _d = target_feature.shape
    if float(N_s)/N_t > 2:
        source_feature = random_select_src(source_feature, target_feature)
    else:
        source_feature = source_feature.copy()

    print('num_source is {}, num_target is {}, ratio is {}\n'.format(N_s, N_t, float(N_s) / N_t)) #check the ratio

    target_feature = target_feature.copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t,dtype=np.int32) # 1->source 0->target
    feature_for_train,feature_for_test, label_for_train,label_for_test = train_test_split(all_feature, all_label, train_size = 0.8)

    # here is train, test split, concatenating the data from source and target

    decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    val_acc = []
    domain_classifiers = []
    
    for decay in decays:
        domain_classifier = MLPClassifier(hidden_layer_sizes=(d, d, 2),activation='relu',alpha=decay)
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
        print('decay is %s, val acc is %s'%(decay, acc))
        
    index = val_acc.index(max(val_acc))
    
    print('val acc is')
    print(val_acc)

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:,:1] / domain_out[:,1:] * N_s * 1.0 / N_t #(Ntr/Nts)*(1-M(fv))/M(fv)

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

def cross_validation_loss(feature_network, predict_network, src_list, target_path, val_list, resize_size, crop_size, batch_size, use_gpu):
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

    tar_list = open(target_path).readlines()
    cross_val_loss = 0

    prep_dict_val = prep_dict_source = prep_dict_target = prep.image_train(resize_size=resize_size, crop_size=crop_size)
    # load different class's image
    
    dsets_src = ImageList(src_list, transform=prep_dict_source)
    dset_loaders_src = util_data.DataLoader(dsets_src, batch_size=batch_size, shuffle=True, num_workers=4)

    dsets_val = ImageList(val_list, transform=prep_dict_val)
    dset_loaders_val = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)

    dsets_tar = ImageList(tar_list, transform=prep_dict_target)
    dset_loaders_tar = util_data.DataLoader(dsets_val, batch_size=batch_size, shuffle=True, num_workers=4)

    # prepare source feature
    iter_src = iter(dset_loaders_src)
    src_input, src_labels = iter_src.next()
    if use_gpu:
        src_input, src_labels = Variable(src_input).cuda(), Variable(src_labels).cuda()
    else:
        src_input, src_labels = Variable(src_input), Variable(src_labels)
    src_feature, _ = feature_network(src_input)
    for _ in range(len(src_list) - 1):
        src_input, src_labels = iter_src.next()
        src_feature_new, _ = feature_network(src_input)
        src_feature = np.append(src_feature, src_feature_new, axis=0)

    # prepare target feature
    iter_tar = iter(dset_loaders_tar)
    tar_input, _ = iter_tar.next()
    if use_gpu:
        tar_input, _ = Variable(tar_input).cuda(), Variable(_).cuda()
    else:
        src_input, _ = Variable(tar_input), Variable(_)
    tar_feature, _ = feature_network(tar_input)
    for _ in range(len(tar_list) - 1):
        tar_input, _ = iter_tar.next()
        tar_feature_new, _ = feature_network(tar_input)
        tar_feature = np.append(tar_feature, tar_feature_new, axis=0)

    # prepare validation feature and predicted label for validation
    iter_val = iter(dset_loaders_val)
    val_input, val_labels = iter_val.next()
    if use_gpu:
        val_input, val_labels = Variable(val_input).cuda(), Variable(val_labels).cuda()
    else:
        val_input, val_labels = Variable(val_input), Variable(val_labels)
    val_feature, _ = feature_network(val_input)
    pred_label = predict_network(val_input)[1]
    w, h = pred_label.shape
    error = np.zeors(1)
    error[0] = predict_loss(cls, pred_label.reshape(1, w*h)).numpy()
    error = error.reshape(1,1)
    for _ in range(len(val_list) - 1):
        val_input, val_labels = iter_val.next()
        val_feature_new, _ = feature_network(val_input)
        val_feature = np.append(val_feature, val_feature_new, axis=0)
        error = np.append(error, [[predict_loss(cls, predict_network(val_input)[1]).numpy()]], axis=0)

    print('The class is {}\n'.format(cls))
    weight = get_weight(src_feature, tar_feature, val_feature)
    cross_val_loss = cross_val_loss + get_dev_risk(weight, error)

    return cross_val_loss
