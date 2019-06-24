import math

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

def dimension_red(src_list):
