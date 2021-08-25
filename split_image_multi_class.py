import os
import shutil
import numpy as np
import core_split as split

"""
List variable as parameter as listed below, carefully change value as dependency
"""
path_ori_benign = 'original/benign/SOB'
path_ori_malignant = 'original/malignant/SOB'
path_subclass_scenario = 'subclass_scenario'
list_magnification = ['40X', '100X', '200X', '400X']
sub_class_benign = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
sub_class_malignant = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
subclasses = ['tubular_adenoma', 'phyllodes_tumor', 'papillary_carcinoma',
              'mucinous_carcinoma', 'lobular_carcinoma', 'fibroadenoma',
              'ductal_carcinoma', 'adenosis']
rate_test = 10
rate_val = 10

if __name__ == '__main__':
    list_dir_subclasses = split.find_list_dir(
        sub_class_benign,
        sub_class_malignant,
        path_ori_benign,
        path_ori_malignant
    )
    list_file_classes = split.file_splitting_subclass(list_dir_subclasses, subclasses, list_magnification)
    split.get_info_total(list_file_classes, subclasses)
    list_subclasses = [_class['split'] for _class in list_file_classes]
    dict_magnification_subclasses = split.build_dict_magnification_subclasses(list_subclasses)

    # copying test image file
    already_test = []
    for magnification in dict_magnification_subclasses:
        for sub_class in dict_magnification_subclasses[magnification]:
            choosed = dict_magnification_subclasses[magnification][sub_class]
            len_test = split.test_len(choosed, rate_test)
            rand = np.random.choice(len(choosed), len_test, False)
            for image in rand:
                shutil.copy(choosed[image], os.path.join(path_subclass_scenario, 'test', magnification, sub_class))
                already_test.append(choosed[image])
    print("test size {}".format(len(already_test)))

    for sub_class in list_subclasses:
        for file in sub_class:
            lvl = file.split("\\")[3]
            scl = file.split("/")[2].split("\\")[1]
            if file in already_test:
                continue
            if lvl not in dict_magnification_subclasses:
                dict_magnification_subclasses[lvl] = {}
            if scl not in dict_magnification_subclasses[lvl]:
                dict_magnification_subclasses[lvl][scl] = []
            dict_magnification_subclasses[lvl][scl].append(file)

    # copying validation image file
    already_val = []
    for magnification in dict_magnification_subclasses:
        for sub_class in dict_magnification_subclasses[magnification]:
            choosed = dict_magnification_subclasses[magnification][sub_class]
            len_test = split.val_len(choosed, rate_val)
            rand = np.random.choice(len(choosed), len_test, False)
            for image in rand:
                shutil.copy(choosed[image], os.path.join(path_subclass_scenario, 'val', magnification, sub_class))
                already_val.append(choosed[image])
    print("validation size {}".format(len(already_val)))

    dict_magnification_subclasses = {}
    for sub_class in list_subclasses:
        for file in sub_class:
            lvl = file.split("\\")[3]
            scl = file.split("/")[2].split("\\")[1]
            if file in already_test or file in already_val:
                continue
            if lvl not in dict_magnification_subclasses:
                dict_magnification_subclasses[lvl] = {}
            if scl not in dict_magnification_subclasses[lvl]:
                dict_magnification_subclasses[lvl][scl] = []
            dict_magnification_subclasses[lvl][scl].append(file)

    # copying training image file
    already_train = []
    for magnification in dict_magnification_subclasses:
        for sub_class in dict_magnification_subclasses[magnification]:
            choosed = dict_magnification_subclasses[magnification][sub_class]
            for image in choosed:
                shutil.copy(image, os.path.join(path_subclass_scenario, 'train', magnification, sub_class))
                already_train.append(image)
    print("training size {}".format(len(already_train)))

