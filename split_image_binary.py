import os
import shutil
import numpy as np
import core_split as split

"""
List variable as parameter as listed below, carefully change value as dependency
"""
path_ori_benign = 'original/benign/SOB'
path_ori_malignant = 'original/malignant/SOB'
path_binary_scenario = 'binary_scenario'
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

    list_sub_class_benign = split.query_result_split_binary(list_file_classes, path_ori_benign)
    list_sub_class_malignant = split.query_result_split_binary(list_file_classes, path_ori_malignant)
    dict_magnification_benign = split.build_dict_magnification_binary(list_sub_class_benign)
    dict_magnification_malignant = split.build_dict_magnification_binary(list_sub_class_malignant)

    # copying test image file
    already_test = []
    for _class in [dict_magnification_malignant, dict_magnification_benign]:
        for magnification in _class:
            len_test = split.test_len(_class[magnification], rate_test)
            rand = np.random.choice(len(_class[magnification]), len_test, False)
            for image in rand:
                if _class == dict_magnification_malignant:
                    shutil.copy(_class[magnification][image],
                                os.path.join(path_binary_scenario, 'test', magnification, 'malignant'))
                else:
                    shutil.copy(_class[magnification][image],
                                os.path.join(path_binary_scenario, 'test', magnification, 'benign'))
                already_test.append(_class[magnification][image])

    # copying validation image file
    dict_val_benign = {}
    dict_val_malignant = {}
    for magnification in list_magnification:
        dict_val_benign[magnification] = [x for x in dict_magnification_benign[magnification] if x not in already_test]
        dict_val_malignant[magnification] = [x for x in dict_magnification_malignant[magnification] if
                                             x not in already_test]
    already_val = []
    for _class in [dict_val_benign, dict_val_malignant]:
        for magnification in _class:
            len_input = split.val_len(_class[magnification], rate_val)
            rand = np.random.choice(len(_class[magnification]), len_input, False)
            for image in rand:
                if _class == dict_val_malignant:
                    shutil.copy(_class[magnification][image],
                                os.path.join(path_binary_scenario, 'val', magnification, 'malignant'))
                else:
                    shutil.copy(_class[magnification][image],
                                os.path.join(path_binary_scenario, 'val', magnification, 'benign'))
                already_val.append(_class[magnification][image])

    # copying training file
    dict_tr_benign = {}
    dict_tr_malignant = {}
    for magnification in list_magnification:
        dict_tr_benign[magnification] = [x for x in dict_val_benign[magnification] if x not in already_val]
        dict_tr_malignant[magnification] = [x for x in dict_val_malignant[magnification] if x not in already_val]
    already_train = []
    for _class in [dict_tr_benign, dict_tr_malignant]:
        for magnification in _class:
            for image in _class[magnification]:
                if _class == dict_tr_malignant:
                    shutil.copy(image, os.path.join(path_binary_scenario, 'train', magnification, 'malignant'))
                else:
                    shutil.copy(image, os.path.join(path_binary_scenario, 'train', magnification, 'benign'))
                already_train.append(image)
