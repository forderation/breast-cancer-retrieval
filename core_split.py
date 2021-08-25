import os

list_dir_def = lambda base_dir, dir_name: [x for x in os.listdir(os.path.join(base_dir, dir_name))]


def find_list_dir(subclasses_benign, _subclass_malignant, _path_ori_benign, _path_ori_malignant):
    """
    find list dir available prefix on SOB
    :param _path_ori_malignant:
    :param _path_ori_benign:
    :param subclasses_benign:
    :param _subclass_malignant:
    :return: dictionary with keys subclasses of labels
    """
    _list_dir_result = {}
    for _classes in subclasses_benign:
        if _classes not in _list_dir_result: _list_dir_result[_classes] = {}
        _list_dir_result[_classes]['list'] = list_dir_def(_path_ori_benign, _classes)
        _list_dir_result[_classes]['path'] = _path_ori_benign
    for _classes in _subclass_malignant:
        if _classes not in _list_dir_result: _list_dir_result[_classes] = {}
        _list_dir_result[_classes]['list'] = list_dir_def(_path_ori_malignant, _classes)
        _list_dir_result[_classes]['path'] = _path_ori_malignant
    return _list_dir_result


def file_splitting(list_dir_subclass, path_ori_class, magnifications, dir_name_subclass):
    """
    Find list image
    :param list_dir_subclass:
    :param path_ori_class:
    :param magnifications:
    :param dir_name_subclass:
    :return:
    """
    _list_file_ori_dir = []
    for _subclass_dir in list_dir_subclass:
        for _x in magnifications:
            _dir_path = os.path.join(path_ori_class, dir_name_subclass, _subclass_dir, _x)
            for (_dir_path, dir_names, filenames) in os.walk(_dir_path):
                for _file in filenames:
                    _list_file_ori_dir.append(os.path.join(_dir_path, _file))
    return _list_file_ori_dir


def file_splitting_subclass(_list_dir_subclasses, _subclasses, _list_magnification):
    """
    Wrapper function of file_splitting
    :param _list_magnification:
    :param _subclasses:
    :param _list_dir_subclasses:
    :return: dictionary
    """
    for _class in _subclasses:
        _list_dir_subclasses[_class]['split'] = file_splitting(
            _list_dir_subclasses[_class]['list'],
            _list_dir_subclasses[_class]['path'],
            _list_magnification,
            _class
        )
    return _list_dir_subclasses


test_len = lambda len_x, rate_test: int((rate_test / 100) * len(len_x))

val_len = lambda len_x, rate_val: int((rate_val / 100) * len(len_x))


def get_info_total(_list_file_classes, _subclass):
    """
    Print information images on all subclasses
    :param _subclass:
    :param _list_file_classes:
    :return: void
    """
    _total_all = 0
    for _class in _subclass:
        _total = _list_file_classes[_class]['split']
        print("{} class have total images {}".format(_class, _total))
        _total_all += _total
    print("total images all class {}".format(_total_all))


def build_dict_magnification_subclasses(_list_subclasses):
    """
    builder dictionary on multi subclass to be used in splitting file
    :param _list_subclasses:
    :return: dictionary
    """
    _dict_magnification_subclasses = {}
    for _sub_class in _list_subclasses:
        for _file in _sub_class:
            _lvl = _file.split("\\")[3]
            _scl = _file.split("/")[2].split("\\")[1]
            if _lvl not in _dict_magnification_subclasses:
                _dict_magnification_subclasses[_lvl] = {}
            if _scl not in _dict_magnification_subclasses[_lvl]:
                _dict_magnification_subclasses[_lvl][_scl] = []
            _dict_magnification_subclasses[_lvl][_scl].append(_file)
    return _dict_magnification_subclasses


def build_dict_magnification_binary(_list_subclass):
    """
    builder dictionary on binary class to be used in splitting file
    :param _list_subclass:
    :return:
    """
    _dict_magnification = {}
    for _sub_class in _list_subclass:
        for _file in _sub_class:
            _lvl = _file.split("/")[5]
            if _lvl not in _dict_magnification:
                _dict_magnification[_lvl] = [_file]
            else:
                _dict_magnification[_lvl].append(_file)


def query_result_split_binary(_list_file_classes, _path):
    """
    query to find where is as binary class
    :param _list_file_classes:
    :param _path:
    :return:
    """
    _query_result = []
    for _classes in _list_file_classes:
        if _classes['path'] == _path:
            _query_result.append(_classes['split'])
    return _query_result
