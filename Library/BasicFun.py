import math
import os
import time

import numpy as np
import torch as tc
from matplotlib import pyplot as plt


def binary_strings(num):
    s = list()
    length = len(str(bin(num - 1))[2:])
    for n in range(num):
        b = str(bin(n))[2:]
        l0 = len(b)
        if length > l0:
            b = "".join([("0" * (length - l0)), b])
        s.append(b)
    return s


def combine_dicts(dic_def, dic_new, deep_copy=False):
    # dic_def中的重复key值将被dic_new覆盖
    import copy

    if dic_new is None:
        return dic_def
    if deep_copy:
        return dict(copy.deepcopy(dic_def), **copy.deepcopy(dic_new))
    else:
        return dict(dic_def, **dic_new)


def compare_dicts(dict1, dict2, name1="dict1", name2="dict2"):
    same = True
    for x in dict1:
        if x not in dict2:
            print(str(x) + ": in " + name1 + " but not in " + name2)
            same = False
        elif dict1[x] != dict2[x]:
            print(str(x) + ": value in " + name1 + " different from " + name2)
            same = False
    for x in dict2:
        if x not in dict1:
            print(str(x) + ": in " + name2 + " but not in " + name1)
            same = False
    return same


def convert_nums_to_abc(nums, n0=0):
    s = ""
    n0 = n0 + 97
    for m in nums:
        s += chr(m + n0)
    return s


def choose_device(n=0):
    if n == "cpu":
        return "cpu"
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:" + str(n))
            else:
                return tc.device("cuda" + str(n)[4:])
        else:
            return tc.device("cpu")


def empty_list(num, content=None):
    return [content] * num


def find_indexes_value_in_list(x, value):
    return [n for n, v in enumerate(x) if v == value]


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        if print_screen:
            print(content)
    else:
        if append:
            way = "ab"
        else:
            way = "wb"
        with open(file, way, buffering=0) as log:
            log.write((content + "\n").encode(encoding="utf-8"))
        if print_screen:
            print(content)


def indexes_eq2einsum_eq(indexes):
    eq = convert_nums_to_abc(indexes[0])
    for n in range(1, len(indexes) - 1):
        eq += "," + convert_nums_to_abc(indexes[n])
    eq += "->" + convert_nums_to_abc(indexes[-1])
    return eq


def join_path(path1, path2):
    if path1 is None:
        path_file = path2
    elif path2 is None:
        path_file = path1
    elif (path1 is None) and (path2 is None):
        path_file = "./"
    else:
        path_file = os.path.join(path1, path2)
    return path_file


def list_eq2einsum_eq(eq):
    # 将list表示的equation转化为einsum函数的equation
    # list中的数字不能超过25！！！
    # 例如[[0, 1], [0, 2], [1, 2]] 转为 'ab,ac->bc'
    # 例如[[0, 1], [0, 1], []] 转为 'ab,ab->'
    length = len(eq)
    eq_str = ""
    for n in range(length - 1):
        tmp = [chr(m + 97) for m in eq[n]]
        eq_str = eq_str + "".join(tmp) + ","
    eq_str = eq_str[:-1] + "->"
    tmp = [chr(m + 97) for m in eq[-1]]
    return eq_str + "".join(tmp)


def load(path_file, names=None, device="cpu", return_tuple=True):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file, map_location=device)
            if len(data) == 1:
                return list(data.values())[0]
            else:
                if return_tuple:
                    return tuple(data[x] for x in data)
                else:
                    return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                if names in tmp:
                    return tmp[names]
                else:
                    return None
            elif type(names) in [tuple, list]:
                nn = len(names)
                data = list(range(nn))
                for i in range(nn):
                    if names[i] in tmp:
                        data[i] = tmp[names[i]]
                return tuple(data)
            else:
                return None
    else:
        return None


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot(
    x,
    *y,
    marker="s",
    linestyle="-",
    markersize=None,
    markerfacecolor=None,
    markeredgewidth=None,
    xlabel=None,
    ylabel=None,
    legend=None,
):
    if type(x) is tc.Tensor:
        x = x.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        figs = list()
        if type(marker) is str:
            marker = [marker] * len(y)
        if type(linestyle) is str:
            linestyle = [linestyle] * len(y)
        for n, y0 in enumerate(y):
            if type(y0) is tc.Tensor:
                y0 = y0.cpu().numpy()
            (fig,) = ax.plot(
                x,
                y0,
                marker=marker[n],
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle[n],
            )
            fig.set_markerfacecolor(markerfacecolor)
            figs.append(fig)
    else:
        (figs,) = ax.plot(
            np.arange(len(x)),
            x,
            marker=marker,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linestyle=linestyle,
        )
        figs.set_markerfacecolor(markerfacecolor)
        figs = [figs]
    if legend is not None:
        plt.legend(figs, legend)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def print_dict(
    a, keys=None, welcome="", style_sep=": ", end="\n", file=None, print_screen=True, append=True
):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    fprint(express, file, print_screen, append)
    return express


def plot_multi_imgs(imgs, num_rows=1):
    # imgs.shape = (num_img, size0, size1)
    if type(imgs) is tc.Tensor:
        imgs = imgs.cpu().numpy()
    num = imgs.shape[0]
    fig = plt.figure()
    num_col = math.ceil(num / num_rows)
    for n in range(num):
        ax = fig.add_subplot(num_rows, num_col, n + 1)
        ax.imshow(imgs[n])
    plt.show()


def print_progress_bar(n_current, n_total, message=""):
    if n_current == (n_total - 1):
        message += "\t" + chr(9646) * 10 + "done! \n"
    else:
        x1 = math.floor(n_current / n_total * 10)
        x2 = math.floor(n_current / n_total * 100) % 10
        message += "\t" + chr(9646) * x1 + str(x2) + chr(9647) * (9 - x1)
    print("\r" + message, end="")
    time.sleep(0.01)


def print_mat(mat):
    if type(mat) is tc.Tensor:
        mat = mat.numpy()
    for x in mat:
        print(list(x))


def remove_list1_from_list0(list0, list1):
    for x in list1:
        list0.remove(x)
    return list0


def replace_value(x, value0, value_new):
    x_ = np.array(x)
    x_[x_ == value0] = value_new
    return list(x_)


def save(path, file, data, names, append=False):
    if path is None:
        path_file = file
    elif file is None:
        path_file = path
    else:
        mkdir(path)
        path_file = os.path.join(path, file)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    if append and os.path.isfile(path_file):
        data0 = load(path_file, return_tuple=False)
        tmp = dict(data0, **tmp)
    tc.save(tmp, path_file)


def search_file(path, exp):
    import re

    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result


def show_multiple_images(
    imgs, lxy=None, titles=None, save_name=None, show=True, cmap="coolwarm", img_size=None
):
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["Songti SC"]

    if cmap is None:
        cmap = plt.cm.gray
    if type(imgs) is tc.Tensor:
        imgs = imgs.cpu().numpy()
    if (type(imgs) is np.ndarray) and (imgs.ndim == 2):
        plt.imshow(imgs, cmap=cmap)
        plt.show()
        return None
    ni = len(imgs)
    if lxy is None:
        lx = int(np.sqrt(ni))
        ly = int((ni + 1) / lx)
    else:
        lx, ly = tuple(lxy)
    if lx == -1:
        lx = int((ni + 1) / ly)
    elif ly == -1:
        ly = int((ni + 1) / lx)

    for n in range(ni):
        plt.subplot(lx, ly, n + 1)
        if type(imgs[n]) is tc.Tensor:
            tmp = imgs[n].cpu().numpy()
        else:
            tmp = imgs[n]
        if img_size is not None:
            tmp = tmp.reshape(img_size)
        if tmp.ndim == 2:
            plt.imshow(tmp, cmap=cmap)
        else:
            plt.imshow(tmp)
        if titles is not None:
            plt.title(str(titles[n]))
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
    # plt.colorbar()
    if type(save_name) is str:
        plt.savefig(save_name)
    if show:
        plt.show()


def sort_list(a, order):
    return [a[i] for i in order]


def supplementary(set1, set2):
    if type(set1) is tc.Tensor:
        set1 = set1.cpu().tolist()
    if type(set2) is tc.Tensor:
        set2 = set2.cpu().tolist()
    return set(set1) - set(set2)


# -------------------------------------
# From ZZS
def compare_iterables(a_list, b_list):
    from collections.abc import Iterable

    if isinstance(a_list, Iterable) and isinstance(b_list, Iterable):
        xx = [x for x in a_list if x in b_list]
        if len(xx) > 0:
            return True
        else:
            return False
    else:
        return False


def inverse_permutation(perm):
    # perm is a torch tensor
    if not isinstance(perm, tc.Tensor):
        perm = tc.tensor(perm)
    inv = tc.empty_like(perm)
    inv[perm] = tc.arange(perm.size(0), device=perm.device)
    return inv.tolist()
