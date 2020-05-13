from random import shuffle
import glob
import os
import sys

train_folder = "/media/wooram/DATA/00.DL_datasets/ImageNet/2012/Images/train"
val_folder = "/media/wooram/DATA/00.DL_datasets/ImageNet/2012/Images/val"
dataset_root = "/media/wooram/DATA/00.DL_datasets/ImageNet/2012/Images"


def gen_list(set_folder, root_folder, setname):
    img_subfolders = sorted(os.listdir(set_folder))
    img_list = []
    for i, subfolder in enumerate(img_subfolders):
        sub_list = glob.glob(os.path.join(set_folder, subfolder, "*.*"))
        for files in sub_list:
            img_list.append([i, subfolder, files])
    shuffle(img_list)
    labels = []
    filenames = []
    for label, subfolder, filename in img_list:
        labels.append(str(label) + "    " + subfolder)
        filenames.append(filename)
    with open(os.path.join(root_folder, "info", "%s.txt" % setname), "w") as txt:
        txt.write("\n".join(filenames))
        txt.close()
    with open(os.path.join(root_folder, "info", "%s_labels.txt" % setname), "w") as txt:
        txt.write("\n".join(labels))
        txt.close()
gen_list(train_folder, dataset_root, "train")
gen_list(val_folder, dataset_root, "val")
