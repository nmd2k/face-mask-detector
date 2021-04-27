import os
from tqdm import tqdm

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_yolo(files, is_train=False):
    pbar = tqdm(range(len(files)))
    for i in pbar:
        if (files[i][-3:] == 'txt'):
            if is_train:
                os.rename(PATH+'train/'+files[i], PATH+'labels/train/'+files[i])
            else:
                os.rename(PATH+'val/'+files[i], PATH+'labels/test/'+files[i])
        
        else:
            if is_train:
                os.rename(PATH+'train/'+files[i], PATH+'images/train/'+files[i])
            else:
                os.rename(PATH+'val/'+files[i], PATH+'images/test/'+files[i])

# path
PATH = 'data/'
TRAIN_PATH = 'data/train/'
VALID_PATH = 'data/val/'
images_dir = PATH+'images/'
labels_dir = PATH+'labels/'
train_image_dir = images_dir+'train/'
test_image_dir = images_dir+'test/'
train_label_dir = labels_dir+'train/'
test_label_dir = labels_dir+'test/'

create = [images_dir, labels_dir, train_image_dir, test_image_dir, train_label_dir, test_label_dir]
for dir in create:
    create_dir(dir)

# get list file
train_file = list(os.listdir(TRAIN_PATH))
valid_file = list(os.listdir(VALID_PATH))


convert_to_yolo(train_file, is_train=True)
convert_to_yolo(valid_file, is_train=False)