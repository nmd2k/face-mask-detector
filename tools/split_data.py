import os
import random
import secrets
from shutil import copyfile

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

PATH = 'data/images/train'
random.seed(42)

files = list(os.listdir(PATH))
val_files = []

train_len = len(files)
valid_len = round(train_len*0.2)
train_len = train_len - valid_len

print("Train length:", train_len)
print("Valid length:", valid_len)

create_dir('data/images/valid')
create_dir('data/labels/valid')

i=0
while i < valid_len:
    file = secrets.choice(files)
    if file not in val_files:
        val_files.append(file)
    else: i-=1

    i+=1

for file in val_files:
    copyfile('data/images/train/'+file, 'data/images/valid/'+file)
    copyfile('data/labels/train/'+file[:-3]+'txt', 'data/labels/valid/'+file[:-3]+'txt')
    os.remove('data/images/train/'+file)
    os.remove('data/labels/train/'+file[:-3]+'txt')

print('Done')