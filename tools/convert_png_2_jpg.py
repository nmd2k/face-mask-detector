import os
from PIL import Image

PATH = 'FaceMaskDataset/val/'

file = list({i[:-3] for i in os.listdir(PATH) if i[-3:] == 'png'})

print(file)

for image in file:
    img = Image.open(PATH+image+'png')
    img.save(PATH+image+'jpg')
    os.remove(PATH+image+'png')
