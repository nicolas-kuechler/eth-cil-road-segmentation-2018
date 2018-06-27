from PIL import Image
import os


dir = './../validation_extension/images/'
count = 0

for filename in os.listdir(dir):
    img = Image.open(dir + '/' + filename)
    if img.mode == 'RGBA':
        img.convert('RGB').save(dir + '/' + filename)
        count += 1

print(f'Converted {count} Images from RGBA to RGB')
