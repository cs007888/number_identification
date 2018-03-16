import random
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NUM = [chr(i) for i in range(0x30, 0x39)]
UPPER = [chr(i) for i in range(0x41, 0x5a)]
LOWER = [chr(i) for i in range(0x61, 0x7a)]
LETTER = (NUM + UPPER + LOWER)*2

PIC_SIZE = 120, 40 # width and high
BACKGROUND_COLOR = 200, 256, 200 # green
FONT = 'C:/Windows/Fonts/Calibri.ttf'
FONT_SIZE = 24
FONT_COLOR = 0, 0, 0 # black

PIC_PATH = 'home/'
SUFFIX = '.png'
SAMPLES = 1

def get_text():
    char_count = random.randint(4,6)
    return random.sample(LETTER, char_count)

def gene_line(draw, size):
    width, height = size
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill = (0, 0, 0))

def draw(img_size):
    image = Image.new('RGB', img_size, BACKGROUND_COLOR)
    font = ImageFont.truetype(FONT, FONT_SIZE)
    draw = ImageDraw.Draw(image)
    text = get_text()
    label = ''.join(text)
    for i in range(len(text)):
        draw.text((i*16, random.randint(0, 12)), text[i], font=font, fill=FONT_COLOR)
        gene_line(draw, img_size)
    return image, label

def get_index():
    names = []
    for _, _, files in os.walk(PIC_PATH):
        for filename in files:
            name, _ = os.path.splitext(filename)
            names.append(int(name))
    if len(names) == 0:
        return -1
    else:
        sorted(names)
        return(names[-1])

def write_label(filename, label):
    label_file = open('label.txt', 'w')
    label_file.write

# def get_verify_pic():
#     if not os.path.isdir(PIC_PATH):
#         os.makedirs(PIC_PATH)
#     index = get_index() + 1
#     return draw()
#     label_file = open('label.txt', 'a')
#     for i in range(index, SAMPLES + index):
#         image, label = draw()
#         pic_name = str(i) + SUFFIX
#         print(np.float32(image).shape)
#         image.save(PIC_PATH + pic_name, 'png')
#         label_file.write(pic_name + ':' + label + '\n')
#     label_file.close()


# if __name__ == '__main__':
#     get_verify_pic()