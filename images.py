
from __future__ import print_function
import os

from PIL import Image , ImageChops



for index in range(29):
  path = [f for f in os.listdir('cifar_LA_images_8_eps') if 'index'+str(index)+ 'orig' in f or 'index'+str(index)+ 'adv' in f] 
  result = Image.new("RGB", (630, 200),color=(255,255,255,0))
  orig_class = ''
  adv_class = ''
  print(path)
  if 'adv' in path[0]:
    img1 = Image.open('cifar_LA_images_8_eps/' + path[1])
    img1.thumbnail((200, 200), Image.ANTIALIAS)
    img2 = Image.open('cifar_LA_images_8_eps/' + path[0])
    img2.thumbnail((200, 200), Image.ANTIALIAS)
    orig_class = path[1].split('orig')[1].split('.png')[0]
    adv_class = path[0].split('adv')[1].split('.png')[0]
  elif 'orig' in path[0]:
    img1 = Image.open('cifar_LA_images_8_eps/' + path[0])
    img1.thumbnail((200, 200), Image.ANTIALIAS)
    img2 = Image.open('cifar_LA_images_8_eps/' + path[1])
    img2.thumbnail((200, 200), Image.ANTIALIAS)
    orig_class = path[0].split('orig')[1].split('.png')[0]
    adv_class = path[1].split('adv')[1].split('.png')[0]
  else:
    continue
  img3 = ImageChops.subtract(img1,img2)
  d1 = img1.getdata()
  d2 = img2.getdata()
  # print(d1.shape)
  img3.thumbnail((200, 200), Image.ANTIALIAS)
  x = 0
  y = 0
  w, h = img1.size
  w1, h1 = img2.size
  w2, h2 = img3.size
  print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
  print('pos {0},{1} size {2},{3}'.format(x, y, w1, h1))
  print('pos {0},{1} size {2},{3}'.format(x, y, w2, h2))
  result.paste(img1, (x, y, x + w, y + h))
  result.paste(img2, (x+210, y, x+ 210 + w1, y  + h1))
  result.paste(img3, (x+420, y, x+ 420 + w2, y  + h2))
  result.save('cifar_LA_adversarial_examples/' + orig_class + '_' + adv_class + '.jpg')