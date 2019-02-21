from pymouse import PyMouse
from pykeyboard import PyKeyboard
from PIL import ImageGrab

import time

m = PyMouse()
k = PyKeyboard()
#m.click(370, 1065, 1)
time.sleep(0.5)
#m.click(371, 1065, 1)
pointx = 960
pointy = 540
size =int(500 / 2)
box = (pointx- size,pointy -size,pointx + size,pointy + size)
path = r'D:\RockGan\3DGANServer\tool\top/'
# for i in range(1000,1001):
#     number = i
#     m.click(22, 55, 1)
#     time.sleep(0.5)
#     m.click(60, 145, 1)
#
#     time.sleep(0.5)
#     # m.click(1120, 540, 1)
#     # time.sleep(0.5)
#     m.click(600, 300, 1)
#     time.sleep(1)
#     k.type_string(str(number) + ".obj")
#     time.sleep(0.5)
#     k.tap_key(k.enter_key)
#     time.sleep(1.5)
#     m.click(1230, 560, 1)
#     m.click(1231, 560, 1)
#     time.sleep(0.5)
#     m.click(100, 66, 1)
#     time.sleep(1)
#     # m.click(1800, 295, 1)
#     # m.click(1801, 295, 1)
#     # time.sleep(0.5)
#     #
#     #
#     # k.tap_key(k.right_key)
#     # time.sleep(1)
#     # k.type_string("90")
#     # time.sleep(0.5)
#     # k.tap_key(k.enter_key)
#     # time.sleep(0.5)
#     # k.tap_key(k.down_key)
#     # time.sleep(1)
#     # k.type_string("150")
#     # time.sleep(0.5)
#     # k.tap_key(k.enter_key)
#     # time.sleep(0.5)
#     #
#     # m.click(1800, 342, 1)
#     # m.click(1801, 342, 1)
#     # time.sleep(0.5)
#     im = ImageGrab.grab(box)
#     pf = path + str(number) + '.png'
#     im.save(pf ,'png')
#     time.sleep(0.5)

# for i in range(201,300):
#     number = i
#     time.sleep(0.1)
#     m.click(460, 150, 1)
#     time.sleep(0.5)
#     m.click(455, 155, 1)
#     time.sleep(0.5)
#     m.click(500, 240, 1)
#     time.sleep(1)
#     m.click(700, 330, 1)
#     time.sleep(0.5)
#     k.type_string(str(number) + ".obj")
#     time.sleep(0.5)
#     m.click(1200, 600, 1)
#     time.sleep(1.4)
#     m.click(1300, 600, 1)
#     time.sleep(1)
#
#     im = ImageGrab.grab(box)
#     pf = path + str(number) + '.png'
#     im.save(pf ,'png')
#m.click(160, 160, 1) #remove
#
# x_dim, y_dim = m.screen_size()
# print(x_dim,y_dim)
# localx = 270
# localy = 133
#
m.click(570, 1065, 1)



# for i in range(901,1000):
#     number = i
#     time.sleep(0.5)
#     m.click(1835, 1002, 1) #new
#     #time.sleep(0.2)
#     #m.click(1823,467,1)#open
#     time.sleep(1)
#     k.type_string(str(number) + ".obj")
#     time.sleep(1)
#     m.click(1750, 1000,1) #close
#     time.sleep(0.6)
#     m.click(1835, 100,1)  #save
#     time.sleep(0.5)
#     k.type_string(str(number)+".vox")
#     time.sleep(0.5)
#     m.click(1750, 1000, 1)
#     time.sleep(0.5)

for i in range(0,201):
    time.sleep(5)
    m.click(1678, 852, 1)
    print(i)