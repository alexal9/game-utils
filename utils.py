import pyautogui as p
import cv2
import pytesseract
from PIL import Image
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def clamp(val, a, b):
    return max(a, min(val, b))

def click(obj):
    if obj:
        l,t,w,h = obj
        # center is l + w/2 and assume 3 std with size w/6
        x = random.normalvariate(l + w/2, w/6)
        y = random.normalvariate(t + h/2, h/6)
        return ( clamp(int(x), l, l+w), clamp(int(y), t, t+h) )

# revive, ok, start, replay, giftbox, collect, close, no, prepare, start, play, ...
def find(button, source, confidence = 0.6, show = False):
    """
    button: button name, ex: "play"
    source: PIL Image object
    confidence: match confidence
    show: show plot

    return bytearray for server to send to client
    """
    # img = Image.open(source)
    obj = p.locate(button + ".png", source, confidence = confidence)
    if obj:
        l,t,w,h = obj
        x,y = click(obj)
        
        if show:
            plt.imshow(source)
            # show button location
            plt.gca().add_patch(Rectangle((l,t),w,h,linewidth=1,edgecolor='r',facecolor='none'))
            # show click location
            plt.gca().add_patch(Rectangle((x-5,y-5),11,11,linewidth=1,edgecolor='r',facecolor='none'))
            plt.show()
        
        return bytearray( ','.join(str(i) for i in ('click',x,y)), 'utf-8' )

def get_status(source):
    # PIL.Image.size gives (w,h), np shape gives (h,w,4)
    # get top third of image to check for victory or defeat
    header = np.asarray(source)[:source.size[1]//3,:,:]
    data = pytesseract.image_to_string(header)

    print(data)
    if 'Reward' in data:
        return b'reward'
    elif 'VICTORY' in data:
        return b'victory'
    elif 'DEFEATED' in data:
        return b'defeated'
    else:
        return b'running'

def victory(source):
    # click, click, ok, ok (event), replay
