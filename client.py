import cv2
import io
import socket
import struct
import pickle
from PIL import Image

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.26', 8888))
connection = client_socket.makefile('wb')

img = Image.open("/Users/alexanderlee/Desktop/silver_star.png")
# img = cv2.imread("/Users/alexanderlee/Desktop/silver_star.png")
# data = pickle.dumps(img, 0)
array = io.BytesIO()
img.save(array, format = img.format)
data = array.getvalue()
size = len(data)

print(size)
client_socket.sendall(struct.pack(">L", size) + data)
