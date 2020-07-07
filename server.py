import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
from PIL import Image
import io
import matplotlib.pyplot as plt
import signal

# utility module
import utils

HOST='192.168.1.26'
PORT=18888

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

def keyboardInterruptHandler(signal, frame):
    print("\nKeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    print("Closing socket")
    s.close()
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

try:
    while True:
        conn,addr=s.accept()
        print("Got connection from", addr)

        # receiving image data 
        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        # while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            chunk = conn.recv(4096)
            if not chunk:
                raise RuntimeError("socket connection broken")
            data += chunk

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            chunk = conn.recv(4096)
            if not chunk:
                raise RuntimeError("socket connection broken")
            data += chunk
        frame_data = data[:msg_size]
        data = data[msg_size:]
        image = Image.open(io.BytesIO(frame_data))

        # process image and send result to client
        # result = utils.find("ok", image)
        result = utils.process(image)
        # msg = struct.pack(">I", ''.join([chr(x) for x in result]))
        print(result)
        # msg = bytearray(','.join(str(i) for i in result), 'utf-8')
        conn.send(result)
        conn.close()    
except Exception as e:
    print("error occured", e)
    s.close()


