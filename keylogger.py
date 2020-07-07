import keyboard
import pyautogui as p
import time

def on_press(key):
	if key.name == 's':
		p.screenshot(str(time.time()) + ".png")

keyboard.on_press(on_press)

while True:
	if keyboard.is_pressed('q'):
		break