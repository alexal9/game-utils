import pyautogui as p
import time
import cv2
import keyboard
import pytesseract
import sys

def clamp(val, a, b):
	return max(a, min(val, b))

class Farmer:
	VICTORY = 0
	DEFEAT = 0
	RUNS = 0

	WIDTH = 1152 * 2
	HEIGHT = 683 * 2

	def press_button(self, button, sleep_duration = 1, confidence = 0.6):
		time.sleep(sleep_duration)
		img = p.screenshot('temp.png')
		obj = p.locate(button + '.png', img, confidence = confidence)
		if obj:
			l,t,w,h = obj
			p.click(l/2 + w//4, t/2 + h//4, clicks = 1)
			return True
		else:
			return False

	def farm(self, timer):
		# we expect the run to take a certain amount of time on average (add 5 for loading and post-processing)
		Farmer.RUNS += 1
		time.sleep(timer)
		while True:
			# process again every 5 seconds if still battling
			time.sleep(5)
			# get bounding box of application
			img = p.screenshot('temp.png')
			l,t,w,h = p.locate('exit.png', img, confidence = 0.95)
			img = cv2.imread('temp.png')
			# processing is img[y, x]
			img_victory = img[t+72+125:t+440, l:l+Farmer.WIDTH]
			img_duration = img[t+72:t+72+75, l+Farmer.WIDTH-298:l+Farmer.WIDTH]
			cv2.imwrite('duration.png', img_duration)
			# try to look for victory text
			data = pytesseract.image_to_string(img_victory)
			# try to find revive text
			dead = p.locate('revive.png', 'temp.png', confidence = 0.6)
			if 'VICTORY' in data:
				if not self.victory():
					return False
				return True
			elif dead:
				self.defeat()
				return True

	def victory(self):
		Farmer.VICTORY += 1
		print('victory')
		# get dungeon run time
		l,t,w,h = p.locate('exit.png', 'temp.png', confidence = 0.95)
		img = cv2.imread('temp.png')
		# get reward
		p.click((2*l + Farmer.WIDTH)//4, (2*t + Farmer.HEIGHT)//4)
		time.sleep(0.5)
		p.click((2*l + Farmer.WIDTH)//4, (2*t + Farmer.HEIGHT)//4)
		# press ok
		self.press_button('ok')
		# pressing ok on rune drop popup results in double click somehow
		# check to see if double click (ok -> prepare)

		##### for event, see if event item dropped
		self.press_button('ok')

		found = self.press_button('start')
		# otherwise pressed ok on item drop popup
		if not found:
			found = self.press_button('replay', 0)

		# could not press start or replay, that means we are out of energy
		# if we did not collect energy then try again
		if not found:
			while True:
				print('trying to collect energy')
				self.press_button('giftbox', 0)
				found = self.press_button('collect')
				# nothing to collect
				if not found:
					return False

				self.press_button('close', 0, confidence = 0.8)
				found = self.press_button('replay')
				break

		# no more energy, stop farming
		return found

	def defeat(self):
		print('defeat')
		Farmer.DEFEAT += 1
		self.press_button('no')
		l,t,w,h = p.locate('exit.png', 'temp.png', confidence = 0.95)
		p.click((2*l + Farmer.WIDTH)//4, (2*t + Farmer.HEIGHT)//4)
		time.sleep(0.5)
		self.press_button('prepare')
		self.press_button('start')

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Specify the average time in seconds of the dungeon run')
	else:
		f = Farmer()
		f.press_button('play', confidence = 0.9)
		while True:			
			result = f.farm(int(sys.argv[-1]))
			print('Runs/victory/defeat {}/{}/{}'.format(f.RUNS, f.VICTORY, f.DEFEAT))
			if not result:
				break

