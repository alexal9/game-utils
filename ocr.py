# import the necessary packages
import pytesseract
import cv2
import numpy as np
from collections import OrderedDict

class Rune:
	qualityDict = {"Normal": 1, "Magic": 2, "Rare": 3, "Hero": 4, "Legend": 5}
	fileHeader = "/Users/alexanderlee/Desktop/"

	def __init__(self, imgPath):
		self._createRune(imgPath)

	def _createRune(self, imgPath):
		img1 = cv2.imread(imgPath, 0) # gray
		img2 = cv2.pyrDown(img1) # subsample
		result1 = self.parseResult(img1)
		result2 = self.parseResult(img2)

		if len(result2) > len(result1):
			result1, result2 = result2, result1
		self._parseRune(result1)
		if not self.isValidRune():
			self._parseRune(result2, True)

		print("Rune created")
		print("------------")
		if self.qualifier == "":
			print(self.set, 'Rune', '(' + str(self.slot) + ')')
		else:
			print(self.qualifier, self.set, 'Rune', '(' + str(self.slot) + ')')
		print(self.quality)
		for k,v in self.subs.items():
			print(k, v)

	def _parseRune(self, data, update = False):
		if update:
			qualifier = self.qualifier
			runeSet = self.set
			quality = self.quality
			slot = self.slot
			subs = self.subs
		else:
			qualifier = ''
			runeSet = ''
			quality = ''
			slot = 0
			subs = OrderedDict()

		for line in data:
			if line.strip() == '':
				continue

			words = line.split()
			# [qualifier] <set> Rune (<slot>)
			if "Rune (" in line:
				if len(words) == 4:
					# innate rune
					qualifier = words[0]
					runeSet = words[1]
				else:
					qualifier = ''
					runeSet = words[0]
				slot = int(words[-1][1])
			elif words[0] in Rune.qualityDict:
				quality = words[0]
			# <stat> <value>[%]				
			elif self.isStatText(' '.join(words[:-1])):
				stat = ' '.join(words[:-1])
				value = words[-1]
				# print(stat, value)
				if stat in ['ATK','DEF','HP']:
					if value[-1] == "%":
						subs[stat+"%"] = value[1:-1]
					else:
						subs[stat+"+"] = value[1:]
				elif stat == 'SPD':
					subs[stat] = value[1:]
				else:
					subs[stat] = value[1:-1]

		if update and len(subs) != Rune.qualityDict[quality] + (1 if qualifier != "" else 0):
			print("Rune._parseRune() - Substat count mismatch, cannot process rune")
			return

		self.qualifier = qualifier
		self.set = runeSet
		self.quality = quality
		self.slot = slot
		self.subs = subs

	def isValidRune(self):
		return self.set != '' and self.quality != '' and 1 <= self.slot <= 6 and len(self.subs) == Rune.qualityDict[self.quality] + (1 if self.qualifier != "" else 0)

	def isStatText(self, text):
		return text in ['ATK','DEF','HP','SPD','Accuracy','Resistance','CRI Rate','CRI Dmg']

	def validStartingChar(self, char):
		# eg: (6) for rune slot, +4% for sub value, 4 Set, Stat
	 	return char in ['(', '+'] or char.isdigit() or char.isupper()

	def parseResult(self, img):
		data = pytesseract.image_to_data(img)
		lines = data.split('\n')
		numCols = len(lines[0].split('\t'))
		lineBreak = True
		result = []
		temp = ''
		for line in lines[1:]:
			cols = line.split('\t')
			if len(cols) == numCols and int(cols[-2]) > 60 and self.validStartingChar(cols[-1][0]):
				if cols[-1] in Rune.qualityDict:
					result.append(temp.strip())
					result.append('')
					result.append(cols[-1])
					temp = ''
				else:
					temp += cols[-1] + ' '
					lineBreak = True
			elif lineBreak:
				result.append(temp.strip())
				result.append('')
				temp = ''
				lineBreak = False
		print(result)
		return result

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	# help="type of preprocessing to be done")
# args = vars(ap.parse_args())

# load the example image and convert it to grayscale
# image = cv2.imread(args["image"])

# image = cv2.imread("/Users/alexanderlee/Desktop/IMG_0488_edit.png")
# imgPath = "/Users/alexanderlee/Desktop/IMG_2355 copy.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_2355.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_0488_edit.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_0488 copy.png"

# gray = cv2.imread(imgPath, 0)
# gray = cv2.imread(args["image"], 0)
# test = cv2.pyrDown(gray)
# test = gray

# test = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Test", test)
# test2 = cv2.medianBlur(test, 3)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# test2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# test2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
# test2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
# cv2.imshow("Test2", test2)
 
# check to see if we should apply thresholding to preprocess the
# image
# if args["preprocess"] == "thresh":
 
# make a check to see if median blurring should be done to remove
# noise
# elif args["preprocess"] == "blur":
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

"""
#--- dilation on the green channel ---
dilated_img = cv2.dilate(image[:,:,1], np.ones((5, 5), np.uint8))
# dilated_img = cv2.dilate(image, np.ones((5, 5), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 3)

#--- finding absolute difference to preserve edges ---
diff_img = 255 - cv2.absdiff(image[:,:,1], bg_img)
# diff_img = 255 - cv2.absdiff(image, bg_img)

#--- normalizing between 0 to 255 ---
norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('norm_img', cv2.resize(norm_img, (0, 0), fx = 0.5, fy = 0.5))

#--- Otsu threshold ---
th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('th', cv2.resize(th, (0, 0), fx = 0.5, fy = 0.5))
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
cv2.imwrite(filename, th)
"""

# dilated_img = cv2.dilate(gray[:,:,1], np.ones((3, 3), np.uint8))
# bg_img = cv2.medianBlur(dilated_img, 3)
# diff_img = 255 - cv2.absdiff(gray[:,:,1], dilated_img)
# norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# cv2.imshow('norm_img', cv2.resize(norm_img, (0, 0), fx = 0.5, fy = 0.5))
# th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('th', cv2.resize(th, (0, 0), fx = 0.5, fy = 0.5))

# print(test.shape, test2.shape)
# comb = np.mean(np.array([test, test2]), axis = 0)
# print(comb.shape)
# gray = norm_img
# gray = test2
# gray = test

# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file

# print(pytesseract.image_to_string(gray, lang = 'eng', config = "--psm 1"))
# print('--------')
# print(pytesseract.image_to_string(gray))
# print(pytesseract.image_to_string(test2))
# print('--------')
# print(pytesseract.image_to_string(gray))

# print(pytesseract.image_to_boxes(Image.open(filename)))


# data = pytesseract.image_to_data(gray)
# print(data)
# count, result = parseResult(data)

# maxCount = 0
# finalResult = []
# for preprocess in [gray, test]:
# 	data = pytesseract.image_to_data(preprocess)
# 	print(data)
# 	print('--------')
# 	result = parseResult(data)
# 	if count > maxCount:
# 		maxCount = count
# 		finalResult = result

# print(finalResult)

# print(*map(lambda img: parseResult(pytesseract.image_to_data(img)), (gray, test)))
# result = mergeResult(*map(lambda img: parseResult(pytesseract.image_to_data(img)), (gray, test)))
# print('\n'.join(i for i in result if i.strip() != ""))

# print(data)
# print(data["conf", "text"])

# os.remove(filename)
 
# show the output images
# cv2.imshow("Image", norm_img)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

