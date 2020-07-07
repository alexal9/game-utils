import cv2
import numpy as np
import pytesseract

# imgPath = "/Users/alexanderlee/Desktop/IMG_0488_edit.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_2355 copy.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_2355.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_0488.png"
## imgPath = "/Users/alexanderlee/Desktop/IMG_0200.png"
imgPath = "/Users/alexanderlee/Desktop/IMG_2408.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_2425.png"
# imgPath = "/Users/alexanderlee/Desktop/IMG_2415.png"

## working images 2411, 2410/F, 2425, 2420/F, 2411, 2410/F, 2409/F 
## failed 2424, 2423, 2419, 2418,
qualityDict = {"Normal": 1, "Magic": 2, "Rare": 3, "Hero": 4, "Legend": 5}

def validStartingChar(char, isRune):
	# eg: (6) for rune slot, +4% for sub value, 4 Set, Stat
 	return char in ['(', '+'] or (char.isupper() if isRune else char.isalnum())

def stripInvalid(s, isRune):
	i = 0
	while i < len(s):
		if char.isupper() if isRune else char.isalnum():
			return s[i:]
		i += 1
	return ''

def isStatText(text):
	return ' '.join(text.split()[:-1]) in ['ATK','DEF','HP','SPD','Accuracy','Resistance','CRI Rate','CRI Dmg']

def parseResult(data, y, isRune):
	print(data, isRune)
	lines = data.split('\n')
	numCols = len(lines[0].split('\t'))
	temp = []
	for line in lines[1:]:
		cols = line.split('\t')
		if len(cols) == numCols and int(cols[-2]) > 65 and validStartingChar(cols[-1][0], isRune):
			temp.append(cols[-1])
	result = ' '.join(temp)
	header = ''
	if isStatText(result):
		# format substat to be easily parsable 
		if temp[-1][-1] == '%':
			if temp[0] in ['ATK','DEF','HP']:
				temp[0] += '%'
			temp[-1] = temp[-1][1:-1]
		else:
			if temp[0] in ['ATK','DEF','HP']:
				temp[0] += '+'
			temp[-1] = temp[-1][1:]
		result = ' '.join(temp)
		header = '(M) ' if y < 100 else ('(I) ' if 100 <= y <= 140 else '(S) ')
	return header + result

def isSameColor(color1, color2, threshold = 5):
	return all(abs(color1[i] - color2[i] < threshold) for i in range(len(color1)))

def countStars(imgPath, threshold = 0.97, showImage = False):
	# img = '/Users/alexanderlee/Desktop/IMG_0488.png'
	img_rgb = cv2.imread(imgPath)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('/Users/alexanderlee/Desktop/silver_star.png', 0)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)
	threshold = 0.97
	# threshold = 0.94
	loc = np.where( res >= threshold)
	count = 0
	# boxes = non_max_suppression_fast([i for i in zip(*loc[::-1])], 0.9)

	boxes = non_max_suppression_fast(np.array([ (x,y,x+w,y+h) for x,y in zip(*loc[::-1]) ]), 0.3)
	# for pt in zip(*loc[::-1]):
	for pt in boxes:
	    # print("match at", pt)
	    count += 1
	    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
	    cv2.rectangle(img_rgb, (pt[0], pt[1]), (pt[2], pt[3]), (0,0,255), 2)

	# print("number of matches", count)
	# cv2.imwrite('.png',img_rgb)
	if showImage:
		cv2.imshow("Result", img_rgb)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return count

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def processImage(imgPath, dungeonRun = True, showImage = False):
	large = cv2.imread(imgPath)
	rgb = cv2.pyrDown(large)
	small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

	# get dimensions of subsampled image and restrict processing area
	origH, origW = small.shape[:2]

	# dungeon item drop box offset
	xOffsetStart = origW*3//10
	xOffsetEnd = origW*7//10
	if dungeonRun:
		yOffsetStart = origH*23//100
		yOffsetEnd = origH*66//100
	else:
		yOffsetStart = origH*16//100
		yOffsetEnd = origH*59//100

	bgColor = [i for i in rgb[yOffsetStart + origH*12//100, xOffsetStart + origW*32//100]]
	# print("bgColor", bgColor, yOffsetStart + origH*12//100, xOffsetStart + origW*32//100)

	# BGR format
	rareBg = [74, 65, 28]
	heroBg = [65, 24, 88]
	legendBg = [17, 39, 108]
	popupBg = [15, 24, 35]

	if isSameColor(bgColor, popupBg):
		print("not a rune")
		runeType = ""
	if isSameColor(bgColor, rareBg):
		runeType = "Rare"
	elif isSameColor(bgColor, heroBg):
		runeType = "Hero"
	elif isSameColor(bgColor, legendBg):
		runeType = "Legend"
	else:
		print("unknown")
		return 
	print("begin processing rune")

	# show processed area
	cv2.rectangle(rgb, (xOffsetStart, yOffsetStart), (xOffsetEnd, yOffsetEnd), (0, 0, 255), 2)
	# image sub section
	small = small[yOffsetStart:yOffsetEnd, xOffsetStart:xOffsetEnd]

	#### get stars ####
	numStars = countStars(imgPath)
	# starOffsetX = origW*2//100
	# starOffsetY = origH*8//100
	# rectSize = origW*8//100

	# starROI = small[starOffsetY:starOffsetY+rectSize, starOffsetX:starOffsetX+rectSize]

	# print("starROI start", xOffsetStart+starOffsetX, yOffsetStart+starOffsetY)
	# cv2.rectangle(rgb, (xOffsetStart+starOffsetX, yOffsetStart+starOffsetY), (xOffsetStart+starOffsetX+rectSize, yOffsetStart+starOffsetY+rectSize), (255, 0, 0), 2)

	# (9,3) dual kernel, 0.05 0.03/0.02 - BEST SO FAR
	# possible (6,5) kernel

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
	grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	# using RETR_EXTERNAL instead of RETR_CCOMP
	contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	mask = np.zeros(bw.shape, dtype=np.uint8)

	texts = []
	for idx in range(len(contours)):
	    x, y, w, h = cv2.boundingRect(contours[idx])
	    mask[y:y+h, x:x+w] = 0
	    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
	    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

	    if r > 0.5 and w > 8 and h > 8:
	        orig = small.copy()

	        # display texts here
	        origH, origW = orig.shape[:2]

	        dx = int(w * 0.05)
	        dy = int(h * 0.03)

	        startX = max(0, x - dx)
	        startY = max(0, y - dy)
	        endX = min(origW, x + w + 2 * dx)
	        endY = min(origH, y + h + 2 * dy)

	        # cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
	        cv2.rectangle(rgb, (xOffsetStart+startX, yOffsetStart+startY), (xOffsetStart+endX-1, yOffsetStart+endY-1), (0, 255, 0), 2)
	        roi = orig[startY:endY, startX:endX]
	        # text = pytesseract.image_to_string(roi, config = '--psm 6')
	        data = pytesseract.image_to_data(roi, config = '--oem 1 --psm 6')
	        text = parseResult(data, y, runeType != "")
	        if text != "":
	        	texts.insert(0, text)

	# postprocessing rune if text not extracted correctly
	hasInnateSub = len(texts[0].split()) == 4
	runeSlot = texts[0][-2]

	# should find a more efficient way to check if stuff have been processed corrrectly but w/e
	# check for rune quality
	if texts[1] != runeType and runeType not in texts:
		texts.insert(1, runeType)

	# add rune star number
	texts.insert(2, str(numStars)) 

	# check for main sub
	if len(texts) != qualityDict[runeType] + (4 if hasInnateSub else 3):
		mainSub = '(M) ' 
		if runeSlot == '1':
			mainSub += "ATK+ " + '15' if numStars == 5 else '22'
		elif runeSlot == '3':
			mainSub += "DEF+ " + '15' if numStars == 5 else '22'
		elif runeSlot == '5':
			mainSub += "HP+ " + '270' if numStars == 5 else '360'
		if len(mainSub) > 4 and mainSub not in texts:
			texts.insert(3, mainSub)

	# check if rune has all of the information (processed correctly	+ postprocessed missing information)
	if texts[1] != runeType or len(texts) != qualityDict[runeType] + (4 if hasInnateSub else 3) or (hasInnateSub and texts[4][:3] != '(I)'):
		print("could not successfully process rune - missing rune type, main stat, innate stat when it exists, or not enough subs for given quality")
	else:
		print("successfully processed rune")
	print(texts)

	if showImage:
		cv2.imshow('rects', rgb)

		while True:
			if cv2.waitKey(1) == ord('q'):
				break
		cv2.destroyAllWindows()


# processImage(imgPath, dungeonRun = True, showImage = True)
processImage(imgPath, dungeonRun = False, showImage = True)


