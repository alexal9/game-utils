import sys
import numpy as np

def simulate(offense, defense):
	offense_bar = np.zeros(len(offense))
	defense_bar = np.zeros(len(defense))
	overflow = []

	tick = 0
	moved = False

	while True:
		if moved:
			command = input()
			items = command.split()
			if command == 'q':
				break
			# aoe boost (or decrease)
			elif len(items) == 3 and items[0] == 'boost':
				if items[1] == 'o':
					offense_bar += int(items[2])
				elif items[1] == 'd':
					defense_bar += int(items[2])
			# single target boost (or decrease)
			elif len(items) == 4 and items[0] == 'single':
				if items[1] == 'o':
					offense_bar[ int(items[2]) ] += int(items[3])
				elif items[1] == 'd': 
					defense_bar[ int(items[2]) ] += int(items[3])
			moved = False

		tick += 1
		# add tick
		for i in range(len(offense)):
			if i not in overflow:
				offense_bar[i] += offense[i] * 0.07
		for i in range(len(defense)):
			if i+len(offense) not in overflow:
				defense_bar[i] += defense[i] * 0.07

		# if atk bar > 150, add to overflow stack

		# observed behavior is slightly more lenient on spd tuning for attack
		# possibly some extra logic/mechanics that need to be determined/implemented
		# this simulation requires stricter spd tuning for offense teams 
		comb = np.hstack((offense_bar, defense_bar))
		# max_val = np.max(comb)
		while np.max(np.round(comb)) >= 150:
			# print(max_val)
			comb = np.round(comb)
			index = np.argmax(comb)
			if index not in overflow:
				overflow.append(index)
			comb[index] = 0
			# max_val = np.max(np.round(comb))
			# print(index, overflow, offense_bar, defense_bar)

		print(tick, offense_bar, defense_bar, overflow)
		index = -1

		if overflow:
			index = overflow.pop(0)
		elif np.max(comb) > 100:
			index = np.argmax(comb)

		if index != -1:
			if index < len(offense_bar):
				print('offense unit', index + 1, 'took a turn')
				offense_bar[index] = 0
			else:
				print('defense unit', index - len(offense_bar) + 1, 'took a turn')
				defense_bar[index - len(offense_bar)] = 0
			moved = True

			print(tick, offense_bar, defense_bar, overflow)

if __name__ == '__main__':
	# offense = np.array([286, 239, 205])
	# offense = np.array([106, 99, 105, 103]) * 1.34 + np.array([189, 198, 147, 149])
	# defense = np.array([106, 99, 105, 103]) * 1.34 + np.array([189, 198, 147, 149])

	# vanessa bastet purian lushen
	offense = np.array([101, 99, 97, 103]) * 1.48 + np.array([115, 198, 143, 40])

	# bastet bernard lushen lushen
	# offense = np.array([99, 111, 103, 103]) * 1.15 + np.array([198, 125, 75, 75])
	# defense = np.array([101, 120, 102, 105]) * 1.48 + np.array([115, 134, 152, 144])
	defense = np.array([101, 120, 102, 105]) * 1.48 + np.array([115, 163, 152, 144])
	# defense = np.array([101, 111, 102, 105]) * 1.48 + np.array([115, 119, 152, 144])

	# giants sim
	# offense = np.array([103, 102, 101, 103, 103]) * 1.15 + np.array([75, 55, 41, 30, 34])
	# boss left right
	# defense = np.array([75, 75, 100])
	simulate(offense, defense)