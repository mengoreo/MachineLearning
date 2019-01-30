import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

DATADIR = '/Users/mengoreo/Downloads/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50


try:
	with open('data.pickle', 'rb') as f:
		training_data = pickle.load(f)
except:
	training_data = []
	def create_training_data():
		for category in CATEGORIES:
			path = os.path.join(DATADIR, category)
			class_num = CATEGORIES.index(category)
			for img in os.listdir(path):
				try:
					img_array = cv2.imread(os.path.join(path, img),
										cv2.IMREAD_GRAYSCALE)
					# make all images to be the same size
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
					training_data.append([new_array, class_num])
				except Exception as e:
					# print(e)
					pass

	create_training_data()

data = open('data.pickle', 'wb')
pickle.dump(training_data, data)

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)
X = np.array(X)
# gray scale, RGB 3
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle.dump(X, open('X.pickle', 'wb'))
pickle.dump(y, open('y.pickle', 'wb'))

# print(training_data[0][-1])
# plt.imshow(training_data[0][0], cmap='gray')
# plt.show()