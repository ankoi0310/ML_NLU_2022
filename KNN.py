import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

####################
path = 'dataset'
testSize = 0.2
validationSize = 0.2
####################


# PREPOSSESSING FUNCTION FOR IMAGES FOR TRAINING
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# READ IMAGES FROM FOLDERS
imagesPerLabel = []  # LIST CONTAINING ALL THE IMAGES
labels = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS LABEL OF IMAGES
myList = os.listdir(path)


# IMPORT IMAGES AND LABELS
count = len(myList)
for label in os.listdir(path):
    images = os.listdir(path + "/" + label)
    for image in images:
        currentImage = cv2.imread(path + "/" + label + "/" + image)
        currentImage = cv2.resize(currentImage, (32, 32))
        imagesPerLabel.append(currentImage)
        labels.append(label)
    print(label, end=" ")
print("Total Images Detected: ", len(imagesPerLabel))  # 75000 images
print("Total IDs Detected: ", len(labels))


# CONVERT TO NUMPY ARRAY
imagesPerLabel = np.array(imagesPerLabel)
labels = np.array(labels)
print(imagesPerLabel.shape)
print(labels.shape)


# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(imagesPerLabel, labels, test_size=testSize)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationSize)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)


# PLOT BAR CHART FOR DISTRIBUTION OF IMAGES
numOfImages = []
for x in range(0, count):
    print(len(np.where(y_train == labels[x])[0]))
    numOfImages.append(len(np.where(y_train == labels[x])[0]))
print(numOfImages)


plt.figure(figsize=(10, 5))
plt.bar(range(0, count), numOfImages)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


# PREPOSSESSING
X_train = np.array([preProcessing(x) for x in X_train])
X_test = np.array([preProcessing(x) for x in X_test])
X_validation = np.array([preProcessing(x) for x in X_validation])


# RESHAPE FOR KNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1] * X_validation.shape[2])


def createModel():
    # KNN MODEL
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn


# CREATE MODEL
model = createModel()


# PREDICTION
y_pred = model.predict(X_test)


# EVALUATION
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# PREDICTION FOR VALIDATION SET
y_pred_validation = model.predict(X_validation)


# EVALUATION FOR VALIDATION SET
accuracy_validation = accuracy_score(y_validation, y_pred_validation)
print(f"Accuracy for Validation Set: {accuracy_validation}")


# PREDICTION FOR TEST SET
y_pred_test = model.predict(X_test)


# EVALUATION FOR TEST SET
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy for Test Set: {accuracy_test}")


# PREDICTION FOR NEW IMAGE
newImage = cv2.imread("basmati (4).jpg")
newImage = cv2.resize(newImage, (32, 32))
newImage = preProcessing(newImage)
newImage = newImage.reshape(1, newImage.shape[0] * newImage.shape[1])


# PREDICTION FOR NEW IMAGE
y_pred_newImage = model.predict(newImage)
print(y_pred_newImage)
