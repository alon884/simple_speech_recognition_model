
import json
import numpy as np
from sklearn.model_selection import train_test_split


DATA_PATH = "/media/alon/DATA/ProjectsForCV/proj1/dataset"
SAVED_MODEL_PATH = "/media/alon/DATA/ProjectsForCV/proj1/model.h5"

LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32


def load_dataset(data_path):
	with open(data_path,"r") as fp:
		data = json.load(fp)

	# extract inputs and targets
	X = np.array(data["MFCCs"])
	y = np.array(data["labels"])

	return X,y



def get_data_splits(data_path, test_size=0.1):

	# load dataset
	X,y = load_dataset(data_path)

	# create train/validation/test splits
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
	X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=test_validation)

	# convert inputs from 2d to 3d arrays
	X_train = X_train[...,np.newaxis]
	X_validation = X_validation[...,np.newaxis]
	X_test = X_test[...,np.newaxis]

	return X_train, X_validation, X_test, y_train, y_validation, y_test



def main():

	# load train/validation/test data splits
	X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)


	# build the CNN model
	input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # coefficients 13, 1)

	model = build_model(input_shape, LEARNING_RATE)

	# train the model
	model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))

	# evaluate the model
	test_error, test_accuracy = model.evaluate(X_test, y_test)
	print(f"Test error: {test_error}, test_accuracy: {test_accuracy}")

	# save the model
	model.save(SAVED_MODEL_PATH)