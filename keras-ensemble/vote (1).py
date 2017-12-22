"""
Noah Gundotra November 2017
for SUSA Data Consulting

Sums the predictions of 8 models and prints accuracy on datasets
"""
import pandas as pd
import numpy as np
import keras

def load_model_npz(args):
	"""
	Takes in list of filenames of model_results to load in
	Files are expected to be lists of [train,test]
	"""
	train_xs = []
	test_xs = []
	for npz in args:
		tmp = np.load(npz)
		train_xs.append(tmp['arr_0'])
		test_xs.append(tmp['arr_1'])
	return train_xs, test_xs

def vote(model_arr,test_arr=None):
	"""
	Sums output of 8 models and predicts the category that has max prob

	model_arr: list of [?,7] arrays
	test_arr: dataset of [?,] truth labels elem of [0, 6]
	"""
	# print(model_arr,len(model_arr),len(model_arr[0]),len(model_arr[0][0]))
	vote_list = np.zeros((len(model_arr[0]),len(model_arr[0][0])))
	for arr in model_arr:
	    vote_list = vote_list+arr

	vote_list = np.argmax(vote_list, axis=1)

	if test_arr is not None:
		print("accuracy is {}".format(np.mean(vote_list==test_arr)))
	return vote_list

if __name__ == '__main__':
	# Load in the data=
	train_data_y = pd.read_pickle('normalized_fer2013_labels.pkl').astype(int)
	test_data_y = pd.read_pickle('normalized_test_fer2013_labels.pkl').astype(int)

	# Load in the predictions of the 4 Convolutional and 4 ResNet models
	conv_train, conv_test = load_model_npz(['convolutional{}.npz'.format(i) for i in [0,2,4,6]])
	res_train, res_test = load_model_npz(['resnetty{}.npz'.format(i) for i in [1,3,5,7]])

	print("\n"*3)
	print("-"*20)
	print("Train Data\n")
	out_train = vote(conv_train+res_train,train_data_y)
	print("-"*20)
	print("Test Data\n")
	out_test = vote(conv_test+res_test,test_data_y)
	print("-"*20)
	print("\n\nDone\n\n")


