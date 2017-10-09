import argparse
import matplotlib.pyplot as plt
import DicomDataManager as DDM
import os
import numpy as np
import timeit
from sklearn.cluster import KMeans
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def parser_init(parser):
	model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

	"""initialize parse arguments"""
	parser.add_argument('--data-folder', type=str, default='./data', metavar='PATH',\
		help='path to data folder')
	parser.add_argument('--cuda', type=bool, default=True, metavar='B',\
		help='specify usage of cuda (default: True)')
	parser.add_argument('--snapshot', type=str, default='./snapshot', metavar='PATH', \
		help='snapshot save location')
	
	args = parser.parse_args()

	args.cuda = args.cuda and torch.cuda.is_available()

	# change parser value here
	args.snapshot = '/home/jacky/disk0/projects/Jaw/snapshot-classification/snapshot_2150.pth.tar'
	args.data_folder = "/home/jacky/disk0/projects/Jaw/Data-DICOM/1_sorted/test"
	args.arch = 'resnet18'
	
	return args

def load_data(data_path,workers=0):
	# apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread
	transform = transforms.Compose([DDM.Rescale((224,224,0)),
		DDM.ToTensor()])  # 3rd dim with size 0 refers to no change in size

	# load data
	data_set = DDM.DicomDataSet(os.path.join(data_path),transform=transform,train=False)
	data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)

	return data_loader

def evaluate(data_loader,model,batch_size,cuda=True,probability=False):
	model.eval()

	if cuda:
		model = torch.nn.DataParallel(model) # multiple GPU parallelization
		model.cuda()
		cudnn.benchmark = True

	result = []

	for batch_idx, data in enumerate(data_loader):
		# get the inputs
		img = data['image']

		lower_list=[]
		upper_list=[]
		timer = timeit.default_timer()
		print(data['case_name'])

		for slice_num in range(img.size()[1]):
			input = img[:,slice_num:slice_num+1,:,:]

			# pytorch built in resnet need 3 channel input
			input_np = input.numpy()
			input_3C = np.repeat(input_np,3,axis=1)

			input = torch.from_numpy(input_3C)
			input_var = torch.autograd.Variable(input, volatile=True)

			# compute output
			output = model(input_var)
			_, predicted = torch.max(output.data, 1)

			slice_num = img.size()[1]-slice_num
			if predicted.cpu().numpy() == 1:
				lower_list.append(int(slice_num))
			elif predicted.cpu().numpy() == 2:
				upper_list.append(int(slice_num))

		# 	# plot the slice 
		# 	fig1 = plt.figure(1)
		# 	plt.ion()
		# 	plt.imshow(input_var.data.cpu().numpy()[0,0,...],cmap='gray')
		# 	if predicted.cpu().numpy() == 1:
		# 		plt.title('Lower, Slice: {}/{}'.format(slice_num,img.size()[1]))
		# 		print(predicted.cpu().numpy(),slice_num)
		# 	elif predicted.cpu().numpy() == 2:
		# 		plt.title('Upper, Slice: {}/{}'.format(slice_num,img.size()[1]))
		# 		print(predicted.cpu().numpy(),slice_num)
		# 	else:
		# 		plt.title('Nil, Slice: {}/{}'.format(slice_num,img.size()[1]))
		# 		print(predicted.cpu().numpy(),slice_num)
		# 	plt.axis('off')
		# 	plt.draw()
		# 	plt.pause(0.0001)
		# plt.close(fig1)

		batchTime = timeit.default_timer() - timer

		result.append([data['case_name'],sorted(lower_list),sorted(upper_list)])
	return result

def reject_outliers(data, m=2):
	data=np.asarray(data)
	return data[abs(data - np.mean(data)) < m * np.std(data)]

def StringRangeToList(stringRange):
	result = []
	for part in stringRange.split(','):
		if '-' in part:
			a, b = part.split('-')
			a, b = int(a), int(b)
			result.extend(range(a, b + 1))
		else:
			a = int(part)
			result.append(a)
	return result

def RemoveCommonElements(a, b):
	# remove elements in a if appears in b
	a_new = []
	b_new = []
	for i in a:
		if i not in b:
			a_new.append(i)

	return a_new

def main(parser):
	args = parser_init(parser)

	if args.cuda and torch.cuda.is_available():
		print('CUDA acceleration: Yes')
	else:
		if not torch.cuda.is_available():
			print('CUDA device not found')
		print('CUDA acceleration: No')

	# load snapshot
	if os.path.isfile(args.snapshot):
		print("Loading snapshot '{}'...".format(args.snapshot))
		snapshot = torch.load(args.snapshot)
		print("=> Snapshot '{}' loaded (epoch {})"
			.format(args.snapshot, snapshot['epoch']))
	else:
		print("No checkpoint found at '{}', evaluation abort".format(args.snapshot))
		return

	# create model
	arch = args.arch
	# arch = snapshot['arch']

	print("=> creating model '{}'".format(arch))
	model = models.__dict__[arch]()
	model.fc = nn.Linear(512, 3) # assuming that the fc layer has 512 neurons with 2 classes, otherwise change it 

	if arch.startswith('alexnet') or arch.startswith('vgg'):
		model.features = torch.nn.DataParallel(model.features)
	else:
		model = torch.nn.DataParallel(model)

	if args.cuda:
		model.cuda()

	# load state dict from snapshot
	model.load_state_dict(snapshot['state_dict'])

	# load data
	print 'loading data...'
	data_loader = load_data(args.data_folder)
	print 'finish loading data'
	result = evaluate(data_loader,model,1,args.cuda)

	# load result csv to check accuracy
	csv_file = "/home/jacky/disk0/projects/Jaw/classification_annotation/set1_selected.csv"
	csv_data = pd.read_csv(csv_file)
			

	# process the result into csv
	lower_accuracy_tp_avg = 0
	upper_accuracy_tp_avg = 0
	lower_accuracy_fp_avg = 0
	upper_accuracy_fp_avg = 0

	for i in range(len(result)):
		case = os.path.basename(result[i][0][0])
		lower = result[i][1]
		upper = result[i][2]

		# remove outliners from lower and upper list
		lower = reject_outliers(lower)
		upper = reject_outliers(upper)

		# check accuracy
		csv_idx = csv_data.Name[csv_data.Name == case].index.tolist()[0]
		lower_list_GT = StringRangeToList(csv_data.ix[csv_idx, 1:].as_matrix()[0])
		upper_list_GT = StringRangeToList(csv_data.ix[csv_idx, 1:].as_matrix()[1])

		lower_tp_accuracy = float(len(set(lower) & set(lower_list_GT)))/len(lower_list_GT)
		upper_tp_accuracy = float(len(set(upper) & set(upper_list_GT)))/len(upper_list_GT)

		# lower_fp_accuracy = (len(lower)-float(len(set(lower) & set(lower_list_GT))))/len(lower_list_GT)
		# upper_fp_accuracy = (len(upper)-float(len(set(upper) & set(upper_list_GT))))/len(upper_list_GT)
		lower_fp_count = len(lower)-len(set(lower) & set(lower_list_GT))
		upper_fp_count = len(upper)-len(set(upper) & set(upper_list_GT))

		lower_accuracy_tp_avg += lower_tp_accuracy
		upper_accuracy_tp_avg += upper_tp_accuracy

		# lower_accuracy_fp_avg += lower_fp_accuracy
		# upper_accuracy_fp_avg += upper_fp_accuracy

		print case
		print("Lower TP accuracy = {:.2f}%\nUpper TP accuracy = {:.2f}%".format(lower_tp_accuracy*100.0,upper_tp_accuracy*100.0))
		print("Lower count total = {}\nUpper count total = {}".format(len(lower_list_GT),len(upper_list_GT)))
		print("Lower TP count = {}\nUpper TP count = {}".format(len(set(lower) & set(lower_list_GT)),len(set(upper) & set(upper_list_GT))))
		# print("Lower FP accuracy = {:.2f}%\nUpper FP accuracy = {:.2f}%".format(lower_fp_accuracy*100.0,upper_fp_accuracy*100.0))
		print("Lower FP count = {}\nUpper FP count = {}".format(lower_fp_count,upper_fp_count))
		# print("Lower FP list")
		# print(RemoveCommonElements(lower,lower_list_GT))
		# print("Upper FP list")
		# print(RemoveCommonElements(upper,upper_list_GT))
		print upper

	print("Avg Lower TP accuracy = {:.2f}%\nAvg Upper TP accuracy = {:.2f}%".format(lower_accuracy_tp_avg/len(result)*100.0,upper_accuracy_tp_avg/len(result)*100.0))
	# print("Avg Lower FP accuracy = {:.2f}%\nAvg Upper FP accuracy = {:.2f}%".format(lower_accuracy_fp_avg/len(result)*100.0,upper_accuracy_fp_avg/len(result)*100.0))
		# slice_list_of_list = []

		# # use kmean clustering to classify upper jaw and lower jaw
		# for slice_num in result[i][1]:
		# 	slice_list_of_list.append([slice_num])

		# kmeans = KMeans(n_clusters=2, random_state=0).fit(slice_list_of_list)

		# upper = []
		# lower = []

		# if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
		# 	upper_idx = 0
		# 	lower_idx = 1
		# else:
		# 	upper_idx = 1
		# 	lower_idx = 0

		# for j in range(len(result[i][1])):
		# 	if kmeans.labels_[j] == upper_idx:
		# 		upper.append(result[i][1][j])
		# 	else:
		# 		lower.append(result[i][1][j])

		# upper_range = str(min(upper))+'-'+str(max(upper))
		# lower_range = str(min(lower))+'-'+str(max(lower))
		
		# print result[i]
		# print case
		# print upper_range
		# print lower_range

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation Tool')
	main(parser)