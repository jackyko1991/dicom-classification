import argparse
import matplotlib.pyplot as plt
import DicomDataManager as DDM
import os
import numpy as np
import timeit

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
	args.snapshot = '/home/jacky/disk0/projects/Jaw/snapshot-classification/snapshot_225.pth.tar'
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

	for batch_idx, data in enumerate(data_loader):
		# get the inputs
		img = data['image']

		true_list=[]
		timer = timeit.default_timer()
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

			if predicted.cpu().numpy() == 1:
				true_list.append(slice_num+1)

		# 	# plot the slice 
		# 	fig1 = plt.figure(1)
		# 	plt.ion()
		# 	plt.imshow(input_var.data.cpu().numpy()[0,0,...],cmap='gray')
		# 	if predicted.cpu().numpy() == 1:
		# 		plt.title('True, Slice: {}/{}'.format(slice_num+1,img.size()[1]))
		# 		print(predicted.cpu().numpy(),slice_num+1)
		# 	else:
		# 		plt.title('False, Slice: {}/{}'.format(slice_num+1,img.size()[1]))
		# 		print(predicted.cpu().numpy(),slice_num+1)
		# 	plt.axis('off')
		# 	plt.draw()
		# 	plt.pause(0.0001)
		# plt.close(fig1)

		batchTime = timeit.default_timer() - timer
		print(data['case_name'],true_list,batchTime)
		exit()

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
	model.fc = nn.Linear(512, 2) # assuming that the fc layer has 512 neurons with 2 classes, otherwise change it 

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
	evaluate(data_loader,model,1,args.cuda)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation Tool')
	main(parser)