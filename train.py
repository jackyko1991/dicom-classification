import argparse
import os
import shutil
import time
import numpy as np
import timeit

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import DicomDataManager as DDM
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

best_prec1 = 0

def parser_init(parser):
	parser.add_argument('--data-folder', metavar='DIR',default='./data', type=str,
						help='path to dataset')
	parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
						choices=model_names,
						help='model architecture: ' +
							' | '.join(model_names) +
							' (default: resnet18)')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=90, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('-b', '--batch-size', default=256, type=int,
						metavar='N', help='mini-batch size (default: 256)')
	parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--log-interval', '-p', default=25, type=int,
						metavar='N', help='logging frequency (default: 25)')
	parser.add_argument('--resume', default='./snapshot_001.pth.tar', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--csv', default='./annotation.csv', type = str, metavar='PATH',
		help='path to the annotation csv file')
	parser.add_argument('--cuda', default=True, type=bool, metavar='B',
		help='specify usage of cuda (default: True)')
	parser.add_argument('--snapshot', default='./snapshot', type=str, metavar='DIR',
		help='directory to store snapshot')

	args = parser.parse_args()

	args.data_folder = "/home/jacky/disk0/projects/Jaw/Data-DICOM/1_sorted"
	args.csv = "/home/jacky/disk0/projects/Jaw/classification_annotation/set1_selected.csv"
	args.snapshot = "/home/jacky/disk0/projects/Jaw/snapshot-classification"
	args.epoch = 2500
	args.train_batch_size = 10
	args.test_batch_size = 1
	args.workers = 30
	args.log_interval = 3
	return args

def load_data(data_path,csv_path,train_batch_size,test_batch_size, workers=0):
	# # apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread
	# transform = transforms.Compose([NDM.Normalization(),\
	# 	NDM.Resample(0.4),\
	# 	NDM.Padding(patch_size),\
	# 	NDM.RandomCrop(patch_size,drop_ratio),\
	# 	NDM.SitkToTensor(random_rotate=True)])
	transform = transforms.Compose([DDM.RandomSlice(),
		DDM.Rescale(224),
		DDM.ToTensor()])
	# transform = transforms.Compose([DDM.RandomSlice(),
	# 	DDM.ToTensor()])

	# load data
	train_set = DDM.DicomDataSet(os.path.join(data_path,'train'),csv_path,transform=transform,train=True)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	test_set = DDM.DicomDataSet(os.path.join(data_path,'test'),csv_path,transform=transform,train=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,shuffle=True,num_workers=workers,pin_memory=True)

	return [train_loader,test_loader]

def main(parser):
	args = parser_init(parser)

	# create model
	print("=> creating model '{}'".format(args.arch))
	model = models.__dict__[args.arch]()
	model.fc = nn.Linear(512, 2) # assuming that the fc layer has 512 neurons with 2 classes, otherwise change it 

	if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
	else:
		model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	start_epoch = 1

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> Loading snapshot '{}'".format(args.resume))
			snapshot = torch.load(args.resume)
			start_epoch = snapshot['epoch'] + 1
			model.load_state_dict(snapshot['state_dict'])
			print("=> Snapshot '{}' loaded (epoch {})"
				.format(args.resume, snapshot['epoch']))
		else:
			print("=> No checkpoint found at '{}', training starts from epoch 1".format(args.resume))

	if args.epochs < start_epoch:
		print("Epoch to train is less than the one in snapshot, training abort")
		return

	torch.backends.cudnn.enabled = False
	# cudnn.benchmark = True

	# Data loading code
	[train_loader, test_loader] = load_data(args.data_folder, args.csv,args.train_batch_size, args.test_batch_size, args.workers)

	print("Number of training data: "+str(len(train_loader.dataset)))
	print("Number of testing data: "+str(len(test_loader.dataset)))

	# initialize values for plotting
	train_loss_record = np.zeros(args.epochs)
	train_loss_record[:] = np.NAN
	test_accuracy_record = np.empty(args.epochs)
	test_accuracy_record[:] = np.NAN

	if os.path.isfile(args.resume):
		train_loss_record[0:start_epoch-1] = snapshot['train_loss'][0:start_epoch-1]
		test_accuracy_record[0:start_epoch-1] = snapshot['test_accuracy'][0:start_epoch-1]

	# plot loss and accuracy
	fig0 = plt.figure(0)
	plt.ion()
	ax1 = fig0.add_subplot(1,1,1)
	ax2 = ax1.twinx()
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Accuracy')
	ax1.set_title('Epoch: 0, Train Loss: 0, Test Accuracy: 0, Benchmark: 0 s/epoch')
	ax1.set_xlim([1,start_epoch])
	ax1.set_ylim([0,1])
	ax2.set_ylim([0,1])
	line1, = ax1.plot(range(1,args.epochs+1), train_loss_record, 'k-',label='Train Loss')
	line2, = ax2.plot(range(1,args.epochs+1), test_accuracy_record, 'b-', label='Test Accuracy')
	#legend
	handles, labels = ax1.get_legend_handles_labels()
	plt.legend([line1, line2], \
		['Train Loss', 'Test Accuracy'],loc=2)
	plt.draw()

	# timer for benchmarking
	timer = timeit.default_timer()
	epoch_count = 1

	for epoch in range(1, args.epochs + 1):
		if epoch >= start_epoch:
			print('Epoch: {}'.format(epoch))
			# 	adjust_learning_rate(optimizer, epoch)

			# train for one epoch
			[epoch_train_loss, snapshot] = train(train_loader, model, criterion, optimizer, epoch, args.cuda)
			epoch_test_accuracy = test(test_loader,model,epoch,args.cuda) # test accuracy after when each epoch train ends

			# epoch_train_loss = 0
			train_loss_record[epoch-1] = epoch_train_loss

			# epoch_test_accuracy = 1
			test_accuracy_record[epoch-1] = epoch_test_accuracy

			# update train loss plot
			line1.set_ydata(train_loss_record)
			line2.set_ydata(test_accuracy_record)

			ax1.set_xlim([1,epoch])
			ax1.set_ylim([0,max(train_loss_record)])
			ax1.set_title('Epoch: %s \nTrain Loss: %s, Test Accuracy: %s\n Benchmark: %s s/epoch'\
				%(epoch, \
				"{0:.2f}".format(epoch_train_loss),\
				"{0:.2f}".format(epoch_test_accuracy),\
				"{0:.2f}".format((timeit.default_timer() - timer)/epoch_count)))
			plt.draw()
			plt.pause(0.000000001)

			epoch_count = epoch_count+1

			# save snapshot
			if epoch % args.log_interval == 0:
				snapshot['train_loss'] = train_loss_record
				snapshot['test_accuracy'] = test_accuracy_record
				snapshot_path = args.snapshot + '/snapshot_' + str(epoch) + '.pth.tar'
				torch.save(snapshot, snapshot_path)
				print('Snapshot of epoch {} saved at {}.'.format(epoch, snapshot_path))

	plt.show()

def train(train_loader, model, criterion, optimizer, epoch, cuda=True):
	losses = AverageMeter()

	# switch to train mode
	model.train()

	for batch_idx, data in enumerate(train_loader):
		input = data['image']
		target =  data['annotation']

		# pytorch built in resnet need 3 channel input
		input_np = input.numpy()
		input_3C = np.repeat(input_np,3,axis=1)

		input = torch.from_numpy(input_3C)

		if cuda:
			target = target.cuda(async=True)

		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		# # plot the slice 
		# fig1 = plt.figure(1)
		# plt.ion()
		# plt.imshow(input_var.data.cpu().numpy()[0,0,...],cmap='gray')
		# if target.cpu().numpy() == 1:
		# 	plt.title('True')
		# else:
		# 	plt.title('False')
		# plt.axis('off')
		# plt.draw()
		# plt.pause(0.0001)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		losses.update(loss.data[0], input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print('Training of epoch {} finished. Average Training Loss: {:.6f}'.format(
		epoch, losses.avg))
	snapshot = {'epoch': epoch, \
	'state_dict': model.state_dict(), \
	'optimizer': optimizer.state_dict(), \
	'loss': losses.avg}

	return [losses.avg, snapshot]


def test(test_loader, model, epoch, cuda=True):
	# switch to evaluate mode
	model.eval()

	accuracy = AverageMeter()

	for batch_idx, data in enumerate(test_loader):
		input = data['image']
		target =  data['annotation']

		# pytorch built in resnet need 3 channel input
		input_np = input.numpy()
		input_3C = np.repeat(input_np,3,axis=1)

		input = torch.from_numpy(input_3C)

		if cuda:
			target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)

		# measure accuracy
		_, predicted = torch.max(output.data, 1)
		print type(predicted)
		print type(target)
		print (predicted == target).sum()
		accuracy.update((predicted == target).sum(), target.size(0))

	print('Testing of epoch {} finished. Average Testing Accuracy: {:.2f}%.'.
		format(epoch, accuracy.avg*100))

	return accuracy.avg

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	main(parser)