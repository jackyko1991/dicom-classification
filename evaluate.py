import argparse
import matplotlib.pyplot as plt

import torchvision.models as models
import DicomDataManager as DDM

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

	args.cuda = not args.cuda and torch.cuda.is_available()

	# change parser value here
	args.snapshot = '/home/jacky/disk0/projects/Jaw/snapshot-classification/snapshot_50.pth.tar'
	args.data_folder = '../../data/evaluate'
	
	return args

def load_data(data_path,workers=0):
	# apply transform to input data, support multithreaded reading, num_workers=0 refers to use main thread
	transform = transforms.Compose([DDM.RandomSlice(drop_ratio=0.5),
		DDM.Rescale(224),
		DDM.ToTensor()])

	# img_transform = tvTransform.Compose([tvTransform.ToTensor()])

	# load data
	data_set = data.NiftiDataSet(os.path.join(data_path),transform=data_transform,train=False)
	data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)

	return data_loader

def main(parser):
	args = parser_init(parser)

	if args.cuda and torch.cuda.is_available():
		print('CUDA acceleration: Yes')
	else:
		if not torch.cuda.is_available():
			print('CUDA device not found')
		print('CUDA acceleration: No')

	# create model
	print("=> creating model '{}'".format(args.arch))
	model = models.__dict__[args.arch]()
	model.fc = nn.Linear(512, 2) # assuming that the fc layer has 512 neurons with 2 classes, otherwise change it 

	if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
		model.features = torch.nn.DataParallel(model.features)
	else:
		model = torch.nn.DataParallel(model)

	if args.cuda:
		model.cuda()

	if os.path.isfile(args.snapshot):
		print("Loading snapshot '{}'...".format(args.snapshot))
		snapshot = torch.load(args.snapshot)
		model.load_state_dict(snapshot['state_dict'])
		print("=> Snapshot '{}' loaded (epoch {})"
			.format(args.snapshot, snapshot['epoch']))
	else:
		print("No checkpoint found at '{}', evaluation abort".format(args.snapshot))
		return

	# load data
	print 'loading data...'
	data_loader = load_data(args.data_folder)
	print 'finish loading data'
	evaluate(model,data_loader,args.cuda)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation Tool')
	main(parser)