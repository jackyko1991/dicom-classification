import SimpleITK as sitk
import numpy as np
import os
import glob
import torch
import math
import random
import torchvision
import pandas as pd

class DicomDataSet(torch.utils.data.Dataset):
	"""
	Args:
		data_folder (string): Directory containing DICOM files
		csv_file (string): Path to the csv file with annotations
		transform (callable, optional): Optional transform to be applied on the sample
	"""

	def __init__(self, data_folder, csv_file=None, transform=None, train=True):
		self.data_folder = data_folder
		self.dirlist = os.listdir(data_folder)
		if train:
			self.csv = pd.read_csv(csv_file)
		self.transform = transform
		self.train = train

	def __getitem__(self, idx):
		dicomFolder = os.path.join(self.data_folder,self.dirlist[idx])

		reader = sitk.ImageSeriesReader()
		dicom_names = reader.GetGDCMSeriesFileNames(dicomFolder)
		reader.SetFileNames(dicom_names)
		img = reader.Execute()

		dicom_name = self.dirlist[idx]

		if self.train is True:
			# find the index from csv annotation file
			csv_idx = self.csv.Name[self.csv.Name == dicom_name].index.tolist()[0]
			annotation = self.csv.ix[csv_idx, 1:].as_matrix()

			annotation_list = []
			annotation_list.extend(self.StringRangeToList(annotation[0]))
			annotation_list.extend(self.StringRangeToList(annotation[1]))
		else:
			annotation_list = []

		sample = {'image':img, 'annotation': annotation_list, 'case_name': dicomFolder}

		# apply transform to the data if necessary
		if self.transform:
			sample = self.transform(sample)

		return sample 

	def __len__(self):
		return len(self.dirlist)

	def StringRangeToList(self,stringRange):
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

class Rescale(object):
	"""
	Rescale the image in a sample to a given size.

	Args:
		output_size (int or tuple): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same. Set output size to be zero if want to keep that 
			dimension to be unchanged.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size,output_size,1)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

	def __call__(self, sample):
		image, annotation = sample['image'], sample['annotation']

		self.output_spacing = []
		for i in range(3):
			self.output_size = list(self.output_size) # tuple element cannot change directly, need to convert to list first
			if self.output_size[i] == 0:
				self.output_size[i] = image.GetSize()[i]
			self.output_size = tuple(self.output_size)
			self.output_spacing.append(image.GetSpacing()[i] * (image.GetSize()[i] / float(self.output_size[i])))
		self.output_spacing = tuple(self.output_spacing)

		resampler = sitk.ResampleImageFilter()
		resampler.SetSize(self.output_size);
		resampler.SetOutputSpacing(self.output_spacing);
		resampler.SetOutputOrigin(image.GetOrigin())
		resampler.SetOutputDirection(image.GetDirection())
		resampler.SetNumberOfThreads(30)
		img = resampler.Execute(image);

		return {'image': img, 'annotation': annotation, 'case_name': sample['case_name'], 'slice': sample['slice']}

class GetSlice(object):
	"""
	Get the dicom slice according to the desired location

	Args:
		slice_num (int): Desired output slice number.
	"""

	def __init__(self,slice_num):
		assert isinstance(slice_num, int)
		self.slice_num = slice_num

	def __call__(self,sample):
		image, annotation = sample['image'], sample['annotation']

		size_old = image.GetSize()
		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_old[0],size_old[1],1])
		roiFilter.SetIndex([0,0,slice_num])
		image = roiFilter.Execute(image)

		return {'image': image, 'annotation':annotation, 'case_name': sample['case_name']}

class RandomSlice(object):
	"""
	Return a random slice and annotation will change to desired boolean value
	Drop ratio is implemented for randomly dropout selected with empty label. (Default to be 0.1)
	when drop ratio = 1, all image of that kind will be discarded, else when drop ratio = 0, all image will be accepted
	"""
	def __init__(self,drop_ratio=0.1):
		assert isinstance(drop_ratio, float)
		if drop_ratio >=0 and drop_ratio<=1:
			self.drop_ratio = drop_ratio
		else:
			raise RuntimeError('Drop ratio should be between 0 and 1')

	def __call__(self,sample):
		image, annotations = sample['image'], sample['annotation']

		slice_pass = False

		while not slice_pass:
			slice_num = random.randint(0,image.GetSize()[2]-1)
			if (image.GetSize()[2]-slice_num) in annotations:
				annotation = 1
				slice_pass = True
			else:
				annotation = 0
				slice_pass = False

			if annotation == 0 and self.drop(self.drop_ratio):
				slice_pass = True

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([image.GetSize()[0],image.GetSize()[1],1])
		roiFilter.SetIndex([0,0,slice_num])

		image = roiFilter.Execute(image)

		return {'image': image, 'annotation': annotation, 'case_name': sample['case_name'], 'slice': slice_num+1}

	def drop(self,probability):
		return random.random() <= probability

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, annotation = sample['image'], sample['annotation']

		# pytorch require input to be float
		image = torch.from_numpy(sitk.GetArrayFromImage(image))
		image = image.float()

		return {'image': image,
				'annotation': annotation,
				'case_name': sample['case_name'],
				'slice': sample['slice']}

class Normalization(object):
	"""Normalize an image by setting its mean to zero and variance to one."""

	def __call__(self, sample):
		self.normalizeFilter = sitk.NormalizeImageFilter()
		img, annotation = sample['image'], sample['annotation']
		img = self.normalizeFilter.Execute(img)

		return {'image': img, 'annotation': annotation, 'case_name': sample['case_name']}

class Rotate(object):
	"""
	Rotate the image 

	"""

	def __call__(self,sample):
		return sample


class Resample(object):
	"""Resample the volume in a sample to a given voxel size
	Args:
		voxel_size (float or tuple): Desired output size.
		If float, output volume is isotropic.
		If tuple, output voxel size is matched with voxel size
		Currently only support linear interpolation method
	"""

	def __init__(self, voxel_size):
		assert isinstance(voxel_size, (float, tuple))
		if isinstance(voxel_size, float):
			self.voxel_size = (voxel_size,voxel_size,voxel_size)
		else:
			assert len(voxel_size) == 3
			self.voxel_size = voxel_size

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmentation']

		old_spacing = img.GetSpacing()
		old_size = img.GetSize()

		new_spacing = self.voxel_size

		new_size = []
		for i in range(3):
			new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
		new_size = tuple(new_size)

		resampler = sitk.ResampleImageFilter()
		resampler.SetInterpolator(2)
		resampler.SetOutputSpacing(new_spacing)
		resampler.SetSize(new_size)

		# resample on image
		resampler.SetOutputOrigin(img.GetOrigin())
		resampler.SetOutputDirection(img.GetDirection())
		# print("Resampling image...")
		img = resampler.Execute(img)

		# resample on segmentation
		resampler.SetOutputOrigin(seg.GetOrigin())
		resampler.SetOutputDirection(seg.GetDirection())
		# print("Resampling segmentation...")
		seg = resampler.Execute(seg)

		return {'image': img, 'segmentation': seg}

class SitkToTensor(object):
	"""Convert sitk image to Tensors"""

	def __init__(self, random_rotate = False):
		self.random_rotate = random_rotate

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmentation']

		img_np = sitk.GetArrayFromImage(img)
		img_np = np.float32(img_np)

		seg_np = sitk.GetArrayFromImage(seg)
		seg_np = np.uint8(seg_np)

		if self.random_rotate:
			order = np.random.permutation([0,1,2])
			img_np = np.transpose(img_np, tuple(order))
			seg_np = np.transpose(seg_np, tuple(order))

		img_np_4D = np.zeros((1,img_np.shape[0],img_np.shape[1],img_np.shape[2]))
		img_np_4D[0,:,:,:] = img_np
		img_tensor = torch.from_numpy(img_np_4D).float()
			
		seg_np_4D = np.zeros((1,seg_np.shape[0],seg_np.shape[1],seg_np.shape[2]))
		seg_np_4D[0,:,:,:] = seg_np
		seg_tensor = torch.from_numpy(seg_np_4D).long()

		return {'image': img_tensor, 'segmentation': seg_tensor}

class Padding(object):
	"""Add padding to the image if size is smaller than patch size

	Args:
		output_size (tuple or int): Desired output size. If int, a cubic volume is formed
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert all(i > 0 for i in list(self.output_size))

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']
		size_old = img.GetSize()

		if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
			return sample
		else:
			# print(img.GetSpacing())
			resampler = sitk.ResampleImageFilter()
			resampler.SetInterpolator(2)
			resampler.SetOutputSpacing(img.GetSpacing())
			resampler.SetSize(self.output_size)

			# resample on image
			resampler.SetOutputOrigin(img.GetOrigin())
			resampler.SetOutputDirection(img.GetDirection())
			img = resampler.Execute(img)

			# resample on label
			resampler.SetOutputOrigin(seg.GetOrigin())
			resampler.SetOutputDirection(seg.GetDirection())
			seg = resampler.Execute(seg)

			return {'image': img, 'segmentation': seg}

class RandomCrop(object):
	"""Crop randomly the image in a sample. This is usually used for data augmentation.
		Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.1)
		This transformation only applicable in train mode
	Args:
		output_size (tuple or int): Desired output size. If int, cubic crop is made.
	"""

	def __init__(self, output_size, drop_ratio=0.1):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert isinstance(drop_ratio, float)
		if drop_ratio >=0 and drop_ratio<=1:
			self.drop_ratio = drop_ratio
		else:
			raise RuntimeError('Drop ratio should be between 0 and 1')

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']
		size_old = img.GetSize()
		size_new = self.output_size

		contain_label = False

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

		while not contain_label: 
			# get the start crop coordinate in ijk
			if size_old == size_new:
				[start_i, start_j, start_k] = [0,0,0]
			else:
				start_i = np.random.randint(0, size_old[0]-size_new[0])
				start_j = np.random.randint(0, size_old[1]-size_new[1])
				start_k = np.random.randint(0, size_old[2]-size_new[2])

			roiFilter.SetIndex([start_i,start_j,start_k])

			seg_crop = roiFilter.Execute(seg)
			statFilter = sitk.StatisticsImageFilter()
			statFilter.Execute(seg_crop)

			# will iterate until a sub volume containing label is extracted
			if statFilter.GetSum()<1:
				contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
			else:
				contain_label = True

		img_crop = roiFilter.Execute(img)

		return {'image': img_crop, 'segmentation': seg_crop}

	def drop(self,probability):
		return random.random() <= probability