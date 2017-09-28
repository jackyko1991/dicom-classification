import os
import pandas as pd
import shutil

csv = pd.read_csv("/home/jacky/disk0/projects/Jaw/classification_annotation/set1_selected.csv")
train_list = os.listdir("/home/jacky/disk0/projects/Jaw/Data/jaw/train")
test_list = os.listdir("/home/jacky/disk0/projects/Jaw/Data/jaw/test")

for folder in train_list:
	case = folder[-43:]

	if case in csv['Name'].tolist():
		print case
		src = os.path.join("/home/jacky/disk0/projects/Jaw/Data/jaw/train",folder)
		dest = os.path.join("/home/jacky/disk0/projects/Jaw/Data/jaw_classification",case)
		shutil.copytree(src,dest)

for folder in test_list:
	case = folder[-43:]

	if case in csv['Name'].tolist():
		print case
		src = os.path.join("/home/jacky/disk0/projects/Jaw/Data/jaw/test",folder)
		dest = os.path.join("/home/jacky/disk0/projects/Jaw/Data/jaw_classification",case)
		shutil.copytree(src,dest)