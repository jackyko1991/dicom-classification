import os
import pandas as pd

csv = pd.read_csv("/home/jacky/disk0/projects/Jaw/classification_annotation/set1.csv")
data_list = os.listdir("/home/jacky/disk0/projects/Jaw/Data-DICOM/1_sorted")

# for i in range(len(data_list)):
# 	train_list[i] = train_list[i][-43:]

sorted_csv = pd.DataFrame()
# sorted_csv = sorted_csv.append({'name': 'Zed', 'age': 9, 'height': 2}, ignore_index=True)

for i in range(len(csv)):
	if (csv.Name[i] in data_list):
		sorted_csv = sorted_csv.append(csv.iloc[[i]], ignore_index=True)

sorted_csv.to_csv('/home/jacky/disk0/projects/Jaw/classification_annotation/set1_selected.csv',index=False)