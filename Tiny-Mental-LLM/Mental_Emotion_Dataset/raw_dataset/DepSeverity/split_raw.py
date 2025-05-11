import csv
import random
import copy
from collections import defaultdict
names = ["train", "test"]

datas = {name: [] for name in names}


with open("./raw.csv", "r") as input_file:
	reader = csv.reader(input_file)
	text_label = []
	cnt = 0
	group_by_labels = defaultdict(list)

	for id, row in enumerate(reader):
		if(id == 0):
			continue
		cnt += 1
		group_by_labels[row[1]].append(row[0])
		text_label.append([row[0], row[1]])
	
	for label in group_by_labels.keys():
		n = len(group_by_labels[label])
		_train = group_by_labels[label][: int(0.8 * n)]
		_test = group_by_labels[label][int(0.8 * n): ]
		datas["train"] += [[x, label] for x in _train]
		datas["test"] += [[x, label] for x in _test]
	input_file.close()



for name in names:
	with open(f"./raw/{name}.csv", "w") as output_file:
		writer = csv.writer(output_file)
		writer.writerow(["text", "label"])
		for row in datas[name]:
			writer.writerow(row)
		output_file.close()

