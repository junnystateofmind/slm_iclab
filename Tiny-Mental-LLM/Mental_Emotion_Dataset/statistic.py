import csv
import argparse



def analyze():
	datasets = ["train", "test"]
	names = ["SDCNL", "ISEAR", "DepSeverity", "Dreaddit", "DR"]
	for name in names:
		for dataset in datasets:	
			with open(f"./prompted_dataset/{dataset}/{name}.csv", "r") as input_file:
				reader = csv.reader(input_file)

				cnt = 0
				label_cnt = {}
				label_list = []

				for id, row in enumerate(reader):
					if(id == 0):
						continue
					
					label = row[1].lower()
					
					for id in range(len(label)):
						if(label[id] == '.'):
							label = label[id + 1: ].strip()
							break
					
					if(not (label in label_cnt.keys())):
						label_cnt[label] = 0
					label_cnt[label] += 1
		                        
					if(len(row) == 2):
						label_list.append(row[1])
					cnt += 1

				label_percentage = {}
				sum = 0
				for key in label_cnt.keys():
					sum += label_cnt[key]
				
				for key in label_cnt.keys():
					label_percentage[key] = round(label_cnt[key] / sum,5) * 100

				print(f"For {name} {dataset}")
				print(f"There are {cnt} data entries")
				print(f"Label count {len(label_list)}")
				print(f"Label distribution {label_cnt}")
				print(label_percentage)
				input_file.close()


if __name__ == "__main__":
	
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", required = True)

	args = parser.parse_args()
	"""
	analyze()
