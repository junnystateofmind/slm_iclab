import csv
import random

context_enhancement_prompts = [
"As a psychologist, read this post and answer the following question.",
"If you were a psychologist, read this post and answer the following question."
]


questions =  ["Which level of depression can be assigned to the post?", "Determine how depressed this person is?"]
output_constraints = "Only choose one answer from the following options: Minimum, Mild, Moderate, Severe."

names = ["train", "test"]

cnt = 0

for name in names:
	with open(f"./raw/{name}.csv", "r") as input_file:
		with open(f"../../prompted_dataset/DepSeverity/{name}.csv", "w") as output_file:
			reader = csv.reader(input_file)
			writer = csv.writer(output_file)
			writer.writerow(["text", "label"])
			for id, row in enumerate(reader):
				if(id == 0):
					continue
				question = random.choice(questions)
				context_enhance = random.choice(context_enhancement_prompts)
				prompted_text = row[0] + "\n" + " ".join([context_enhance, question, output_constraints])
				writer.writerow([prompted_text, row[1]])
				cnt += 1
			output_file.close()
		input_file.close()


