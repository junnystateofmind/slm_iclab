# Mental Emotion Dataset

### About

This dataset is constructed by merging 5 smaller datasets: Dreaddit, Depression Reddit, DepSeverity, ISEAR, and SCDNL


### Get Started

The plain text and label in the dataset can be found in `./raw_dataset/{dataset_name}/raw`. The original data can be found in `./raw_dataset/{dataset_name}/original`

In each folder of the raw dataset, there could be 2-3 python files to process the data

- `add_prompt.py`

This python file adds the prompt to the `text` of the raw data and create a new file in `./prompted_dataset/{dataset}` corresponding for `train` and `test` dataset.


- `get_text_and_label.py`

This python file extract `text` and `label` from original data file provided by the data creators

- `split_raw.py`

Some datasets give a single file so we need this file to split them into train and test dataset, with the proportion 90/10


### Note

When running the code, be careful of special character when getting the text from original csv

- Example: ' => â, \n => á