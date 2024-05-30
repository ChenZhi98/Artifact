from datasets import load_dataset
import json
import gzip

# Load your dataset
dataset = load_dataset('{PATH_TO_EXTRACTED_ORG_DATASETS}', data_files='{DATASET_NAME}')

# Split the dataset into train and valid dataset
train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=1)
train_data = train_test_split['train']
valid_data = train_test_split['test']

# Function to compress and save datasets in JSON format
def save_compressed_json(data, filename):
    with gzip.open(filename, 'wt', encoding='UTF-8') as file:
        for sample in data:
            file.write(json.dumps(sample) + '\n')

# Save the train and valid datasets
save_compressed_json(train_data, '{PATH_TO_SAVE_DATASETS}/{DATASET_NAME}_train.json.gz')
save_compressed_json(valid_data, '{PATH_TO_SAVE_DATASETS}/{DATASET_NAME}_valid.json.gz')
