from datasets import Dataset
import pandas as pd
from datasets import load_dataset


google_repos = ['google', 'GoogleCloudPlatform', 'google-research', 'google-deepmind', 'googleapis', 'GoogleChromeLabs',
                'GoogleChrome', 'googlefonts', 'googlemaps', 'googlecodelabs', 'google-developer-training',
                'googlesamples', 'googleworkspace', 'googlearchive', 'google-research-datasets', 'googlecreativelab',
                'GoogleContainerTools', 'googleprojectzero', 'googleanalytics', 'googleads', 'googlecolab',
                'googleinterns', 'google-admin', 'GoogleCodeExporter', 'google-ar', 'google-coral', 'googleforgames',
                'google-marketing-solutions', 'GoogleTrends', 'GoogleCloudPlatformTraining', 'googlevr',
                'GoogleWebComponents']

microsoft_repos = ['microsoft', 'MicrosoftDocs', 'MicrosoftLearning', 'Azure', 'MicrosoftEdge', 'microsoftgraph',
                   'microsoftarchive', 'Microsoft-OpenSource-Labs', 'OfficeDev', 'msintuneappsdk']

facebook_repos = ['facebook', 'facebookresearch', 'facebookincubator', 'facebookexperimental', 'fbsamples']

# Initialize container lists
google_python_files = []
microsoft_python_files = []
facebook_python_files = []


# Load the datasets of github python source codes downloaeded from BigQuery platform
data_shards_directory = "{PATH_TO_PYTHON_DATA_SHARDS}"
dataset = load_dataset('csv', data_files=f'{data_shards_directory}/*', split='train', streaming=True)


# Iterate through all rows in the dataset
for row in dataset:
    orga_name = row['repo_name'].split('/')[0]
    if orga_name in microsoft_repos:
        microsoft_python_files.append(row)
    elif orga_name in google_repos:
        google_python_files.append(row)
    elif orga_name in facebook_repos:
        facebook_python_files.append(row)


google_df = pd.DataFrame(google_python_files)
microsoft_df = pd.DataFrame(microsoft_python_files)
facebook_df = pd.DataFrame(facebook_python_files)


google_df['size'] = google_df['size'].astype(int)
microsoft_df['size'] = microsoft_df['size'].astype(int)
facebook_df['size'] = facebook_df['size'].astype(int)


google_dataset = Dataset.from_pandas(google_df)
microsoft_dataset = Dataset.from_pandas(microsoft_df)
facebook_dataset = Dataset.from_pandas(facebook_df)

# Save extracted datasets  
data_saved_directory = "{PATH_TO_SAVE_EXTRACTED_DATA}"
microsoft_dataset.to_csv(data_saved_directory+"microsoft.csv")
google_dataset.to_csv(data_saved_directory+"google.csv")
facebook_dataset.to_csv(data_saved_directory+"meta.csv")

# Print the number of rows and the sum of the 'size' column for each dataset
print(f"Google: Rows = {len(google_dataset)}, Size sum = {str(sum(google_dataset['size'])/(1024*1024))} MB")
print(f"Microsoft: Rows = {len(microsoft_dataset)}, Size sum = {str(sum(microsoft_dataset['size'])/(1024*1024))} MB")
print(f"Facebook: Rows = {len(facebook_dataset)}, Size sum = {str(sum(facebook_dataset['size'])/(1024*1024))} MB")

