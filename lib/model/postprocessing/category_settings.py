import csv
import os

category_config = {}

category_directory = "./config/post_processing/categories"

def load_category_settings():
    for root, _, files in os.walk(category_directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                if file_name not in category_config:
                    category_config[file_name] = {}  # Initialize as an empty dictionary
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        category_config[file_name][row['OriginalTag']] = {
                            'RenamedTag': row['RenamedTag'],
                            'MinMarkerDuration': row['MinMarkerDuration'],
                            'MaxGap': row['MaxGap'],
                            'RequiredDuration': row['RequiredDuration'],
                            'TagThreshold': row['TagThreshold'],
                        }
    
load_category_settings()