import os
import time
import xml.etree.ElementTree as ET
import csv

# Start the timer
start_time = time.time()

simian_xml_file_name = "PATH_TO_REPORT_FILE"
saved_file_name = "PATH_TO_SAVE_PROCESSED_REPORT_FILE"


# Parse the XML file
tree = ET.parse(simian_xml_file_name)
root = tree.getroot()

# Initialize a list to store the results
results = []

# Initialize a set to store unique fingerprints
unique_fingerprints = set()
sum_of_lines = 0


# Iterate over each 'set' in the XML data
for set_element in root.iter('set'):
    # Extract the fingerprint and lineCount
    fingerprint = set_element.get('fingerprint')
    line_count = int(set_element.get('lineCount'))
    # Collect source files from all blocks
    source_files = [block.get('sourceFile') for block in set_element.findall('block')]

    # Check if all blocks are from different files and if the fingerprint is unique
    if len(set(source_files)) > 1 and fingerprint not in unique_fingerprints:
        # Add the fingerprint to the set of unique fingerprints
        unique_fingerprints.add(fingerprint)
        sum_of_lines += line_count

        # Append the result to the list
        results.append({
            'fingerprint': fingerprint, 
            'lineCount': line_count, 
        })

# Save the results to a CSV file
with open(saved_file_name, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['fingerprint', 'lineCount']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the rows
    for result in results:
        writer.writerow(result)

print("The results have been successfully saved.")

# Print the total number of unique fingerprints and the sum of lines
print(f"Total number of unique fingerprints: {len(unique_fingerprints)}")
print(f"Sum of lines: {sum_of_lines}")

