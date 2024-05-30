import subprocess
import time
import xml.etree.ElementTree as ET


# Define the command to be executed
command = [
    'java', '-jar', 'simian-4.0.0.jar', 
    '-threshold=6',
    '-includes=FILES_TO_INCLUDE',
    '-formatter=xml:REPORT_PATH/REPORT_NAME'
]

# Run the command
subprocess.run(command,capture_output=True)


# Define a function to load and print the key value under 'simian' 'check' 'summary'
def load_and_print_key_value(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Navigate to the 'summary' element directly since it's nested under 'check'
    summary = root.find('.//summary')


    # Print the key-value pairs if the 'summary' element exists
    if summary is not None:
        for key, value in summary.attrib.items():
            print(f"{key}: {value}")

# Call the function with the XML file name
load_and_print_key_value('REPORT_PATH/REPORT_NAME')

