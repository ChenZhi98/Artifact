import ast
import csv
import inspect

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")

def extract_functions_info(source_code):
    # Initialize lists to store function and class information
    func_details = []
    functions_num = 0

    # Parse the content into an AST
    try:
        parsed_content = ast.parse(source_code)
    except SyntaxError:
        print("Error: Invalid Python content.")
        return functions_num, func_details


    # Traverse the AST to find function definitions and class definitions
    for func in ast.walk(parsed_content):
        if isinstance(func, ast.FunctionDef):
            functions_num += 1
            function_name = func.name

            docstring_node = ast.get_docstring(func, clean=False)
            if docstring_node and docstring_node.strip()!="":
                # Get the start line number of the function
                start_lineno = func.lineno - 1  # Adjust for zero-based index

                # Find the end line number of the docstring
                end_lineno = func.body[0].value.end_lineno if func.body else start_lineno

                # Extract the exact text from the source code
                func_text_lines = source_code.splitlines()[start_lineno:end_lineno]
                func_text = "\n".join(func_text_lines)

                tokens = tokenizer.encode(func_text)

                # Typically, the GPT-2 model has a token limit of 1024 tokens
                if len(tokens) <= 512:
                    # Find the first and last occurrence of '"""' to separate the signature and docstring
                    first_quote_index = min(loc for loc, val in enumerate(func_text_lines) if '"""' or "'''" in val)
                    last_quote_index = max(loc for loc, val in enumerate(func_text_lines) if '"""' or "'''" in val)

                    signature = "\n".join(func_text_lines[:first_quote_index])
                    docstring = "\n".join(func_text_lines[first_quote_index:last_quote_index + 1])

                    func_details.append({
                        'func_name': function_name,
                        'signature': signature,
                        'docstring': docstring,
                        'signature_with_docstring': func_text
                    })

    return functions_num, func_details

def main():
    # Read content from a CSV file (replace 'your_csv_file.csv' with the actual file path)
    # Load the dataset with streaming
    dataset = load_dataset(path="PATH_TO_TRAINING_DATAASET", data_files=f'TRAINING_DATASET_NAME', split="train",
                           streaming=True)

    total_extracted_functions = []
    total_function_num = 0
    overall_function_count = 0  

    with open('PATH_TO_SAVE_EXTRACTED_FUCTIONS', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['function id', 'function name','signature with docstring'])

        for index, row in enumerate(dataset):
            content = row['content']

            functions_num, extracted_functions = extract_functions_info(content)

            total_extracted_functions.extend(extracted_functions)
            total_function_num+=functions_num

            for _, func_info in enumerate(extracted_functions, start=1):
                overall_function_count += 1 
                writer.writerow([overall_function_count, func_info['func_name'],func_info['signature_with_docstring']])

            print(f"No.{index} file, number of function docstring:{len(extracted_functions)}")
            print(
                f"Total number of functions:{total_function_num}, total function docstring:{len(total_extracted_functions)}")

            print("========================================================================")

if __name__ == "__main__":
    main()
