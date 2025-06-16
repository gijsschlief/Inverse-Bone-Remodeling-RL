import json
from typing import List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json_data(file_path: str) -> Optional[List[Any]]:
    """
    Reads and parses JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Optional[List[Any]]: Parsed JSON data as a Python list, or None if an error occurs.
    """

    if file_path.startswith('/'):
        logging.info(f"Absolute path provided: {file_path}")
    else:
        logging.warning(f"Relative path provided: {file_path}. This may lead to file not found errors if the script is run from a different directory.")
    if not file_path.endswith('.json'):
        logging.error(f"Invalid file format: {file_path}. Expected a .json file.")
        return None
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logging.error(f"JSON decode error in file: {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None

# Example usage:
if __name__ == "__main__":
    file_path = "/home/gijs/Desktop/Thesis/data/raw/training_data_test.json"
    data = read_json_data(file_path)
    print(data)