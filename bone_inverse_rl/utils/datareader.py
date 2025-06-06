import json
from typing import List, Optional, Any

def read_json_data(file_path: str) -> Optional[List[Any]]:
    """
    Reads and parses JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Optional[List[Any]]: Parsed JSON data as a Python list, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
    return None

# Example usage:
if __name__ == "__main__":
    file_path = "/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_test.json"
    data = read_json_data(file_path)
    print(data)