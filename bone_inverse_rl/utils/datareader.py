import json

def read_json_data(file_path):
    """
    Reads and parses JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: Parsed JSON data as a Python list.
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
# file_path = "/path/to/your/json_file.json"
# data = read_json_data(file_path)
# print(data)