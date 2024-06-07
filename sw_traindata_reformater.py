import json

def reformat_json(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()

        # Strip whitespace and commas, then wrap lines in a list
        json_objects = [json.loads(line.strip()) for line in lines]
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(json_objects, f, ensure_ascii=False, indent=4)
        
        print("JSON file reformatted successfully")
    
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Replace 'data/formatted_poems_cleaned.json' with your actual file path
input_path = 'data/formatted_poems_cleaned.json'
output_path = 'data/formatted_poems_cleaned_reformatted.json'
reformat_json(input_path, output_path)

