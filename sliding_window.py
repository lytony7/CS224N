import json
import pandas as pd

# Read the JSON file
with open('ccpc_test.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse JSON and extract poem contents
poems = [json.loads(line) for line in lines]
contents = [poem['content'] for poem in poems]

# Split each poem content by '|'
split_contents = [content.split('|') for content in contents]

# Define a function to create formatted examples and clean the data
def create_formatted_examples(poems):
    inputs = []
    targets = []
    for poem in poems:
        if len(poem) == 4:  # Ensure the poem has exactly 4 lines
            input_lines = "，".join(poem[:2]) + "。" + poem[2] + "，"
            target_line = poem[3] + "。"
            if "□" not in input_lines and "□" not in target_line:  # Clean data by removing lines containing "□"
                inputs.append(input_lines)
                targets.append(target_line)
    return inputs, targets

# Create formatted examples
inputs, targets = create_formatted_examples(split_contents)

# Convert to DataFrame for better visualization
df = pd.DataFrame({'Input': inputs, 'Target': targets})
# print(df)

# Save to a new JSON file or any other required format
df.to_json('formatted_poems_test.json', orient='records', lines=True, force_ascii=False)
