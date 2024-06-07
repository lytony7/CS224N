import json

# Read the content of the text file
with open('chinese_poems.txt', 'r', encoding='utf-8') as file:
    poems = file.readlines()

# Strip any whitespace characters like '\n' at the end of each line
poems = [poem.strip() for poem in poems]

# Write the poems to a JSON file in the desired format
with open('train.json', 'w', encoding='utf-8') as json_file:
    json.dump(poems, json_file, ensure_ascii=False, indent=4)

print("Transformation complete. The file 'train.json' has been created.")
