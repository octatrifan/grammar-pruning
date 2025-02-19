import json
import spacy 
from spacy.pipeline import EntityRuler
from spacy import displacy
import os
import re

def parse_line(line):
    # split the line into phrase and category
    parts = line.strip().split('\t')
    # edge case for invalid lines
    if len(parts) != 2:
        return None, None  
    
    phrase, category = parts
    
    # return the phrase and the full category string
    return phrase, category.strip()  # Strip any extra whitespace

def process_NER(input_order, dataset = "coffee"):
    # preload an English nlp model
    nlp = spacy.load("en_core_web_sm")

    # add to rule based NER to pipeline
    ner_ruler = nlp.add_pipe("entity_ruler", 
                        # prioritize pre-trained labels first
                        before="ner", 
                        config={"phrase_matcher_attr": "LOWER"})
    
    def read_file_categories(food_type):
        
        # open the folder for the input food type
        file_path = f"FoodOrderingDataset/data/{food_type}/alias"
        
        # grab the names of all the files in that folder
        text_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]

        patterns = []

        # iterate through each file
        for file in text_files:
            # open the current file
            path_to_file = f"{file_path}/{file}"
            with open(path_to_file, 'r') as file:
                # read each line (consists of a phrase and its corrsponding category) of the given file
                for line in file:
                    # skip empty lines
                    if line.strip():  
                        phrase, category_info = parse_line(line)
                        if phrase and category_info:
                            schema = {}
                            schema["pattern"] = phrase
                            schema["label"] = category_info
                            patterns.append(schema)

        return patterns

    category_patterns = read_file_categories(dataset)

    # add the categories to the ner pipeline
    ner_ruler.add_patterns(category_patterns)

    # output array to store found categories
    found_categories = []

    # parse the input order
    doc = nlp(input_order)

    # add all found categories to the output list and return it
    for ent in doc.ents:
        found_categories.append((ent.text, ent.label_))
        
    return found_categories

# Filenames
pizza_file = 'FoodOrderingDataset/processed_data/train_pizza_dataset.json'
sub_file = 'FoodOrderingDataset/processed_data/train_sub_dataset.json'
burrito_file = 'FoodOrderingDataset/processed_data/train_burrito_dataset.json'
# all_file = 'FoodOrderingDataset/processed_data/pizza_dataset_NER.json'
# all_file = 'FoodOrderingDataset/processed_data/sub_dataset_NER.json'
all_file = 'FoodOrderingDataset/processed_data/burrito_dataset_NER.json'

# Function to load JSON data from a file
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Load data from each JSON file
# pizza_data = load_json(pizza_file)
# sub_data = load_json(sub_file)
burrito_data = load_json(burrito_file)

# Merge the data (assuming all are lists of dictionaries)
# all_data = pizza_data + sub_data + burrito_data
# all_data = pizza_data
# all_data = sub_data
all_data = burrito_data

# Function to add the new key with concatenated values
def augment_data(data):
    for item in data:
        input_value = item.get('input', '')
        # used_items_value = item.get('static_extract', [])
        used_items_value = process_NER(input_value, 'pizza')
        # used_items_value = process_NER(input_value, 'sub')
        used_items_value_decoupled = [x[0] + ' - ' + x[1] for x in used_items_value]
        # print(input_value, used_items_value_decoupled)
        used_items_str = ', '.join(used_items_value_decoupled).lower()  # Convert list to lowercase string
        item['input_augmented_ner'] = f"{input_value}\nItems Found: [{used_items_str}]"
    return data

# Augment the merged data
augmented_data = augment_data(all_data)

# Write the augmented data to the new JSON file
with open(all_file, 'w') as f:
    json.dump(augmented_data, f, indent=4)

print(f"Augmented data written to {all_file}")
