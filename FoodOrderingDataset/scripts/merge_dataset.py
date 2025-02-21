import json

# Filenames
pizza_file = 'FoodOrderingDataset/processed_data/train_pizza_dataset.json'
sub_file = 'FoodOrderingDataset/processed_data/train_sub_dataset.json'
burrito_file = 'FoodOrderingDataset/processed_data/train_burrito_dataset.json'
all_file = 'FoodOrderingDataset/processed_data/train_dataset_full.json'

# Function to load JSON data from a file
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Load data from each JSON file
pizza_data = load_json(pizza_file)
sub_data = load_json(sub_file)
burrito_data = load_json(burrito_file)

# Merge the data (assuming all are lists of dictionaries)
all_data = pizza_data + sub_data + burrito_data

# Function to add the new key with concatenated values
def augment_data(data):
    for item in data:
        input_value = item.get('input', '')
        used_items_value = item.get('static_extract', [])
        used_items_str = ', '.join(used_items_value).lower()  # Convert list to lowercase string
        item['input_augmented'] = f"{input_value}\nItems Found: [{used_items_str}]"
    return data

# Augment the merged data
augmented_data = augment_data(all_data)

# Write the augmented data to the new JSON file
with open(all_file, 'w') as f:
    json.dump(augmented_data, f, indent=4)

print(f"Augmented data written to {all_file}")