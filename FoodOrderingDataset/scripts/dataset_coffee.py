import json

coffee_items = ['DRINK_TYPE(americano)', 'DRINK_TYPE(cappuccino)', 'DRINK_TYPE(drip_coffee)', 'DRINK_TYPE(espresso)', 'DRINK_TYPE(hot_chocolate)', 'DRINK_TYPE(latte)', 'NOT(not)', 'ROAST_TYPE(cinnamon_roast)', 'ROAST_TYPE(continental_roast)', 'ROAST_TYPE(dark_roast)', 'ROAST_TYPE(french)', 'ROAST_TYPE(full_city_roast)', 'ROAST_TYPE(guatemalan)', 'ROAST_TYPE(italian)', 'ROAST_TYPE(light_roast)', 'ROAST_TYPE(medium_roast)', 'SIZE(extra_large)', 'SIZE(large)', 'SIZE(regular)', 'SIZE(small)', 'STYLE(decaf)', 'STYLE(flavored)', 'STYLE(iced)', 'STYLE(skinny)', 'TOPPING(ESPRESSO_SHOT_1)', 'TOPPING(ESPRESSO_SHOT_2)', 'TOPPING(ESPRESSO_SHOT_3)', 'TOPPING(ESPRESSO_SHOT_4)', 'TOPPING(caramel_syrup)', 'TOPPING(cinnamon)', 'TOPPING(cinnamon_dolce_syrup)', 'TOPPING(crumbles)', 'TOPPING(drizzles)', 'TOPPING(foam)', 'TOPPING(hazelnut_syrup)', 'TOPPING(honey)', 'TOPPING(raspberry_syrup)', 'TOPPING(syrup)', 'TOPPING(vanilla_syrup)', 'TOPPING(whipped_cream)', 'number(1)', 'number(10)', 'number(11)', 'number(12)', 'number(13)', 'number(14)', 'number(15)', 'number(2)', 'number(3)', 'number(4)', 'number(5)', 'number(6)', 'number(7)', 'number(8)', 'number(9)', 'quantity(extra)', 'quantity(light)']
coffee_items = [x.lower() for x in coffee_items]
# Load the existing JSON data from a file
with open('FoodOrderingDataset/data/coffee/processed_dev.json', 'r') as file:
    data = json.load(file)

# Modify the keys and add new key-value pair
for item in data:  # Assuming the data is a list of dictionaries
    item['input'] = item.pop('SRC', None)  # Replace 'SRC' with 'input'
    item['output_generate'] = item.pop('TOPALIAS', None)
    item['output_extract'] = [x.lower() for x in item.pop('USED_ITEMS', [])]  # Replace 'USED_ITEMS' with 'output' and convert to lower
    item['instruction_extract'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of all the corresponding items that you find in the order. You should use only items from the following list:\n"
        f'{coffee_items}'
    )
    item['instruction_generate'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of the corresponding objects. You should use only the following classes in Python:\n"
        "class Topping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class DrinkOrder:\n"
        "    def __init__(self, number: int = 1, drink_type: Optional[str] = None, size: Optional[str] = None, style: Optional[str] = None, "
        "roast_type: Optional[str] = None, toppings: Optional[List[Topping]] = None) -> None:\n"
        "\n"
        "The output should be a list of those objects.\n"
    )

# Save the modified data to a new file
with open('FoodOrderingDataset/processed_data/coffee_dataset.json', 'w') as file:
    json.dump(data, file, indent=4)