import json

sub_items = ['BASE_SANDWICH(black_forest_ham)', 'BASE_SANDWICH(buffalo_chicken)', 'BASE_SANDWICH(chicken_and_bacon_ranch)', 'BASE_SANDWICH(cold_cut_combo)', 'BASE_SANDWICH(italian)', 'BASE_SANDWICH(meatball_marinara)', 'BASE_SANDWICH(oven_roasted_chicken)', 'BASE_SANDWICH(steak_and_cheese)', 'BASE_SANDWICH(sweet_onion_chicken_teriyaki)', 'BASE_SANDWICH(tuna)', 'BASE_SANDWICH(turkey_breast)', 'BASE_SANDWICH(veggie)', 'DRINK_TYPE(coca_cola)', 'DRINK_TYPE(dasani_water)', 'DRINK_TYPE(gatorade)', 'DRINK_TYPE(honest_kids_super_fruit_punch)', 'DRINK_TYPE(low_fat_milk)', 'DRINK_TYPE(orange_juice)', 'DRINK_TYPE(sprite)', 'DRINK_TYPE(vitamin_water)', 'NOT(not)', 'SIDE_TYPE(baked_lays_original)', 'SIDE_TYPE(chocolate_chip)', 'SIDE_TYPE(doritos_nacho_cheese)', 'SIDE_TYPE(lays_classic)', 'SIDE_TYPE(miss_vickies_jalapeno)', 'SIDE_TYPE(oatmeal_raisin)', 'SIDE_TYPE(raspberry_cheesecake)', 'SIDE_TYPE(sunchips_harvest_cheddar)', 'SIDE_TYPE(white_chip_macadamia_nut)', 'TOPPING(Or(american_cheese,monterey_cheddar,pepperjack,provolone,swiss))', 'TOPPING(Or(green_peppers,banana_peppers,black_pepper))', 'TOPPING(american_cheese)', 'TOPPING(bacon)', 'TOPPING(banana_peppers)', 'TOPPING(black_olives)', 'TOPPING(black_pepper)', 'TOPPING(buffalo_sauce)', 'TOPPING(caesar_sauce)', 'TOPPING(chipotle_southwest)', 'TOPPING(cucumbers)', 'TOPPING(green_peppers)', 'TOPPING(guacamole)', 'TOPPING(honey_mustard)', 'TOPPING(jalapenos)', 'TOPPING(lettuce)', 'TOPPING(marinara_sauce)', 'TOPPING(monterey_cheddar)', 'TOPPING(oil)', 'TOPPING(parmesan_cheese)', 'TOPPING(pepperjack)', 'TOPPING(pepperoni)', 'TOPPING(pickles)', 'TOPPING(provolone)', 'TOPPING(ranch)', 'TOPPING(red_onions)', 'TOPPING(red_wine_vinegar)', 'TOPPING(regular_mayonnaise)', 'TOPPING(salt)', 'TOPPING(spinach)', 'TOPPING(sweet_onion_sauce)', 'TOPPING(swiss)', 'TOPPING(tomatoes)', 'TOPPING(vinaigrette)', 'TOPPING(yellow_mustard)', 'number(1)', 'number(10)', 'number(11)', 'number(12)', 'number(13)', 'number(14)', 'number(15)', 'number(2)', 'number(3)', 'number(4)', 'number(5)', 'number(6)', 'number(7)', 'number(8)', 'number(9)', 'quantity(extra)', 'quantity(light)']
sub_items = [x.lower() for x in sub_items]
# Load the existing JSON data from a file
with open('FoodOrderingDataset/data/sub/processed_train.json', 'r') as file:
    data = json.load(file)

# Modify the keys and add new key-value pair
for item in data:  # Assuming the data is a list of dictionaries
    item['input'] = item.pop('SRC', None)  # Replace 'SRC' with 'input'
    item['output_generate'] = item.pop('TOPALIAS', None)
    item['output_extract'] = [x.lower() for x in item.pop('USED_ITEMS', [])]  # Replace 'USED_ITEMS' with 'output' and convert to lower
    item['instruction_extract'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of all the corresponding items that you find in the order. You should use only items from the following list:\n"
        f'{sub_items}'
    )
    item['instruction_generate'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of the corresponding objects. You should use only the following classes in Python:\n"
        "class Topping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class SandwichOrder:\n"
        "    def __init__(self, number: int = 1, size: Optional[str] = None, base_sandwich: Optional[str] = None, "
        "toppings: Optional[List[Topping]] = None) -> None\n"
        "\n"
        "class DrinkOrder:\n"
        "    def __init__(self, number: int = 1, drink_type: Optional[str] = None) -> None :\n "
        "\n"
         "class SideOrder:\n"
        "    def __init__(self, number: int = 1, side_type: Optional[str] = None) -> None :\n "
        "\n"
        "The output should be a list of those objects.\n"
        "Here's an example:\n"
        "'input': 'i'd like a meatball marinara sandwich and a cold cut combo with pepper jack cheese yellow mustard and red onions',\n"
        "'output': '[SandwichOrder(number=1, base_sandwich='meatball_marinara'), SandwichOrder(number=1, base_sandwich='cold_cut_combo', toppings=[Topping(name='pepperjack'), Topping(name='yellow_mustard'), Topping(name='red_onions')])]',"
    )

# Save the modified data to a new file
with open('FoodOrderingDataset/processed_data/train_sub_dataset.json', 'w') as file:
    json.dump(data, file, indent=4)
