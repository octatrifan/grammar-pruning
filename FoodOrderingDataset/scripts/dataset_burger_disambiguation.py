import json

# Updated list: include separate tokens for drink sizes and side sizes.
burger_items = [
    'DRINK_TYPE(chocolate_shake)', 'DRINK_TYPE(coca_cola)', 'DRINK_TYPE(coffee)', 'DRINK_TYPE(diet_coke)',
    'DRINK_TYPE(dr_pepper)', 'DRINK_TYPE(hot_cocoa)', 'DRINK_TYPE(iced_tea)', 'DRINK_TYPE(milk)',
    'DRINK_TYPE(minute_maid)', 'DRINK_TYPE(pink_lemonade)', 'DRINK_TYPE(root_beer)', 'DRINK_TYPE(seven_up)',
    'DRINK_TYPE(strawberry_shake)', 'DRINK_TYPE(vanilla_shake)', 'DRINK_TYPE(zero_sugar_lemonade)',
    'MAIN_DISH_TYPE(cheese_burger)', 'MAIN_DISH_TYPE(chicken_sandwich)', 'MAIN_DISH_TYPE(double_cheese_burger)',
    'MAIN_DISH_TYPE(ham_burger)', 'MAIN_DISH_TYPE(salad)', 'MAIN_DISH_TYPE(vegan_burger)',
    'NOT(not)',
    'SIDE_TYPE(apple_slices)', 'SIDE_TYPE(baby_carrots)', 'SIDE_TYPE(curly_fries)',
    'SIDE_TYPE(french_fries)', 'SIDE_TYPE(garlic_fries)', 'SIDE_TYPE(sweet_potato_fries)',
    # Disambiguated size tokens for drinks and sides:
    'DRINK_SIZE(extra_large)', 'DRINK_SIZE(large)', 'DRINK_SIZE(medium)', 'DRINK_SIZE(small)',
    'SIDE_SIZE(extra_large)', 'SIDE_SIZE(large)', 'SIDE_SIZE(medium)', 'SIDE_SIZE(small)',
    'TOPPING(all_toppings)', 'TOPPING(bacon)', 'TOPPING(blue_cheese)', 'TOPPING(cheddar)',
    'TOPPING(jalapenos)', 'TOPPING(ketchup)', 'TOPPING(lettuce)', 'TOPPING(mayonnaise)',
    'TOPPING(mustard)', 'TOPPING(onion)', 'TOPPING(pickle)', 'TOPPING(spread)', 'TOPPING(tomato)',
    'number(1)', 'number(10)', 'number(11)', 'number(12)', 'number(13)', 'number(14)', 'number(15)',
    'number(2)', 'number(3)', 'number(4)', 'number(5)', 'number(6)', 'number(7)', 'number(8)', 'number(9)',
    'quantity(extra)', 'quantity(light)'
]
burger_items = [x.lower() for x in burger_items]

# Load the existing JSON data from a file
with open('FoodOrderingDataset/data/burger/processed_dev_disambiguation.json', 'r') as file:
    data = json.load(file)

# Modify the keys and add a new key-value pair
for item in data:  # Assuming the data is a list of dictionaries
    item['input'] = item.pop('SRC', None)  # Replace 'SRC' with 'input'
    item['output_generate'] = item.pop('TOPALIAS', None)
    item['output_extract'] = [x.lower() for x in item.pop('USED_ITEMS', [])]  # Convert to lowercase
    item['instruction_extract'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of all the corresponding items that you find in the order. "
        "You should use only items from the following list:\n"
        f"{burger_items}"
    )
    item['instruction_generate'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of the corresponding objects. You should use only the following classes in Python:\n"
        "class Topping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class BurgerOrder:\n"
        "    def __init__(self, number: int = 1, size: Optional[str] = None, main_dish_type: Optional[str] = None, "
        "toppings: Optional[List[Topping]] = None) -> None\n"
        "\n"
        "class DrinkOrder:\n"
        "    def __init__(self, number: int = 1, drink_type: Optional[str] = None, size: Optional[str] = None) -> None:\n"
        "\n"
        "class SideOrder:\n"
        "    def __init__(self, number: int = 1, side_type: Optional[str] = None, size: Optional[str] = None) -> None:\n"
        "\n"
        "The output should be a list of those objects.\n"
        "Here's an example:\n"
        "'input': 'i would like a vegan burger with lettuce tomatoes and onions and a large order of sweet potato fries',\n"
        "'output': '[BurgerOrder(number=1, main_dish_type='vegan_burger', toppings=[Topping(name='lettuce'), "
        "Topping(name='tomato'), Topping(name='onion')]), SideOrder(number=1, side_type='french_fries', size='large')]',"
    )

# Save the modified data to a new file
with open('FoodOrderingDataset/processed_data/burger_dataset_disambiguation.json', 'w') as file:
    json.dump(data, file, indent=4)
