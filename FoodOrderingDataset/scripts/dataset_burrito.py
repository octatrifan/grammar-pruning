import json

burrito_items = ['BEAN_FILLING(black_beans)', 'BEAN_FILLING(pinto_beans)', 'DRINK_TYPE(blackberry_izze)', 'DRINK_TYPE(bottled_water)', 'DRINK_TYPE(mexican_coca-cola)', 'DRINK_TYPE(nantucket_apple_juice)', 'DRINK_TYPE(nantucket_peach_orange_juice)', 'DRINK_TYPE(nantucket_pineapple_orange_juice)', 'DRINK_TYPE(sparkling_water)', 'DRINK_TYPE(tractor_lemonade)', 'DRINK_TYPE(tractor_organic_black-tea)', 'MAIN_FILLING(barbacoa)', 'MAIN_FILLING(carnitas)', 'MAIN_FILLING(chicken)', 'MAIN_FILLING(steak)', 'MAIN_FILLING(tofu)', 'MAIN_FILLING(veggie)', 'NOT(not)', 'RICE_FILLING(brown_rice)', 'RICE_FILLING(cauliflower_rice)', 'RICE_FILLING(white_rice)', 'SALSA_TOPPING(corn_salsa)', 'SALSA_TOPPING(fresh_tomato_salsa)', 'SALSA_TOPPING(green_chili_salsa)', 'SALSA_TOPPING(red_chili_salsa)', 'SIDE_TYPE(chips)', 'SIDE_TYPE(guacamole)', 'SIDE_TYPE(queso_blanco)', 'TOPPING(cheese)', 'TOPPING(fajita_veggies)', 'TOPPING(guacamole)', 'TOPPING(romaine_lettuce)', 'TOPPING(sour_cream)', 'number(1)', 'number(10)', 'number(11)', 'number(12)', 'number(13)', 'number(14)', 'number(15)', 'number(2)', 'number(3)', 'number(4)', 'number(5)', 'number(6)', 'number(7)', 'number(8)', 'number(9)', 'quantity(extra)', 'quantity(light)']
burrito_items = [x.lower() for x in burrito_items]
# Load the existing JSON data from a file
with open('FoodOrderingDataset/data/burrito/processed_train.json', 'r') as file:
    data = json.load(file)

# Modify the keys and add new key-value pair
for item in data:  # Assuming the data is a list of dictionaries
    item['input'] = item.pop('SRC', None)  # Replace 'SRC' with 'input'
    item['output_generate'] = item.pop('TOPALIAS', None)
    item['output_extract'] = [x.lower() for x in item.pop('USED_ITEMS', [])]  # Replace 'USED_ITEMS' with 'output' and convert to lower
    item['instruction_extract'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of all the corresponding items that you find in the order. You should use only items from the following list:\n"
        f'{burrito_items}'
    )
    item['instruction_generate'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of the corresponding objects. You should use only the following classes in Python:\n"
        "class Topping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class SalsaTopping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class MainClass:\n"
        "    def __init__(self, number: int = 1, main_filling: Optional[str] = None, rice_filling: Optional[str] = None, bean_filling: Optional[str] = None, salsa_toppings: Optional[List[SalsaTopping]] = None, toppings: Optional[List[Topping]] = None) -> None:"        
        "class BurritoOrder(MainClass)"
        "class TacoOrder(MainClass)"
        "class SaladOrder(MainClass)"
        "class BurritoBowlOrder(MainClass)"
        "class QuesadillaOrder(MainClass)"
        "class DrinkOrder:\n"
        "    def __init__(self, number: int = 1, drink_type: Optional[str] = None) -> None:\n"
        "\n"
        "The output should be a list of those objects.\n"
    )

# Save the modified data to a new file
with open('FoodOrderingDataset/processed_data/train_burrito_dataset.json', 'w') as file:
    json.dump(data, file, indent=4)