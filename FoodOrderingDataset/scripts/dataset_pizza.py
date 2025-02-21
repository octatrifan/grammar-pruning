import json

pizza_items = ['NOT(not)', 'containerType(bottle)', 'containerType(can)', 'drinkType(cherry_coke)', 'drinkType(cherry_pepsi)', 'drinkType(coffee)', 'drinkType(coke)', 'drinkType(coke_zero)', 'drinkType(diet_coke)', 'drinkType(diet_ice_tea)', 'drinkType(diet_pepsi)', 'drinkType(diet_sprite)', 'drinkType(dr_pepper)', 'drinkType(fanta)', 'drinkType(ginger_ale)', 'drinkType(ice_tea)', 'drinkType(lemon_ice_tea)', 'drinkType(mountain_dew)', 'drinkType(pellegrino_sparkling_water)', 'drinkType(pepsi)', 'drinkType(perrier_sparkling_water)', 'drinkType(pineapple_soda)', 'drinkType(seven_up)', 'drinkType(sprite)', 'drinkType(water)', 'number(1)', 'number(10)', 'number(11)', 'number(12)', 'number(13)', 'number(14)', 'number(15)', 'number(2)', 'number(3)', 'number(4)', 'number(5)', 'number(6)', 'number(7)', 'number(8)', 'number(9)', 'quantity(extra)', 'quantity(light)', 'size(extra_large)', 'size(large)', 'size(lunch_size)', 'size(medium)', 'size(party_size)', 'size(personal - size)', 'size(personal_size)', 'size(regularsize)', 'size(small)', 'style(all_toppings)', 'style(all_vegetables)', 'style(cauliflower_crust)', 'style(cheese_lover)', 'style(chicago_style)', 'style(combination)', 'style(deep_dish)', 'style(gluten_free_crust)', 'style(hawaiian)', 'style(keto_crust)', 'style(margherita)', 'style(meat_lover)', 'style(mediterranean)', 'style(mexican)', 'style(neapolitan)', 'style(new_york_style)', 'style(sourdough_crust)', 'style(stuffed_crust)', 'style(supreme)', 'style(thick_crust)', 'style(thin_crust)', 'style(vegan)', 'style(vegetarian)', 'topping(alfredo_chicken)', 'topping(american_cheese)', 'topping(anchovies)', 'topping(artichokes)', 'topping(arugula)', 'topping(bacon)', 'topping(balsamic_glaze)', 'topping(banana_peppers)', 'topping(basil)', 'topping(bay_leaves)', 'topping(bbq_chicken)', 'topping(bbq_pulled_pork)', 'topping(bbq_sauce)', 'topping(beans)', 'topping(beef)', 'topping(broccoli)', 'topping(buffalo_chicken)', 'topping(buffalo_mozzarella)', 'topping(buffalo_sauce)', 'topping(caramelized_onions)', 'topping(carrots)', 'topping(cheddar_cheese)', 'topping(cheese)', 'topping(cheeseburger)', 'topping(cherry_tomatoes)', 'topping(chicken)', 'topping(chorizo)', 'topping(cumin)', 'topping(dried_peppers)', 'topping(dried_tomatoes)', 'topping(feta_cheese)', 'topping(fried_onions)', 'topping(garlic_powder)', 'topping(green_olives)', 'topping(green_peppers)', 'topping(grilled_chicken)', 'topping(grilled_pineapple)', 'topping(ham)', 'topping(hot_peppers)', 'topping(italian_sausage)', 'topping(jalapeno_peppers)', 'topping(kalamata_olives)', 'topping(lettuce)', 'topping(low_fat_cheese)', 'topping(meatballs)', 'topping(mozzarella_cheese)', 'topping(mushrooms)', 'topping(olive_oil)', 'topping(olives)', 'topping(onions)', 'topping(oregano)', 'topping(parmesan_cheese)', 'topping(parsley)', 'topping(peas)', 'topping(pecorino_cheese)', 'topping(pepperoni)', 'topping(peppers)', 'topping(pesto)', 'topping(pickles)', 'topping(pineapple)', 'topping(ranch_sauce)', 'topping(red_onions)', 'topping(red_pepper_flakes)', 'topping(red_peppers)', 'topping(ricotta_cheese)', 'topping(roasted_chicken)', 'topping(roasted_garlic)', 'topping(roasted_green_peppers)', 'topping(roasted_peppers)', 'topping(roasted_red_peppers)', 'topping(roasted_tomatoes)', 'topping(rosemary)', 'topping(salami)', 'topping(sauce)', 'topping(sausage)', 'topping(shrimps)', 'topping(spiced_sausage)', 'topping(spicy_red_sauce)', 'topping(spinach)', 'topping(tomato_sauce)', 'topping(tomatoes)', 'topping(tuna)', 'topping(vegan_pepperoni)', 'topping(white_onions)', 'topping(yellow_peppers)', 'vendor(bertucci)', 'vendor(blaze_restaurant)', 'vendor(dominos)', 'vendor(papa_john)', 'vendor(pizza_hut)', 'vendor(zeeks_restaurant)', 'vendor(zpizza_restaurant)', 'volume(1 liter)', 'volume(12 floz)', 'volume(16 oz)', 'volume(16.9  floz)', 'volume(2 liter)', 'volume(20 floz)', 'volume(200 ml)', 'volume(3 liter)', 'volume(500 ml)', 'volume(7.5 floz)', 'volume(8 oz)']
pizza_items = [x.lower() for x in pizza_items]
# Load the existing JSON data from a file
with open('FoodOrderingDataset/data/pizza/processed_train.json', 'r') as file:
    data = json.load(file)

# Modify the keys and add new key-value pair
for item in data:  # Assuming the data is a list of dictionaries
    item['input'] = item.pop('SRC', None)  # Replace 'SRC' with 'input'
    item['output_generate'] = item.pop('TOPALIAS', None)
    item['output_extract'] = [x.lower() for x in item.pop('USED_ITEMS', [])]  # Replace 'USED_ITEMS' with 'output' and convert to lower
    item['instruction_extract'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of all the corresponding items that you find in the order. You should use only items from the following list:\n"
        f'{pizza_items}'
    )
    item['instruction_generate'] = (
        "You are a helpful assistant. You have to take as input a customer order "
        "and output a list of the corresponding objects. You should use only the following classes in Python:\n"
        "class Topping:\n"
        "    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:\n"
        "\n"
        "class PizzaOrder:\n"
        "    def __init__(self, number: int = 1, size: Optional[str] = None, style: Optional[str] = None, "
        "toppings: Optional[List[Topping]] = None) -> None:\n"
        "\n"
        "class DrinkOrder:\n"
        "    def __init__(self, number: int = 1, size: Optional[str] = None, volume: Optional[str] = None, "
        "drink_type: Optional[str] = None, container_type: Optional[str] = None) -> None:\n"
        "\n"
        "The output should be a list of those objects.\n"
        "Here's an example:\n"
        "'input': 'three large pizzas with pecorino cheese and without tuna',\n"
        "'output': '[PizzaOrder(number=3, size=\"large\", toppings=[Topping(name=\"pecorino_cheese\"), Topping(name=\"tuna\", negation=True)])]',"
    )

# Save the modified data to a new file
with open('FoodOrderingDataset/processed_data/train_pizza_dataset.json', 'w') as file:
    json.dump(data, file, indent=4)
