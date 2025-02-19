import json
import os
import re
from fuzzywuzzy import process

from typing import Optional, List


class Topping:
    def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:
        self.name: str = name
        self.qualifier: Optional[str] = qualifier
        self.negation: Optional[bool] = negation

    def __repr__(self) -> str:
        args = [f"name='{self.name}'"]
        if self.qualifier:
            args.append(f"qualifier='{self.qualifier}'")
        if self.negation:
            args.append("negation=True")
      
        # args.append(f"qualifier='{self.qualifier}'")
        # if self.negation:
        #   args.append("negation=True")
        # else:
        #   args.append("negation=False")   
        return f"Topping({', '.join(args)})"



class BurgerOrder:
    def __init__(self, number: int = 1, size: Optional[str] = None, main_dish_type: Optional[str] = None,
                 toppings: Optional[List[Topping]] = None) -> None:
        self.number: int = number
        self.size: Optional[str] = size
        self.main_dish_type: Optional[str] = main_dish_type
        self.toppings: List[Topping] = toppings if toppings else []

    def __repr__(self) -> str:
        args = []
        if self.number != 1:
            args.append(f"number={self.number}")
        if self.size:
            args.append(f"size='{self.size}'")
        if self.main_dish_type:
            args.append(f"main_dish_type='{self.main_dish_type}'")
        if self.toppings:
            toppings_str = ', '.join(map(str, self.toppings))
            args.append(f"toppings=[{toppings_str}]")
        # args.append(f"number={self.number}")
        # args.append(f"size='{self.size}'")
        # args.append(f"main_dish_type='{self.main_dish_type}'")
        # toppings_str = ', '.join(map(str, self.toppings))
        # args.append(f"toppings=[{toppings_str}]")
        return f"BurgerOrder({', '.join(args)})"


class SideOrder:
    def __init__(self, number: int = 1,
                 side_type: str = None, size: Optional[str] = None) -> None:
        self.number: int = number
        self.side_type: str = side_type
        self.size: Optional[str] = size

    def __repr__(self) -> str:
        args = []
        if self.number != 1:
            args.append(f"number={self.number}")
        if self.side_type:
            args.append(f"side_type='{self.side_type}'")
        if self.size:
            args.append(f"size='{self.size}'")
        # args.append(f"number={self.number}")
        # args.append(f"side_type='{self.side_type}'")
        # args.append(f"size='{self.size}'")    
        return f"SideOrder({', '.join(args)})"


class DrinkOrder:
    def __init__(self, number: int = 1,
                 drink_type: Optional[str] = None, size: Optional[str] = None) -> None:
        self.number: int = number
        self.drink_type: Optional[str] = drink_type
        self.size: Optional[str] = size

    def __repr__(self) -> str:
        args = []
        if self.number != 1:
            args.append(f"number={self.number}")
        if self.drink_type:
            args.append(f"drink_type='{self.drink_type}'")
        if self.size:
            args.append(f"size='{self.size}'")
        # args.append(f"number={self.number}")
        # args.append(f"drink_type='{self.drink_type}'")
        # args.append(f"size='{self.size}'")
        return f"DrinkOrder({', '.join(args)})"


def split_balanced_parentheses(data):
    """
    Split a string based on balanced parentheses.

    :param data: A string starting with an open parenthesis '('.
    :return: A list of substrings split based on top-level balanced parentheses.
    """
    if not data:
        return []

    parts = []
    current = []
    balance = 0
    for char in data:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1

        current.append(char)

        # If we return to zero balance, we have completed a top-level segment
        if balance == 0:
            parts.append(''.join(current))
            current = []

    # The below condition is optional and can be included if there's a concern
    # that the input might not be balanced or if you want to capture any remaining characters.
    # if current:
    #     parts.append(''.join(current))

    return parts


def load_aliases():
    directory = 'FoodOrderingDataset/data/burger/alias'
    aliases = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            intent_file = filename.replace('.txt', '')
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    if '\t' in line:
                        aliases.append((line.split('\t')[0].strip(), line.split('\t')[1].strip(),
                                        line.split('\t')[1].strip().split('(')[0],
                                        line.split('\t')[1].strip().split('(')[1][:-1]))

    return aliases


def fuzzy_match(aliases, term, intent):
    matching_aliases = [(alias[0], alias[2], alias[3]) for alias in aliases if alias[2].lower() == intent.lower()]
    if not matching_aliases:
        return None  # If no aliases match the intent, return None

    # Use fuzzy matching to find the closest match to the term
    # Extract just the alias terms (actual strings to match against)
    choices = [alias[2] for alias in matching_aliases]
   
    best_match = process.extractOne(term, choices, score_cutoff=80)

    # If a good enough match is found
    if best_match:
        # Find the original alias tuple to get the canonical form
        canonical_form = next((alias[2] for alias in matching_aliases if alias[2] == best_match[0]), None)
        return canonical_form  # Return the canonical form associated with the best match

    return term  # Return None if no sufficient match is found


aliases = load_aliases()
option_list = set()
for option in aliases:
    option_list.add(option[1])
print(aliases)
print(sorted(list(option_list)))



def parse_topalias(query):
    used_items = []
    orders = []
    order_pattern = re.compile(r'\((\w+)_ORDER (.*?)\)$', re.S)
    for topalias in split_balanced_parentheses(query):
        if topalias.strip():
            matches = re.findall(order_pattern, topalias)
            for order_type, details in matches:
                number = re.search(r'\(NUMBER (\w+) \)', details)
                number = number.group(1) if number else "1"
                number = fuzzy_match(aliases, number, 'number')
                used_items.append(f'NUMBER({number})')

                size = re.search(r'\(SIZE ([^\)]+) \)', details)
                size = size.group(1) if size else None
                if size:
                    size = fuzzy_match(aliases, size, 'size')
                    used_items.append(f'SIZE({size})')

                volume = re.search(r'\(VOLUME ([^\)]+) \)', details)
                volume = volume.group(1) if volume else None
                if volume:
                    volume = fuzzy_match(aliases, volume, 'volume')
                    used_items.append(f'VOLUME({volume})')

                drink_type = re.search(r'\(DRINK_TYPE ([^\)]+) \)', details)
                drink_type = drink_type.group(1) if drink_type else None
                if drink_type:
                    drink_type = fuzzy_match(aliases, drink_type, 'drink_type')
                    used_items.append(f'DRINK_TYPE({drink_type})')

                side_type = re.search(r'\(SIDE_TYPE ([^\)]+) \)', details)
                side_type = side_type.group(1) if side_type else None
                if side_type:
                    side_type = fuzzy_match(aliases, side_type, 'side_type')
                    used_items.append(f'SIDE_TYPE({side_type})')

                main_dish_type = re.search(r'\(MAIN_DISH_TYPE ([^\)]+) \)', details)
                main_dish_type = main_dish_type.group(1) if main_dish_type else None
                if main_dish_type:
                    main_dish_type = fuzzy_match(aliases, main_dish_type, 'main_dish_type')
                    used_items.append(f'MAIN_DISH_TYPE({main_dish_type})')

                style = re.search(r'\(STYLE ([^\)]+) \)', details)
                style = style.group(1) if style else None
                if style:
                    style = fuzzy_match(aliases, style, 'style')
                    used_items.append(f'STYLE({style})')

                toppings = []
                # Work on a copy of the detail string to safely modify it
                detail_copy = details

                # Match complex structures with optional NOT
                complex_patterns = re.findall(
                    r'\(?\s*(NOT\s+)?\(COMPLEX \(QUANTITY ([^)]+)\) \(TOPPING ([^)]+) \)\s*\)\s*\)?', detail_copy)
                for neg, quantity, topping in complex_patterns:
                    negation = bool(neg)
                    if negation:
                        used_items.append(f'NOT(not)')
                    if topping:
                        topping = fuzzy_match(aliases, topping, 'topping')
                        used_items.append(f'TOPPING({topping})')
                    if quantity:
                        quantity = fuzzy_match(aliases, quantity, 'quantity')
                        used_items.append(f'QUANTITY({quantity})')
                    toppings.append(Topping(name=topping, qualifier=quantity, negation=negation))
                    # Remove the parsed complex structure from detail_copy
                    complex_expr = f'({"NOT " if neg else ""}(COMPLEX (QUANTITY {quantity}) (TOPPING {topping})))'
                    detail_copy = detail_copy.replace(complex_expr, '', 1)

                # Match simple toppings in the modified detail string, ensuring they aren't already included
                simple_patterns = re.findall(r'\(?\s*(NOT\s+)?\(TOPPING ([^)]+) \)\s*\)?', detail_copy)
                for neg, topping in simple_patterns:
                    if not any(t.name == topping for t in toppings):  # Check for duplicates
                        negation = bool(neg)
                        if negation:
                            used_items.append(f'NOT(not)')
                        if topping:
                            topping = fuzzy_match(aliases, topping, 'topping')
                            used_items.append(f'TOPPING({topping})')
                        toppings.append(Topping(name=topping, negation=negation))


                if order_type == "MAIN_DISH":
                    orders.append(BurgerOrder(number=number, main_dish_type=main_dish_type, toppings=toppings))
                elif order_type == "DRINK":
                    orders.append(DrinkOrder(number=number, drink_type=drink_type, size=size))
                elif order_type == "SIDE":
                    orders.append(SideOrder(number=number, side_type=side_type, size=size))

    return orders.__str__(), used_items


# Example usage
# example_data = [
#     "(PIZZAORDER (NUMBER three ) (SIZE large ) (COMPLEX (QUANTITY extra) (TOPPING pecorino cheese ) ) (NOT (TOPPING tuna ) ) )",
#     "(DRINKORDER (NUMBER four ) (DRINKTYPE seven ups ) ) (DRINKORDER (NUMBER five ) (VOLUME 500 ml ) (DRINKTYPE ice teas ) ) (DRINKORDER (NUMBER five ) (SIZE extra large ) (DRINKTYPE perriers ) )"
# ]
#
# for data in example_data:
#     print(parse_topalias(data))

def load_aliases(file_path):
    alias_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                key, value = parts
                alias_dict[key] = value
    return alias_dict


def apply_aliases(topalias, aliases):
    words = re.split('(\W+)', topalias)  # Split by non-word characters but keep delimiters
    for i, word in enumerate(words):
        if word.lower() in aliases:
            words[i] = aliases[word.lower()]
        else:
            # Fuzzy matching to find the closest alias if no exact match is found
            closest_match = process.extractOne(word.lower(), aliases.keys())
            if closest_match and closest_match[1] > 80:  # Threshold for fuzzy match quality
                words[i] = aliases[closest_match[0]]
    return ''.join(words)


def main():
    input_file = 'FoodOrderingDataset/data/burger/dev.json'
    output_file = 'FoodOrderingDataset/data/burger/processed_dev.json'

    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())  # Parse each line individually
            topalias = entry['EXR']
            # Filter entries that do not contain '(NOT (STYLE ...))'
            if '(NOT (STYLE' not in topalias:
                new_topalias, used_items = parse_topalias(topalias)
                processed_entry = {
                    "SRC": entry['SRC'],
                    "TOPALIAS": new_topalias,
                    "USED_ITEMS": used_items
                }
                processed_data.append(processed_entry)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, indent=4)


if __name__ == "__main__":
    main()
