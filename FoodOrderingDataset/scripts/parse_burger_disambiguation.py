import json
import os
import re
from fuzzywuzzy import process
from typing import Optional, List

##########################################
# Data Model Classes
##########################################

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
        return f"BurgerOrder({', '.join(args)})"


class SideOrder:
    def __init__(self, number: int = 1, side_type: Optional[str] = None, size: Optional[str] = None) -> None:
        self.number: int = number
        self.side_type: Optional[str] = side_type
        self.size: Optional[str] = size

    def __repr__(self) -> str:
        args = []
        if self.number != 1:
            args.append(f"number={self.number}")
        if self.side_type:
            args.append(f"side_type='{self.side_type}'")
        if self.size:
            args.append(f"size='{self.size}'")
        return f"SideOrder({', '.join(args)})"


class DrinkOrder:
    def __init__(self, number: int = 1, drink_type: Optional[str] = None, size: Optional[str] = None) -> None:
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
        return f"DrinkOrder({', '.join(args)})"


##########################################
# Helper Functions
##########################################

def split_balanced_parentheses(data: str) -> List[str]:
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
        if balance == 0:
            parts.append(''.join(current))
            current = []
    return parts


def load_aliases() -> List[tuple]:
    """
    Load aliases from files in the FoodOrderingDataset/data/burger/alias directory.
    Each line is expected to be tab-separated: phrase<TAB>category.
    The category is expected to be in a format like DRINK_TYPE(something) or SIZE(something).
    Returns a list of tuples: (phrase, full_category, category_name, value).
    """
    directory = 'FoodOrderingDataset/data/burger/alias'
    aliases = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    if '\t' in line:
                        parts = line.split('\t')
                        phrase = parts[0].strip()
                        category = parts[1].strip()
                        # Extract the part before and inside the parentheses
                        category_name = category.split('(')[0]
                        value = category.split('(')[1][:-1] if '(' in category else ""
                        aliases.append((phrase, category, category_name, value))
    return aliases


def fuzzy_match(aliases: List[tuple], term: str, intent: str) -> Optional[str]:
    """
    Given a list of alias tuples, find the best match for 'term' with the specified intent.
    Returns the canonical form if found; otherwise, returns the original term.
    """
    matching_aliases = [(alias[0], alias[2], alias[3]) for alias in aliases if alias[2].lower() == intent.lower()]
    if not matching_aliases:
        return term
    choices = [alias[2] for alias in matching_aliases]
    best_match = process.extractOne(term, choices, score_cutoff=80)
    if best_match:
        canonical_form = next((alias[2] for alias in matching_aliases if alias[2] == best_match[0]), term)
        return canonical_form
    return term


# Load aliases once globally.
aliases = load_aliases()
option_list = set()
for option in aliases:
    option_list.add(option[1])
print("Aliases loaded:")
print(aliases)
print("Unique category strings:")
print(sorted(list(option_list)))


##########################################
# Main Parsing Function
##########################################

def parse_topalias(query: str) -> (str, List[str]):
    """
    Parse a topalias string and extract orders and used items.
    This version looks for size tokens labeled DRINK_SIZE, SIDE_SIZE, or generic SIZE.
    Returns a string representation of orders and a list of used item strings.
    """
    used_items = []
    orders = []
    # Pattern to match order types at the end of the line.
    order_pattern = re.compile(r'\((\w+)_ORDER (.*?)\)$', re.S)
    
    for topalias in split_balanced_parentheses(query):
        if topalias.strip():
            matches = re.findall(order_pattern, topalias)
            for order_type, details in matches:
                # Extract NUMBER
                number = re.search(r'\(NUMBER (\w+) \)', details)
                number = number.group(1) if number else "1"
                number = fuzzy_match(aliases, number, 'number')
                used_items.append(f'NUMBER({number})')

                # Extract size tokens.
                # Look for DRINK_SIZE, SIDE_SIZE, or generic SIZE.
                size = None
                drink_size = None
                side_size = None
                size_match = re.search(r'\((DRINK_SIZE|SIDE_SIZE|SIZE) ([^\)]+) \)', details)
                if size_match:
                    size_label = size_match.group(1)
                    size_value = size_match.group(2)
                    size_value = fuzzy_match(aliases, size_value, 'size')
                    if size_label.upper() == 'DRINK_SIZE':
                        used_items.append(f'DRINK_SIZE({size_value})')
                        drink_size = size_value
                    elif size_label.upper() == 'SIDE_SIZE':
                        used_items.append(f'SIDE_SIZE({size_value})')
                        side_size = size_value
                    else:
                        used_items.append(f'SIZE({size_value})')
                        size = size_value

                # Extract VOLUME
                volume = re.search(r'\(VOLUME ([^\)]+) \)', details)
                volume = volume.group(1) if volume else None
                if volume:
                    volume = fuzzy_match(aliases, volume, 'volume')
                    used_items.append(f'VOLUME({volume})')

                # Extract DRINK_TYPE
                drink_type = re.search(r'\(DRINK_TYPE ([^\)]+) \)', details)
                drink_type = drink_type.group(1) if drink_type else None
                if drink_type:
                    drink_type = fuzzy_match(aliases, drink_type, 'drink_type')
                    used_items.append(f'DRINK_TYPE({drink_type})')

                # Extract SIDE_TYPE
                side_type = re.search(r'\(SIDE_TYPE ([^\)]+) \)', details)
                side_type = side_type.group(1) if side_type else None
                if side_type:
                    side_type = fuzzy_match(aliases, side_type, 'side_type')
                    used_items.append(f'SIDE_TYPE({side_type})')

                # Extract MAIN_DISH_TYPE
                main_dish_type = re.search(r'\(MAIN_DISH_TYPE ([^\)]+) \)', details)
                main_dish_type = main_dish_type.group(1) if main_dish_type else None
                if main_dish_type:
                    main_dish_type = fuzzy_match(aliases, main_dish_type, 'main_dish_type')
                    used_items.append(f'MAIN_DISH_TYPE({main_dish_type})')

                # Extract STYLE
                style = re.search(r'\(STYLE ([^\)]+) \)', details)
                style = style.group(1) if style else None
                if style:
                    style = fuzzy_match(aliases, style, 'style')
                    used_items.append(f'STYLE({style})')

                # Process toppings (both complex and simple patterns)
                toppings = []
                detail_copy = details
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
                    complex_expr = f'({"NOT " if neg else ""}(COMPLEX (QUANTITY {quantity}) (TOPPING {topping})))'
                    detail_copy = detail_copy.replace(complex_expr, '', 1)

                simple_patterns = re.findall(r'\(?\s*(NOT\s+)?\(TOPPING ([^)]+) \)\s*\)?', detail_copy)
                for neg, topping in simple_patterns:
                    if not any(t.name == topping for t in toppings):
                        negation = bool(neg)
                        if negation:
                            used_items.append(f'NOT(not)')
                        if topping:
                            topping = fuzzy_match(aliases, topping, 'topping')
                            used_items.append(f'TOPPING({topping})')
                        toppings.append(Topping(name=topping, negation=negation))

                # Create the corresponding order based on order_type.
                if order_type.upper() == "MAIN_DISH":
                    orders.append(BurgerOrder(number=number, main_dish_type=main_dish_type, toppings=toppings))
                elif order_type.upper() == "DRINK":
                    # For a drink order, prefer the disambiguated drink_size (or fallback to generic size).
                    orders.append(DrinkOrder(number=number, drink_type=drink_type, size=drink_size or size))
                elif order_type.upper() == "SIDE":
                    # For a side order, prefer the disambiguated side_size (or fallback to generic size).
                    orders.append(SideOrder(number=number, side_type=side_type, size=side_size or size))
    return orders.__str__(), used_items


def load_aliases_from_file(file_path: str) -> dict:
    alias_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                key, value = parts
                alias_dict[key] = value
    return alias_dict


def apply_aliases(topalias: str, aliases: dict) -> str:
    words = re.split('(\W+)', topalias)  # Split by non-word characters but keep delimiters.
    for i, word in enumerate(words):
        if word.lower() in aliases:
            words[i] = aliases[word.lower()]
        else:
            closest_match = process.extractOne(word.lower(), aliases.keys())
            if closest_match and closest_match[1] > 80:
                words[i] = aliases[closest_match[0]]
    return ''.join(words)


##########################################
# Main Function
##########################################

def main():
    input_file = 'FoodOrderingDataset/data/burger/dev.json'
    output_file = 'FoodOrderingDataset/data/burger/processed_dev_disambiguation.json'

    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())  # Parse each line individually.
            topalias = entry['EXR']
            # Filter out entries that contain '(NOT (STYLE ...))'
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
