# %%
import time
import guidance
from guidance import models, gen, one_or_more, select, zero_or_more, regex, optional, capture

model_name = "your_model_name"

model = models.LlamaCpp(f"{model_name}.gguf", n_gpu_layers=-1, n_ctx=1024)


# %%
@guidance(stateless=False)
def burgerOrder(lm):
    lm += 'number=' + select(number_values + ['1'], name='numberNames')
    if lm['numberNames'] != '1':
        number_values.remove(lm['numberNames'])
    elif '1' in number_values:
        number_values.remove(lm['numberNames'])
    
    if main_dish_type_values:
        lm += select(
            [", main_dish_type='" + select(main_dish_type_values, name="burgerTypeName") + "'", ""],
            name='burgerTypeFlag'
        )
        if lm['burgerTypeFlag'] != "":
            main_dish_type_values.remove(lm['burgerTypeName'])
    
    if topping_values:
        lm += select([", toppings=[", ""], name='toppingsFlag')
    
        if lm['toppingsFlag'] != "":
            for i in topping_values[:]:
                lm += topping()
                if not topping_values:
                    lm += ']'
                    break
                lm += select([", ", "]"], name="finishedListToppings")
                if lm['finishedListToppings'] == "]":
                    break
    
    return lm + ")"


@guidance(stateless=False)
def topping(lm):
    lm += "Topping(name="
    if topping_values:
        lm += "'" + select(topping_values, name='toppingName') + "'"
        topping_values.remove(lm['toppingName'])
    
    if quantity_values:
        lm += select(
            [", qualifier='" + select(quantity_values, name="qualifierName") + "'", ""],
            name='qualifierFlag'
        )
        if lm["qualifierFlag"] != "":
            quantity_values.remove(lm['qualifierName'])
    
    if not_values:
        lm += select([", negation=True", ""], name='negationFlag')
        if lm['negationFlag']:
            not_values.remove('not')
    
    lm += ")"
    return lm


@guidance(stateless=False)
def drinkOrder(lm):
    lm += 'number=' + select(number_values + ['1'], name='numberNames')
    if lm['numberNames'] != '1':
        number_values.remove(lm['numberNames'])
    elif '1' in number_values:
        number_values.remove(lm['numberNames'])
    
    if drink_type_values:
        lm += select(
            [", drink_type='" + select(drink_type_values, name="drinkTypeName") + "'", ""],
            name='drinkTypeFlag'
        )
        if lm["drinkTypeFlag"] != "":
            drink_type_values.remove(lm['drinkTypeName'])
    
    if drink_size_values:
        lm += select(
            [", size='" + select(drink_size_values, name='drinkSizeName') + "'", ""],
            name='drinkSizeFlag'
        )
        if lm["drinkSizeFlag"] != "":
            drink_size_values.remove(lm['drinkSizeName'])
    
    return lm + ")"


@guidance(stateless=False)
def sideOrder(lm):
    lm += 'number=' + select(number_values + ['1'], name='numberNames')
    if lm['numberNames'] != '1':
        number_values.remove(lm['numberNames'])
    elif '1' in number_values:
        number_values.remove(lm['numberNames'])
    
    if side_type_values:
        lm += select(
            [", side_type='" + select(side_type_values, name='sideTypeName') + "'", ""],
            name='sideTypeFlag'
        )
        if lm['sideTypeFlag'] != '':
            side_type_values.remove(lm['sideTypeName'])
    
    if side_size_values:
        lm += select(
            [", size='" + select(side_size_values, name='sideSizeName') + "'", ""],
            name='sideSizeFlag'
        )
        if lm['sideSizeFlag'] != '':
            side_size_values.remove(lm['sideSizeName'])
    return lm + ")"


@guidance(stateless=False)
def validOrderBurger(lm):
    lm += "["
    first = True
    for i in range(7):
        choices = []
        if main_dish_type_values:
            choices.append("BurgerOrder(")
        if drink_type_values:
            choices.append("DrinkOrder(")
        if side_type_values:
            choices.append("SideOrder(")
    
        if choices:
            if not first:
                lm += ", "
            else:
                first = False
    
            lm += select(choices, name='choice')
            if lm['choice'] == "BurgerOrder(":
                lm += burgerOrder()
            elif lm['choice'] == "SideOrder(":
                lm += sideOrder()
            else:
                lm += drinkOrder()
        else:
            break
    lm += "]"
    return lm


# %%
instruction_generate_burger = """You are a helpful assistant. You have to take as input a customer order and output a list of the corresponding objects. You should use only the following classes in Python:
      class Topping:
            def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:
            
      class BurgerOrder:
            def __init__(self, number: int = 1, size: Optional[str] = None, main_dish_type: Optional[str] = None, toppings: Optional[List[Topping]] = None) -> None
       
      class DrinkOrder:
            def __init__(self, number: int = 1, drink_type: Optional[str] = None, size: Optional[str] = None) -> None :
      
      class SideOrder:
            def __init__(self, number: int = 1, side_type: Optional[str] = None, size: Optional[str] = None) -> None :
      
      The output should be a list of those objects.\n"
      Here's an example:
      'input': 'i would like a vegan burger with lettuce tomatoes and onions and a large order of sweet potato fries',
      'output': '[BurgerOrder(number=1, main_dish_type='vegan_burger', toppings=[Topping(name='lettuce'), Topping(name='tomato'), Topping(name='onion')]), SideOrder(number=1, side_type='french_fries', size='large')]',"""

# %%
import json
import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import os

def parse_line(line):
    parts = line.strip().split('\t')
    if len(parts) != 2:
        return None, None
    phrase, category = parts
    return phrase, category.strip()

def get_all_ngrams(tokens, max_n):
    ngrams = set()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join([token.text for token in tokens[i:i + n]])
            ngrams.add(ngram)
    return ngrams

def init_pipeline(dataset="burger"):
    nlp = spacy.load("en_core_web_sm")
    
    ner_ruler = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
    
    def read_file_categories(food_type):
        file_path = f"FoodOrderingDataset/data/{food_type}/alias"
        text_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        patterns = []
        for file in text_files:
            path_to_file = os.path.join(file_path, file)
            with open(path_to_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        phrase, category_info = parse_line(line)
                        if phrase and category_info:
                            schema = {"pattern": phrase, "label": category_info}
                            patterns.append(schema)
        return patterns

    category_patterns = read_file_categories(dataset)
    ner_ruler.add_patterns(category_patterns)
    
    global DRINK_KEYWORDS, SIDE_KEYWORDS
    DRINK_KEYWORDS = {pat["pattern"].lower() for pat in category_patterns if pat["label"].startswith("DRINK_TYPE")}
    SIDE_KEYWORDS = {pat["pattern"].lower() for pat in category_patterns if pat["label"].startswith("SIDE_TYPE")}
    
    nlp.add_pipe("disambiguate_size", after="ner")
    
    return nlp

@Language.component("disambiguate_size")
def disambiguate_size(doc):
    new_ents = []
    
    max_drink = max((len(phrase.split()) for phrase in DRINK_KEYWORDS), default=1)
    max_side = max((len(phrase.split()) for phrase in SIDE_KEYWORDS), default=1)
    max_n = max(max_drink, max_side)
    
    for ent in doc.ents:
        if "SIZE" in ent.label_:
            start = max(0, ent.start)
            end = min(len(doc), ent.end + 0)
            context_tokens = []
            while end <= len(doc):
                context_tokens = [token for token in doc[start:end]]
                context_ngrams = get_all_ngrams(context_tokens, max_n)
                if context_ngrams & DRINK_KEYWORDS:
                    new_ent = spacy.tokens.Span(doc, ent.start, ent.end, label="DRINK_" + ent.label_)
                    new_ents.append(new_ent)
                    break
                elif context_ngrams & SIDE_KEYWORDS:
                    new_ent = spacy.tokens.Span(doc, ent.start, ent.end, label="SIDE_" + ent.label_)
                    new_ents.append(new_ent)
                    break
                else:
                    if end == len(doc) and start == ent.start:
                        start -= 1
                    else:
                        end += 1
        else:
            new_ents.append(ent)
    doc.ents = tuple(new_ents)
    return doc

def process_NER(input_order):
    found_categories = []
    doc = nlp(input_order)
    for ent in doc.ents:
        found_categories.append((ent.text, ent.label_))
    return found_categories

nlp = init_pipeline()


# %%
import json
from collections import defaultdict

existing_data = []

with open('FoodOrderingDataset/processed_data/burger_dataset_disambiguation.json', 'r') as file:
    data = json.load(file)

input_list = []
for obj in data:
    ner_start = time.time()
    input_value = obj.get("input", "No input key found")
    output_value = obj.get("output_extract", "No output key found")
    output_generate = obj.get("output_generate", "No output key found")
    
    used_items_value = process_NER(input_value)
    used_items_value_decoupled = [f"{ent_text} - {ent_label}" for ent_text, ent_label in used_items_value]
    used_items_str = ', '.join(used_items_value_decoupled).lower()

    input_augmented_file = input_value + "\nItems Found: " + used_items_str
    input_list.append((input_value, input_augmented_file, output_generate, used_items_value, used_items_str))

import random

random_indices = random.sample(range(len(input_list)), 10)

for i in random_indices:
    (initial_input, input_augmented, expected, used_items_value, used_items_str) = input_list[i]
    
    lm = model + f"""\ 
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction_generate_burger}
    ### Input:
    {input_augmented}

    ### Response:
    """

    ttft_simple_start = time.perf_counter()
    temp = lm + select(['a'])
    ttft_simple_end = time.perf_counter()
    ttft_simple = ttft_simple_end - ttft_simple_start
    print(f'Complex time: {ttft_simple}')

    ttft_complex_start = time.perf_counter()
    temp2 = model + 'Joke: ' + select(['a'])
    ttft_complex_end = time.perf_counter()
    ttft_complex = ttft_complex_end - ttft_complex_start
    print(f'Simple time: {ttft_complex}')
    print(f'Difference: {ttft_simple - ttft_complex}')
    items = []
    for ent_text, ent_label in used_items_value:
        if "(" in ent_label:
            base_label = ent_label.split("(")[0].lower()
            canonical_text = ent_label.split("(")[1].lower()[:-1]
        else:
            continue
        
        items.append((base_label, canonical_text))
    
    items_dict = defaultdict(list)
    for item_type, item_value in items:
        items_dict[item_type].append(item_value)

    keys_to_extract = ['drink_size', 'side_size', 'main_dish_type', 'topping', 'quantity', 'not', 'drink_type', 'side_type', 'number']
    
    for key in keys_to_extract:
        globals()[f"{key}_values"] = items_dict.get(key, [])
    
    break

print('eof')