# %%
import time
import guidance
from guidance import models, gen, one_or_more, select, zero_or_more, regex, optional, capture
from llama_cpp import Llama

model_name = "your_model_name"

model = models.LlamaCpp(f"{model_name}.gguf", n_gpu_layers=-1, n_ctx=1024)


# %%
@guidance(stateless=False)
def drinkOrderCoffee(lm):
    lm += "DrinkOrder("
    lm += select(["number="+regex("\d+"), ""], name='numberFlag')
    if drink_type_values:
      lm += select([", drink_type='" + select(drink_type_values, name='drinkTypeName')+"'", ""], name='drinkTypeFlag')
      if lm['drinkTypeFlag'] != "":
        drink_type_values.remove(lm['drinkTypeName'])

    if roast_type_values:
      lm += select([", roast_type='"+select(roast_type_values, name='roastTypeName')+"'", ""], name='drinkTypeFlag')
      if lm['drinkTypeFlag'] != "":
        roast_type_values.remove(lm['roastTypeName'])

    if size_values:
      lm += select([", size='"+select(size_values, name='sizeName')+"'", ""], name='sizeFlag')
      if lm['sizeFlag'] != "":
        size_values.remove(lm['sizeName'])

    if style_values:
      lm += select([", style='"+select(style_values, name='styleName')+"'", ""], name='styleFlag')
      if lm['styleFlag'] != "":
        style_values.remove(lm['styleName'])

    if topping_values:
      lm += select([", toppings=[", ""], name='toppingsFlag')
      if lm['toppingsFlag']:
        for i in topping_values[:]:
          lm += toppingCoffee()
          if not topping_values:
            lm += ']'
            break
          lm += select([", ", "]"], name="finishedListToppings")
          if lm['finishedListToppings'] == "]":
            break



    return lm + ")"

@guidance(stateless=False)
def toppingCoffee(lm):
  lm += "Topping(name="
  if topping_values:
    lm += "'" + select(topping_values, name='toppingName') + "'"
    topping_values.remove(lm['toppingName'])

  if quantity_values:
    lm += select([", qualifier='" + select(quantity_values, name='qualifierName') + "'", ""], name='qualifierFlag')
    if lm["qualifierFlag"] != "":
      quantity_values.remove(lm['qualifierName'])

  if not_values:
    lm += select([", negation=True", ""], name='negationFlag')
    if lm['negationFlag'] != "":
      not_values.remove('not')

  lm += ")"
  return lm

@guidance(stateless=False)
def validOrderCoffee(lm):
  lm += "["
  first = True
  for i in range(7):
    if drink_type_values:
      if not first:
        lm += select(", ", "")
      else:
        first = False

      lm += drinkOrderCoffee()
    else:
      break
  return lm +']'


# %%
instruction_generate_coffee = """You are a helpful assistant. You have to take as input a customer order and output a list of the corresponding objects. You should use only the following classes in Python:
class Topping:
      def __init__(self, name: str, qualifier: Optional[str] = None, negation: Optional[bool] = False) -> None:

class DrinkOrder:
      def __init__(self, number: int = 1, drink_type: Optional[str] = None, size: Optional[str] = None, style: Optional[str] = None, roast_type: Optional[str] = None, toppings: Optional[List[Topping]] = None) -> None:

The output should be a list of those objects."""

import json
import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import os

# %%
def parse_line(line):
    parts = line.strip().split('\t')
    if len(parts) != 2:
        return None, None

    phrase, category = parts

    return phrase, category.strip()

def init_pipeline(dataset = "coffee"):
  nlp = spacy.load("en_core_web_sm")
  ner_ruler = nlp.add_pipe("entity_ruler",
                      before="ner",
                      config={"phrase_matcher_attr": "LOWER"})

  def read_file_categories(food_type):
      file_path = f"FoodOrderingDataset/data/{food_type}/alias"

      text_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]

      patterns = []

      for file in text_files:
          path_to_file = f"{file_path}/{file}"
          with open(path_to_file, 'r') as file:
              for line in file:
                  if line.strip():
                      phrase, category_info = parse_line(line)
                      if phrase and category_info:
                          schema = {}
                          schema["pattern"] = phrase
                          schema["label"] = category_info
                          patterns.append(schema)

      return patterns

  category_patterns = read_file_categories(dataset)

  ner_ruler.add_patterns(category_patterns)
  return nlp


nlp = init_pipeline()

def process_NER(input_order):
    found_categories = []

    doc = nlp(input_order)

    for ent in doc.ents:
        found_categories.append((ent.text, ent.label_))

    return found_categories


# %%
import json
from collections import defaultdict

existing_data = []

file_path = f'FoodOrderingDataset/output/{model_name}-coffee-NER.json'

existing_data=[]

with open('FoodOrderingDataset/processed_data/coffee_dataset.json', 'r') as file:
    data = json.load(file)

input_list = []
ner_times = []
for obj in data:
    start_time = time.perf_counter()
    input_value = obj.get("input", "No input key found")
    output_value = obj.get("output_extract", "No output key found")
    output_generate = obj.get("output_generate", "No output key found")
    used_items_value = process_NER(input_value)
    used_items_value_decoupled = [x[0] + ' - ' + x[1] for x in used_items_value]
    used_items_str = ', '.join(used_items_value_decoupled).lower()

    input_augmented_file = input_value + "\nItems Found: " + used_items_str
    input_list.append((input_value, input_augmented_file, output_generate, used_items_value, used_items_str))
    end_time = time.perf_counter()
    ner_times.append(end_time - start_time)

import numpy as np
print(f"Average NER Time: {np.mean(ner_times)}")

import random
import re

random_indices = [15, 2, 3, 4, 5, 6, 7, 10, 11, 19, 29]

tokens = []
total_times = []
tpss = []
kv_cache = 0



for i in random_indices:
    (initial_input, input_augmented, expected, used_items_value, used_items_str) = input_list[i]
    
    lm = model + f"""\ 
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction_generate_coffee}
    ### Input:
    {input_augmented}

    ### Response:
    """
    items = []
    for item in used_items_value:
      match = re.search(r'(\w+)\(([^)]+)\)', item[1])
      if match:
        items.append((match.group(1), match.group(2)))
    items_dict = defaultdict(list)
    for item_type, item_value in items:
        items_dict[item_type.lower()].append(item_value.lower())

    items_dict = dict(items_dict)
    keys_to_extract = ['topping', 'size', 'number', 'drink_type', 'roast_type', 'not', 'style', 'quantity']

    for key in keys_to_extract:
        globals()[f"{key}_values"] = items_dict.get(key, [])
    
    start_time = time.perf_counter()

    ans = lm + gen(max_tokens=1)

    end_time = time.perf_counter()

    if kv_cache == 1:
        print(f'TTFT w/ KV: {end_time - start_time}')
        break
        
    else:
        print(f'Model Name: {model_name}')
        print(f'TTFT w/o KV: {end_time - start_time}')
        kv_cache = 1

print('eof')
