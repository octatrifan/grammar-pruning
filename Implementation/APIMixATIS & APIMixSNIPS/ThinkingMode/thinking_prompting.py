import os
import json
import re
import time

from llama_cpp import Llama  # For local GGUF models

MODEL_PATH = "Qwen3-4B-BF16.gguf"
INPUT_FILE_NAME = "atis_data_augmented.json"
OUTPUT_FILE_NAME = "timetest"
N_GPU_LAYERS = -1
CTX_SIZE = 16000
MAX_TOKENS_GENERATION = 1000
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 30
REPEAT_PENALTY = 1.1
NUM_ENTRIES_TO_PROCESS = 0

VANILLA_DOMAIN_SYSTEM_INSTRUCTION = (
    "You are an expert task-oriented conversational model specialized in the ATIS (Air Travel Information System) domain. "
    "Your task is to generate a structured semantic representation of the user's entire request by parsing it into a list of one or more predefined function calls. "
    "Analyze the complete user query, identify all appropriate function(s), and populate their parameters with values extracted from the query.\n\n"

    "### Output Format:\n"
    "Your output must be a Python-style list of one or more function call strings. Each function call should be represented as a string in the following format:\n"
    "  - \"function_name(parameter1='value1', parameter2='value2', ...)\"\n"
    "If multiple functions are needed for the query, include each as a separate string within the list, in the order they appear to be intended by the user. For example:\n"
    "  User Query: list california airports , list la \n"
    "  Your Output: [\"atis_airport(state='california', city=None, modifier=None)\", \"atis_city(city='los angeles', airline=None)\"] \n"
    "For any slot parameter defined for a chosen function that is not mentioned or inferable from the user’s query, you MUST explicitly assign `slot_name=None` for that parameter in the output string.\n"
    "Ensure your entire output is ONLY this Python-style list of function call strings and nothing else (no explanations, no conversational text before or after the list).\n\n"

    "### Functions and Their Descriptions:\n\n"
    "1. atis_airport: Provides information about airports, including listing airports in specific locations.\n"
    "   - state: The state where the airport is located (e.g., 'california').\n"
    "   - city: The city where the airport is located (e.g., 'los angeles').\n"
    "   - modifier: A descriptive qualifier for the airport (e.g., 'closest').\n\n"
    "2. atis_city: Identifies or lists cities associated with specific airlines or flights.\n"
    "   - city: The name of the city (e.g., 'la').\n"
    "   - airline: The name of an airline that operates in the city (e.g., 'northwest').\n\n"
    "3. atis_quantity: how many / number of flights or other entities matching specific criteria.\n"
    "   - airline: The name of the airline (e.g., 'canadian airlines').\n"
    "   - aircraft: The type of aircraft (e.g., '320', 'dh8').\n\n"
    "4. atis_airfare: Provides information about and ticket availability and prices.\n"
    "   - departure_city: The starting city for the flight.\n"
    "   - destination_city: The destination city for the flight.\n"
    "   - class: The service class of the ticket (e.g., 'first class', 'coach').\n"
    "   - relative_cost: Price condition (e.g., 'under').\n"
    "   - fare: The fare value (e.g., '200 dollars').\n"
    "   - departure_day: The day of departure (e.g., 'saturday').\n"
    "   - airline_code: The code of the airline (e.g., 'twa').\n\n"
    "5. atis_flight_no: flight numbers matching the query criteria.\n"
    "   - departure_city: The starting city of the flight.\n"
    "   - destination_city: The destination city of the flight.\n"
    "   - airline: The name of the airline.\n"
    "   - relative_departure_time: Describes a time condition (e.g., 'before').\n"
    "   - departure_time: The exact departure time (e.g., '8 am').\n"
    "   - departure_day: The day of departure (e.g., 'thursday').\n\n"
    "6. atis_capacity: seating capacity of specific aircraft.\n"
    "   - aircraft: The model of the aircraft (e.g., '757', '733').\n"
    "   - airline: The name of the airline operating the aircraft.\n\n"
    "7. atis_distance: Provides the distance between two cities.\n"
    "   - departure_city: The starting city.\n"
    "   - destination_city: The destination city.\n\n"
    "8. atis_aircraft: Provides details about specific aircraft models or types, not about capacity.\n"
    "   - aircraft: The type of aircraft (e.g., 'm80').\n"
    "   - departure_city: The city of departure.\n"
    "   - destination_city: The city of destination.\n"
    "   - relative_departure_time: Describes a time condition (e.g., 'before').\n"
    "   - departure_period: The period of the day (e.g., 'noon').\n"
    "   - airline: The name of the airline operating the aircraft.\n"
    "   - departure_time: The exact departure time (e.g., '4:19pm').\n\n"
    "9. atis_day_name: Identifies the day of the week on which flights operate between two cities.\n"
    "   - departure_city: The starting city of the flight.\n"
    "   - destination_city: The destination city of the flight.\n\n"
    "10. atis_ground_service: Provides information about ground transportation options, not about price/cost\n"
    "    - transport_type: The type of ground transport (e.g., 'limousine', 'car').\n"
    "    - city: The city where the ground transportation is available.\n\n"
    "11. atis_abbreviation: Use of fare codes or airline codes or aircraft codes.\n"
    "    - fare_code: The code representing a fare type (e.g., 'q').\n"
    "    - airline_code: The code representing an airline (e.g., 'AA').\n"
    "    - aircraft: The type of aircraft (e.g., 'd9s').\n\n"
    "12. atis_meal: Provides information about meals available on specific flights.\n"
    "    - flight_number: The flight number (e.g., 'AA665').\n"
    "    - meal_type: The type of meal (e.g., 'meals').\n"
    "    - departure_day: The day of departure.\n"
    "    - departure_period: The time of day (e.g., 'morning').\n"
    "    - meal_description: The description of the meal (e.g., 'snacks').\n"
    "    - airline: The airline providing the meal.\n\n"
    "13. atis_flight: Provides general information about flights between cities.\n"
    "    - departure_city: The starting city of the flight.\n"
    "    - destination_city: The destination city of the flight.\n"
    "    - departure_month: The month of departure (e.g., 'april').\n"
    "    - departure_day_number: The day of the month (e.g., 'fifth').\n"
    "    - airline: The airline operating the flight.\n\n"
    "14. atis_ground_fare: How much it costs for ground transportation options.\n"
    "    - transport_type: The type of ground transport (e.g., 'limousine').\n"
    "    - city: The city where the ground fare applies.\n\n"
    "15. atis_airline: Provides information about airlines.\n"
    "    - airline_code: The code representing an airline (e.g., 'AA').\n"
    "    - departure_city: The starting city of the airline.\n"
    "    - destination_city: The destination city of the airline.\n\n"
    "16. atis_flight_time: Provides the flight/arrival/departure time for flights.\n"
    "    - departure_city: The starting city.\n"
    "    - destination_city: The destination city.\n"
    "    - flight_time: Whether it's the arrival or departure time (e.g., 'arrival' or 'departure').\n\n"
    # Placeholder for the schema hint with example values
    "### Slot Value Examples (for guidance, not exhaustive - prioritize query content):\n"
    "{schema_example_hint_string}\n\n"
    "### Processing Instructions for User Query:\n"
    "1. Identify all distinct intents mentioned by the user in the query provided below.\n"
    "2. For each identified intent, determine the correct function name from the list above.\n"
    "3. For each chosen function, extract all relevant slot values from the user query. Use the slot value examples above for guidance on expected value types, but prioritize values directly from the user query.\n"
    "4. For every slot defined for a chosen function, if its value is not found in the query, you MUST output `slot_name=None`.\n"
    "5. Format your entire response as a Python list of these function call strings. ONLY output this list."
)

# Your ATIS schema with example values
schema_with_example_values_atis = {
    "atis_airport": {"state": ["california", "oregon", "texas", "arizona", "florida"], "modifier": ["closest"],
                     "city": ["la", "ontario", "new york"]},
    "atis_city": {"city": ["portland", "la"], "airline": ["spirit", "lufthansa", "eastwest", "northeast", "northwest"]},
    "atis_quantity": {
        "airline": ["british airways", "spirit", "canadian airlines", "delta", "lufthansa", "american airlines",
                    "wizzair"], "aircraft": ["j32", "br31", "re89", "j31", "ca32", "320", "dh8", "dh9", "ca31", "j33"]},
    "atis_airfare": {
        "departure_city": ["indianapolis", "boston", "los angeles", "detroit", "chicago", "pittsburgh", "toronto",
                           "nashville", "washington", "columbus"],
        "destination_city": ["seattle", "cleveland", "moscow", "las vegas", "st. petersburg", "st. louis", "chicago",
                             "atlanta", "washington", "memphis", "san jose", "san diego", "tacoma", "montreal"],
        "airline_code": ["twa"], "class": ["first class", "coach"], "relative_cost": ["under"],
        "fare": ["200 dollars", "300 pounds", "100 australian dollars"], "departure_day": ["saturday"]},
    "atis_flight_no": {
        "departure_city": ["houston", "dallas", "cleveland", "phoenix", "chicago", "oakland", "nashville"],
        "destination_city": ["houston", "seattle", "dallas", "boston", "salt lake city", "milwaukee", "tacoma"],
        "airline": ["continental", "american airlines", "british airways"], "relative_departure_time": ["before"],
        "departure_time": ["8 am"], "departure_day": ["thursday"]},
    "atis_distance": {"departure_city": ["dallas", "boston", "san francisco", "los angeles", "toronto", "baltimore"],
                      "destination_city": ["dallas", "san francisco", "los angeles", "moscow", "new york", "ottawa",
                                           "toronto"]},
    "atis_aircraft": {"aircraft": ["m81", "m80"], "departure_city": ["cleveland"], "destination_city": ["dallas"],
                      "relative_departure_time": ["before"], "departure_period": ["noon"], "airline": ["american"],
                      "departure_time": ["4:19pm"]},
    "atis_day_name": {"departure_city": ["san jose", "los angeles", "moscow", "nashville"],
                      "destination_city": ["portland", "dallas", "new york", "nashville", "tacoma"]},
    "atis_ground_service": {"transport_type": ["car", "limousine"],
                            "city": ["portland", "dallas", "burbank", "porto", "los angeles", "new york", "las vegas",
                                     "berlin", "st. louis", "salt lake city", "fort worth", "denver", "milwaukee",
                                     "washington", "tampa"]},
    "atis_abbreviation": {"fare_code": ["qo", "m", "l", "f", "g", "q"], "airline_code": ["uat", "ua"],
                          "aircraft": ["d9s"]},
    "atis_meal": {"flight_number": ["BA665", "AA811", "BA123", "BA354", "BA121", "LH468", "AA665", "TK382", "TK12"],
                  "meal_type": ["meals", "drinks"], "departure_day": ["tuesday"],
                  "departure_period": ["evening", "morning"], "meal_description": ["snacks"],
                  "airline": ["wizzair", "tower air"]},
    "atis_flight": {"departure_city": ["cleveland", "los angeles", "pittsburgh", "memphis", "montreal"],
                    "destination_city": ["indianapolis", "burbank", "cleveland", "long beach", "milwaukee", "san jose"],
                    "departure_month": ["april"], "departure_day_number": ["fifth"], "airline": ["alaska airlines"]},
    "atis_capacity": {"aircraft": ["757", "73s", "733", "m80", "747", "be1"], "airline": ["delta"],
                      "departure_city": ["seattle"], "destination_city": ["salt lake city"]},
    "atis_flight_time": {"departure_city": ["la", "michigan", "detroit"],
                         "destination_city": ["san francisco", "westchester", "manchester", "new york", "michigan"],
                         "flight_time": ["departure", "arrival"]},
    "atis_ground_fare": {"transport_type": ["limousine"],
                         "city": ["boston", "bucharest", "los angeles", "moscow", "denver", "michigan", "san diego"]},
    "atis_airline": {"airline_code": ["aa", "as", "BA", "us", "sr", "hp", "pr", "rp", "DA"],
                     "departure_city": ["seattle", "washington", "toronto"],
                     "destination_city": ["st. louis", "salt lake city", "san diego", "columbus"]}
}

ALL_INTENT_SLOTS_ATIS_FROM_EXAMPLES = {
    intent: list(slots.keys()) for intent, slots in schema_with_example_values_atis.items()
}


def load_llm(model_path: str, n_gpu_layers: int, n_ctx: int):
    print(f"Loading GGUF model from: {model_path}")
    print(f"GPU layers: {n_gpu_layers}, Context size: {n_ctx}")
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
        print("Model loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading GGUF model from {model_path}: {e}")
        return None


def get_llm_completion(
        llm: Llama,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        max_tokens: int
):
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=["User:", "### Input:", "\n\n\n", "Output:", "Human:", "Assistant:",
                  "### Processing Instructions for User Query:"]  # Added more stop tokens
        )
        generated_text = output['choices'][0]['text'].strip()
        return generated_text
    except Exception as e:
        print(f"Error during LLM completion: {e}")
        return ""


def format_schema_hint(example_schema_data: dict, all_intent_slots_for_domain: dict) -> str:
    schema_hint_str = ""
    if example_schema_data:
        schema_hint_parts = []
        for intent_name, slots_data in example_schema_data.items():
            if intent_name in all_intent_slots_for_domain:  # Check if intent is known
                slot_examples_strs = []
                for slot_name, ex_values in slots_data.items():
                    if ex_values and isinstance(ex_values, list) and ex_values[0] is not None:
                        # Show only first 1-2 examples for brevity
                        display_examples = [f"'{v}'" for v in ex_values[:2]]
                        slot_examples_strs.append(
                            f"{slot_name} (e.g., {', '.join(display_examples)}{'...' if len(ex_values) > 2 else ''})")
                if slot_examples_strs:
                    schema_hint_parts.append(f"  For Intent '{intent_name}': {'; '.join(slot_examples_strs)}")

        if schema_hint_parts:
            schema_hint_str = "\n".join(schema_hint_parts[:7])  # Show hints for first few intents
            if len(schema_hint_parts) > 7:
                schema_hint_str += "\n  ... (more examples available but not listed for brevity)"
    return schema_hint_str if schema_hint_str else "No specific slot value examples provided."


def get_vanilla_parsed_output(
        llm: Llama,
        user_query: str,
        base_domain_instruction: str,
        example_schema_data: dict,  # Your schema with example values
        all_intent_slots_for_domain: dict,  # To help format the schema hint
        generation_cfg: dict
) -> list[str]:
    schema_hint = format_schema_hint(example_schema_data, all_intent_slots_for_domain)

    final_domain_instruction = base_domain_instruction.replace(
        "{schema_example_hint_string}",
        schema_hint
    )

    prompt = (
        f"{final_domain_instruction}\n\n"
        "User Query: \"{user_query}\"\n\n"
        "Your Output (must be a Python list of function call strings as specified above, ensure all slots for chosen functions are present, using slot_name=None if not in query):\n"
    ).format(user_query=user_query)

    raw_llm_output = get_llm_completion(llm, prompt, **generation_cfg)
    print(f"  Raw LLM Output for Vanilla Prompt: {raw_llm_output}")

    parsed_calls = []
    # Attempt to extract content if it's wrapped in list-like brackets
    list_content_match = re.search(r"^\s*\[(.*)\]\s*$", raw_llm_output, re.DOTALL)

    content_to_parse = raw_llm_output
    if list_content_match:
        content_to_parse = list_content_match.group(1).strip()

    found_calls = re.findall(r"([a-zA-Z0-9_]+\s*\(.*?\))", content_to_parse)

    if found_calls:
        for call_str in found_calls:
            call_str = call_str.strip()
            if re.fullmatch(r"[a-zA-Z0-9_]+\s*\((.*)\)", call_str):  # Validate basic structure
                parsed_calls.append(call_str)
            else:
                print(f"    Warning: Found potential call '{call_str}' but it failed validation.")
    elif re.fullmatch(r"[a-zA-Z0-9_]+\s*\((.*)\)", raw_llm_output):  # Check if the whole output is a single call
        parsed_calls.append(raw_llm_output)
    else:
        print(f"    Warning: LLM output does not look like a Python list of calls or a single call: '{raw_llm_output}'")
        messy_calls = re.findall(r"([a-zA-Z0-9_]+\s*\(.*?\))", raw_llm_output)
        if messy_calls:
            print(f"    Found {len(messy_calls)} potential calls in messy output (fallback).")
            for call_str in messy_calls:
                call_str = call_str.strip()
                if re.fullmatch(r"[a-zA-Z0-9_]+\s*\((.*)\)", call_str):
                    parsed_calls.append(call_str)

    if not parsed_calls and raw_llm_output:
        print(f"    Could not parse any valid calls from raw output: {raw_llm_output}")

    return parsed_calls


def main():
    llm = load_llm(MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, n_ctx=CTX_SIZE)
    if not llm:
        print("Failed to load LLM. Exiting.")
        return

    generation_config = {
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY, "max_tokens": MAX_TOKENS_GENERATION
    }
    results_log = []

    print(f"Loading data from: {INPUT_FILE_NAME}")
    try:
        with open(INPUT_FILE_NAME, "r", encoding="utf-8") as f:
            actual_data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(actual_data)} entries.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {INPUT_FILE_NAME}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from file: {INPUT_FILE_NAME}. Ensure it's a valid JSONL file.")
        return

    data_to_process = actual_data
    if NUM_ENTRIES_TO_PROCESS > 0 and NUM_ENTRIES_TO_PROCESS < len(actual_data):
        data_to_process = actual_data[:NUM_ENTRIES_TO_PROCESS]

    print(f"Processing {len(data_to_process)} entries with Vanilla Prompting (with schema hint)...")

    for i, entry in enumerate(data_to_process):
        if llm:
            llm.reset()

        user_query = entry.get("input")
        expected_output = entry.get("output")
        extracted_items_from_file = entry.get("input_items", {})

        if user_query is None:
            print(f"Warning: Entry {i + 1} is missing 'input' query. Skipping.")
            continue

        print(f"\n--- Processing Entry {i + 1}/{len(data_to_process)} (Original Query: '{user_query}') ---")

        vanilla_llm_output_list = get_vanilla_parsed_output(
            llm,
            user_query,
            VANILLA_DOMAIN_SYSTEM_INSTRUCTION,
            schema_with_example_values_atis,
            ALL_INTENT_SLOTS_ATIS_FROM_EXAMPLES,
            generation_config
        )

        print(f"  Input: '{user_query}'")
        print(f"  Expected: {expected_output}")
        print(f"  Vanilla LLM Generated Output (Answer): {vanilla_llm_output_list}")

        results_log.append({
            "input": user_query,
            "expected": expected_output,
            "answer": vanilla_llm_output_list,
            "extracted_items": extracted_items_from_file
        })

    output_dir = os.path.dirname(OUTPUT_FILE_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nFull results log saved to {OUTPUT_FILE_NAME}")

    print("\n--- Evaluation Guidance ---")
    print(
        f"Compare 'answer' (Vanilla LLM's output) with 'expected' in '{OUTPUT_FILE_NAME}' using your evaluation script.")
    print(
        "This version used a single 'vanilla' prompt including ATIS schema examples as a HINT, but no NER pre-extracted items hint.")


if __name__ == "__main__":
    if MODEL_PATH == "/path/to/your/model.gguf" or not os.path.exists(MODEL_PATH):
        print(
            "ERROR: Please update the MODEL_PATH variable at the top of the script with the actual path to your GGUF model file.")
    else:
        start= time.time()
        main()
        print(time.time()-start)