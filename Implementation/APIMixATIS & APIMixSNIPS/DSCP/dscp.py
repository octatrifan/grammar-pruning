import argparse
import os
import json
import re
from llama_cpp import Llama
import time

MODEL_PATH = "Qwen3-4B-BF16.gguf"
INPUT_FILE_NAME = "atis_data_augmented.json"
OUTPUT_FILE_NAME = "testime.json"
N_GPU_LAYERS = -1
CTX_SIZE = 16000
MAX_TOKENS_GENERATION = 1000
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 30
REPEAT_PENALTY = 1.1
NUM_ENTRIES_TO_PROCESS = 0

DOMAIN_SYSTEM_INSTRUCTION = (
    "You are an expert task-oriented conversational model specialized in the ATIS (Air Travel Information System) domain. "
    "Your task is to generate a structured semantic representation of the user's SUB-QUERY using ONE predefined function and its parameters. "
    "Analyze the sub-query, identify the single most appropriate function, and populate its parameters with values extracted from THIS SUB-QUERY ONLY.\n\n"

    "### Output Structure:\n"
    "Your entire output must be ONLY the SINGLE function call string, and nothing else. It must be in the following format:\n"
    "  \"function_name(parameter='value', parameter2='value', ...)\"\n"
    "For any slot parameter defined for the chosen function that is not mentioned or inferable from the user’s sub-query, explicitly assign `slot_name=None`.\n"
    "Strictly adhere to this structure for the ONE most relevant function. Do not include any explanations or conversational text.\n\n"

    "### Functions and Their Descriptions (Full list for context):\n\n"
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
    "### Slot Value Examples (for guidance, not exhaustive):\n"
    "{schema_example_hint_string}\n\n"
    "### Task for this specific input:\n"
    "Analyze the user's sub-query, identify the single most appropriate function from the list above, and populate its parameters with values extracted from THIS SUB-QUERY ONLY. "
    "Use the slot value examples for guidance on expected value types. Prioritize values directly from the sub-query. "
    "If a parameter defined for the chosen function is not mentioned in this sub-query, assign `parameter_name=None` for that parameter. "
    "Your entire response should be just the single function call string."
)

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

ALL_INTENT_SLOTS_ATIS = {
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
            stop=["User:", "### Input:", "\n\n\n", "Output:", "Sub-queries:", "Sub-query:", "Human:", "Assistant:",
                  "Numbered Sub-queries:"]
        )
        generated_text = output['choices'][0]['text'].strip()
        return generated_text
    except Exception as e:
        print(f"Error during LLM completion: {e}")
        return ""



def dscp_divide(llm: Llama, user_query: str, generation_cfg: dict) -> list[str]:
    """
    Step 1: Divide the user query into single-intent sub-utterances, expecting a numbered list.
    Guides the model to produce 1, 2, or at most 3 distinct sub-queries.
    Post-processes for uniqueness.
    """
    prompt = (
        "You are an expert query decomposer. Your task is to break down the given 'User Utterance' into distinct, single-intent sub-queries. "
        "It is absolutely crucial that you only identify sub-queries that genuinely represent separate and unique intents expressed by the user. "
        "If the user utterance expresses only one distinct intent, output only that single sub-query as '1. <sub-query_1>'. "
        "If it expresses two distinct intents, output two unique sub-queries, starting with '1. <sub-query_1>' and '2. <sub-query_2>'. "
        "If it expresses three distinct intents, output three unique sub-queries, such as '1. ...', '2. ...', and '3. ...'. "
        "Do NOT repeat sub-queries. Do NOT invent sub-queries if they are not present. Do NOT output more than 3 sub-queries. "
        "List each identified sub-query on a new line, prefixed with a number and a period (e.g., '1. sub-query one'). "
        "Output ONLY the numbered sub-queries, with no other text, explanations, or conversational phrases before or after the list.\n\n"
        "User Utterance: \"{user_query}\"\n\n"
        "Numbered Sub-queries (ONLY distinct and unique intents, 1 to 3 lines maximum, one per line):\n"
    ).format(user_query=user_query)

    response = get_llm_completion(llm, prompt, **generation_cfg)

    parsed_sub_utterances = []
    for line in response.split('\n'):
        line = line.strip()
        match = re.match(r"^\s*\d+\.\s*(.+)", line)
        if match:
            sub_query_text = match.group(1).strip()
            if sub_query_text:
                parsed_sub_utterances.append(sub_query_text)

    # Filter for uniqueness while preserving order
    unique_sub_utterances = []
    seen_utterances = set()
    for sub_utt in parsed_sub_utterances:
        if sub_utt not in seen_utterances:
            unique_sub_utterances.append(sub_utt)
            seen_utterances.add(sub_utt)

    if not unique_sub_utterances and user_query:
        print(
            "Warning: Division step (numbered list & uniqueness filter) produced no valid sub-utterances. Using the original query as a single sub-utterance.")
        return [user_query]

    return unique_sub_utterances[:3]  # Cap at 3 unique sub-utterances


def dscp_solve(llm: Llama, sub_utterance: str, base_domain_instruction: str,
               example_schema_data: dict, all_intent_slots_for_domain: dict,
               generation_cfg: dict) -> list[str]:
    schema_hint_str = ""
    if example_schema_data:
        schema_hint_parts = []
        for intent_name, slots_data in example_schema_data.items():
            if intent_name in all_intent_slots_for_domain:
                slot_examples_strs = []
                for slot_name, ex_values in slots_data.items():
                    if ex_values and isinstance(ex_values, list) and ex_values[0] is not None:
                        slot_examples_strs.append(f"{slot_name} (e.g., '{ex_values[0]}')")
                if slot_examples_strs:
                    schema_hint_parts.append(f"  Intent '{intent_name}': {'; '.join(slot_examples_strs)}")
        if schema_hint_parts:
            schema_hint_str = "\n".join(schema_hint_parts[:5])
            if len(schema_hint_parts) > 5:
                schema_hint_str += "\n  ..."

    final_domain_instruction = base_domain_instruction.replace(
        "{schema_example_hint_string}",
        schema_hint_str if schema_hint_str else "No specific slot value examples provided for this context."
    )

    prompt = (
        f"{final_domain_instruction}\n\n"
        "User's Sub-query: \"{sub_utterance}\"\n\n"
        "Your entire response must be only the single function call string. Do not add any other text.\n"
        "Structured Output (single function call string adhering to the format specified above):\n"
    ).format(sub_utterance=sub_utterance)

    response = get_llm_completion(llm, prompt, **generation_cfg)

    call_strings = []
    cleaned_response = response.strip()

    single_call_match = re.fullmatch(r"([a-zA-Z0-9_]+)\s*\((.*)\)", cleaned_response)

    if single_call_match:
        call_strings.append(cleaned_response)
    elif cleaned_response:
        first_match = re.search(r"([a-zA-Z0-9_]+\s*\(.*?\))", cleaned_response)
        if first_match:
            plausible_call = first_match.group(1).strip()
            if re.fullmatch(r"[a-zA-Z0-9_]+\s*\((.*)\)", plausible_call):
                call_strings.append(plausible_call)
            else:
                print(
                    f"Warning: Solve step for '{sub_utterance}' fallback extraction failed full validation: '{plausible_call}'. Original: '{cleaned_response}'")
        else:
            print(
                f"Warning: Solve step for '{sub_utterance}' did not produce a recognized single function call string nor a plausible fallback. Got: '{cleaned_response}'")

    return call_strings


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

    print(f"Processing {len(data_to_process)} entries...")

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

        current_sub_utterances = dscp_divide(llm, user_query, generation_config)
        print(f"  Divided into {len(current_sub_utterances)} unique sub-utterances: {current_sub_utterances}")

        final_intent_calls_for_query = []
        if not current_sub_utterances:
            print("  No sub_utterances from division step.")
        else:
            for sub_idx, sub_utt in enumerate(current_sub_utterances):  # Already capped at 3 unique
                if not sub_utt.strip():
                    continue
                print(f"  Solving sub-utterance {sub_idx + 1}/{len(current_sub_utterances)}: '{sub_utt}'")
                intent_call_strings = dscp_solve(
                    llm,
                    sub_utt,
                    DOMAIN_SYSTEM_INSTRUCTION,
                    schema_with_example_values_atis,
                    ALL_INTENT_SLOTS_ATIS,
                    generation_config
                )
                if intent_call_strings:
                    final_intent_calls_for_query.extend(intent_call_strings)
                else:
                    print(f"  No valid intent call generated for sub-utterance: '{sub_utt}'")

        print(f"  Input: '{user_query}'")
        print(f"  Expected: {expected_output}")
        print(f"  DSCP Generated Output (Answer): {final_intent_calls_for_query}")

        results_log.append({
            "input": user_query,
            "expected": expected_output,
            "answer": final_intent_calls_for_query,
            "extracted_items": extracted_items_from_file
        })

    output_dir = os.path.dirname(OUTPUT_FILE_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nFull results log saved to {OUTPUT_FILE_NAME}")

    print("\n--- Evaluation Guidance ---")
    print(f"Compare 'answer' (DSCP LLM's output) with 'expected' in '{OUTPUT_FILE_NAME}' using your evaluation script.")


if __name__ == "__main__":
    if MODEL_PATH == "/path/to/your/model.gguf" or not os.path.exists(MODEL_PATH):
        print(
            "ERROR: Please update the MODEL_PATH variable at the top of the script with the actual path to your GGUF model file.")
    else:
        start_time = time.time()
        main()
        print("--- %s seconds ---" % (time.time() - start_time))