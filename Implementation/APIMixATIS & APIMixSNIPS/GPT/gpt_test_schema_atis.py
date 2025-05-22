import argparse
import os
import json
import re
from openai import OpenAI

system_instruction = (
    "You are an expert task-oriented conversational model specialized in the ATIS (Air Travel Information System) domain. "
    "Your task is to generate a structured semantic representation of the user's request using predefined functions and parameters. "
    "Analyze the user query, identify the most appropriate function(s), and populate the parameters with values extracted from the query.\n\n"

    "### Output Format:\n"
    "Your output must be a Python-style list of one or more function calls. Each function call should be represented as a string in the following format:\n"
    "  - \"function_name(parameter='value', parameter='value', ...)\"\n"
    "If multiple functions are needed, include each as a separate string within the list. For example:\n"
    "  User: list california airports , list la \n"
    "  Output: [atis_airport(state='california', modifier=None, city=None), atis_city(city='los angeles', airline=None)] \n"
    "Ensure that the output is syntactically correct and adheres to this structure.\n\n"


    "### MixATIS Intents & Slots:\n\n" 

    "1. atis_airport – Provides information about airports, including listing airports in specific locations.\n"
    "   • state – The state where the airport is located (e.g., 'california').\n"
    "   • city – The city where the airport is located (e.g., 'los angeles').\n"
    "   • modifier – A descriptive qualifier for the airport (e.g., 'closest').\n\n"

    "2. atis_city – Identifies or lists cities associated with specific airlines or flights.\n"
    "   • city – The name of the city (e.g., 'la').\n"
    "   • airline – The name of an airline that operates in the city (e.g., 'northwest').\n\n"

    "3. atis_quantity – how many / number of flights or other entities matching specific criteria.\n"
    "   • airline – The name of the airline (e.g., 'canadian airlines').\n"
    "   • aircraft – The type of aircraft (e.g., '320', 'dh8').\n\n"

    "4. atis_airfare – Provides information about and ticket availability and prices.\n"
    "   • departure_city – The starting city for the flight.\n"
    "   • destination_city – The destination city for the flight.\n"
    "   • class – The service class of the ticket (e.g., 'first class', 'coach').\n"
    "   • relative_cost – Price condition (e.g., 'under').\n"
    "   • fare – The fare value (e.g., '200 dollars').\n"
    "   • departure_day – The day of departure (e.g., 'saturday').\n"
    "   • airline_code – The code of the airline (e.g., 'twa').\n\n"

    "5. atis_flight_no – flight numbers matching the query criteria.\n"
    "   • departure_city – The starting city of the flight.\n"
    "   • destination_city – The destination city of the flight.\n"
    "   • airline – The name of the airline.\n"
    "   • relative_departure_time – Describes a time condition (e.g., 'before').\n"
    "   • departure_time – The exact departure time (e.g., '8 am').\n"
    "   • departure_day – The day of departure (e.g., 'thursday').\n\n"

    "6. atis_capacity – seating capacity of specific aircraft.\n"
    "   • aircraft – The model of the aircraft (e.g., '757', '733').\n"
    "   • airline – The name of the airline operating the aircraft.\n\n"

    "7. atis_distance – Provides the distance between two cities.\n"
    "   • departure_city – The starting city.\n"
    "   • destination_city – The destination city.\n\n"

    "8. atis_aircraft – Provides details about specific aircraft models or types, not about capacity.\n"
    "   • aircraft – The type of aircraft (e.g., 'm80').\n"
    "   • departure_city – The city of departure.\n"
    "   • destination_city – The city of destination.\n"
    "   • relative_departure_time – Describes a time condition (e.g., 'before').\n"
    "   • departure_period – The period of the day (e.g., 'noon').\n"
    "   • airline – The name of the airline operating the aircraft.\n"
    "   • departure_time – The exact departure time (e.g., '4:19pm').\n\n"

    "9. atis_day_name – Identifies the day of the week on which flights operate between two cities.\n"
    "   • departure_city – The starting city of the flight.\n"
    "   • destination_city – The destination city of the flight.\n\n"

    "10. atis_ground_service – Provides information about ground transportation options, not about price/cost\n"
    "    • transport_type – The type of ground transport (e.g., 'limousine', 'car').\n"
    "    • city – The city where the ground transportation is available.\n\n"

    "11. atis_abbreviation – Use of fare codes or airline codes or aircraft codes.\n"
    "    • fare_code – The code representing a fare type (e.g., 'q').\n"
    "    • airline_code – The code representing an airline (e.g., 'AA').\n"
    "    • aircraft – The type of aircraft (e.g., 'd9s').\n\n"

    "12. atis_meal – Provides information about meals available on specific flights.\n"
    "    • flight_number – The flight number (e.g., 'AA665').\n"
    "    • meal_type – The type of meal (e.g., 'meals').\n"
    "    • departure_day – The day of departure.\n"
    "    • departure_period – The time of day (e.g., 'morning').\n"
    "    • meal_description – The description of the meal (e.g., 'snacks').\n"
    "    • airline – The airline providing the meal.\n\n"

    "13. atis_flight – Provides general information about flights between cities.\n"
    "    • departure_city – The starting city of the flight.\n"
    "    • destination_city – The destination city of the flight.\n"
    "    • departure_month – The month of departure (e.g., 'april').\n"
    "    • departure_day_number – The day of the month (e.g., 'fifth').\n"
    "    • airline – The airline operating the flight.\n\n"

    "14. atis_ground_fare – How much it costs for ground transportation options.\n"
    "    • transport_type – The type of ground transport (e.g., 'limousine').\n"
    "    • city – The city where the ground fare applies.\n\n"

    "15. atis_airline – Provides information about airlines.\n"
    "    • airline_code – The code representing an airline (e.g., 'AA').\n"
    "    • departure_city – The starting city of the airline.\n"
    "    • destination_city – The destination city of the airline.\n\n"

    "16. atis_flight_time – Provides the flight/arrival/departure time for flights.\n"
    "    • departure_city – The starting city.\n"
    "    • destination_city – The destination city.\n"
    "    • flight_time – Whether it's the arrival or departure time (e.g., 'arrival' or 'departure').\n\n"

    # IMPORTANT: The header below MUST match what the parser function expects for splitting.
    "### Processing Steps:\n" # Changed from "### Your task:" to match parser
    "- Carefully analyze the user query.\n" # Content from your ATIS instruction
    "- Select the most suitable function(s) based on the query.\n"
    "- Populate the function(s) with parameters and values extracted from the query.\n"
    "- Ensure that the output is a list of function call strings, each following the specified format.\n"
)

schema = {
    "atis_airport": {
        "state": [
            "california",
            "oregon",
            "texas",
            "arizona",
            "florida"
        ],
        "modifier": [
            "closest"
        ],
        "city": [
            "la",
            "ontario",
            "new york"
        ]
    },
    "atis_city": {
        "city": [
            "portland",
            "la"
        ],
        "airline": [
            "spirit",
            "lufthansa",
            "eastwest",
            "northeast",
            "northwest"
        ]
    },
    "atis_quantity": {
        "airline": [
            "british airways",
            "spirit",
            "canadian airlines",
            "delta",
            "lufthansa",
            "american airlines",
            "wizzair"
        ],
        "aircraft": [
            "j32",
            "br31",
            "re89",
            "j31",
            "ca32",
            "320",
            "dh8",
            "dh9",
            "ca31",
            "j33"
        ]
    },
    "atis_airfare": {
        "departure_city": [
            "indianapolis",
            "boston",
            "los angeles",
            "detroit",
            "chicago",
            "pittsburgh",
            "toronto",
            "nashville",
            "washington",
            "columbus"
        ],
        "destination_city": [
            "seattle",
            "cleveland",
            "moscow",
            "las vegas",
            "st. petersburg",
            "st. louis",
            "chicago",
            "atlanta",
            "washington",
            "memphis",
            "san jose",
            "san diego",
            "tacoma",
            "montreal"
        ],
        "airline_code": [
            "twa"
        ],
        "class": [
            "first class",
            "coach"
        ],
        "relative_cost": [
            "under"
        ],
        "fare": [
            "200 dollars",
            "300 pounds",
            "100 australian dollars"
        ],
        "departure_day": [
            "saturday"
        ]
    },
    "atis_flight_no": {
        "departure_city": [
            "houston",
            "dallas",
            "cleveland",
            "phoenix",
            "chicago",
            "oakland",
            "nashville"
        ],
        "destination_city": [
            "houston",
            "seattle",
            "dallas",
            "boston",
            "salt lake city",
            "milwaukee",
            "tacoma"
        ],
        "airline": [
            "continental",
            "american airlines",
            "british airways"
        ],
        "relative_departure_time": [
            "before"
        ],
        "departure_time": [
            "8 am"
        ],
        "departure_day": [
            "thursday"
        ]
    },
    "atis_distance": {
        "departure_city": [
            "dallas",
            "boston",
            "san francisco",
            "los angeles",
            "toronto",
            "baltimore"
        ],
        "destination_city": [
            "dallas",
            "san francisco",
            "los angeles",
            "moscow",
            "new york",
            "ottawa",
            "toronto"
        ]
    },
    "atis_aircraft": {
        "aircraft": [
            "m81",
            "m80"
        ],
        "departure_city": [
            "cleveland"
        ],
        "destination_city": [
            "dallas"
        ],
        "relative_departure_time": [
            "before"
        ],
        "departure_period": [
            "noon"
        ],
        "airline": [
            "american"
        ],
        "departure_time": [
            "4:19pm"
        ]
    },
    "atis_day_name": {
        "departure_city": [
            "san jose",
            "los angeles",
            "moscow",
            "nashville"
        ],
        "destination_city": [
            "portland",
            "dallas",
            "new york",
            "nashville",
            "tacoma"
        ]
    },
    "atis_ground_service": {
        "transport_type": [
            "car",
            "limousine"
        ],
        "city": [
            "portland",
            "dallas",
            "burbank",
            "porto",
            "los angeles",
            "new york",
            "las vegas",
            "berlin",
            "st. louis",
            "salt lake city",
            "fort worth",
            "denver",
            "milwaukee",
            "washington",
            "tampa"
        ]
    },
    "atis_abbreviation": {
        "fare_code": [
            "qo",
            "m",
            "l",
            "f",
            "g",
            "q"
        ],
        "airline_code": [
            "uat",
            "ua"
        ],
        "aircraft": [
            "d9s"
        ]
    },
    "atis_meal": {
        "flight_number": [
            "BA665",
            "AA811",
            "BA123",
            "BA354",
            "BA121",
            "LH468",
            "AA665",
            "TK382",
            "TK12"
        ],
        "meal_type": [
            "meals",
            "drinks"
        ],
        "departure_day": [
            "tuesday"
        ],
        "departure_period": [
            "evening",
            "morning"
        ],
        "meal_description": [
            "snacks"
        ],
        "airline": [
            "wizzair",
            "tower air"
        ]
    },
    "atis_flight": {
        "departure_city": [
            "cleveland",
            "los angeles",
            "pittsburgh",
            "memphis",
            "montreal"
        ],
        "destination_city": [
            "indianapolis",
            "burbank",
            "cleveland",
            "long beach",
            "milwaukee",
            "san jose"
        ],
        "departure_month": [
            "april"
        ],
        "departure_day_number": [
            "fifth"
        ],
        "airline": [
            "alaska airlines"
        ]
    },
    "atis_capacity": {
        "aircraft": [
            "757",
            "73s",
            "733",
            "m80",
            "747",
            "be1"
        ],
        "airline": [
            "delta"
        ],
        "departure_city": [
            "seattle"
        ],
        "destination_city": [
            "salt lake city"
        ]
    },
    "atis_flight_time": {
        "departure_city": [
            "la",
            "michigan",
            "detroit"
        ],
        "destination_city": [
            "san francisco",
            "westchester",
            "manchester",
            "new york",
            "michigan"
        ],
        "flight_time": [
            "departure",
            "arrival"
        ]
    },
    "atis_ground_fare": {
        "transport_type": [
            "limousine"
        ],
        "city": [
            "boston",
            "bucharest",
            "los angeles",
            "moscow",
            "denver",
            "michigan",
            "san diego"
        ]
    },
    "atis_airline": {
        "airline_code": [
            "aa",
            "as",
            "BA",
            "us",
            "sr",
            "hp",
            "pr",
            "rp",
            "DA"
        ],
        "departure_city": [
            "seattle",
            "washington",
            "toronto"
        ],
        "destination_city": [
            "st. louis",
            "salt lake city",
            "san diego",
            "columbus"
        ]
    }
}

REQUIRED_SLOTS_MAP = {
    "atis_airport": [],
    "atis_city": [],
    "atis_quantity": [],
    "atis_airfare": [],
    "atis_flight_no": [],
    "atis_capacity": [],
    "atis_distance": [],
    "atis_aircraft": [],
    "atis_day_name": [],
    "atis_ground_service": [],
    "atis_abbreviation": [],
    "atis_meal": [],
    "atis_flight": [],
    "atis_ground_fare": [],
    "atis_airline": [],
    "atis_flight_time": []
}

def parse_system_instruction_for_intents(instruction_string: str) -> dict:
    intents_data = {}
    try:
        relevant_section = instruction_string.split("### MixATIS Intents & Slots:\n")[1]
        if "### Processing Steps:\n" in relevant_section:
            relevant_section = relevant_section.split("### Processing Steps:\n")[0]
    except IndexError:
        print("Error: Could not find '### MixSNIPS Intents & Slots:' section in system_instruction.")
        return intents_data

    intent_header_pattern = re.compile(r"^\s*(\d+)\.\s*([A-Za-z0-9_]+)\s*–\s*(.*?)\s*$", re.MULTILINE)
    slot_pattern = re.compile(r"^\s*•\s*([a-zA-Z_][a-zA-Z0-9_]+)\s*–\s*(.*?)\s*$", re.MULTILINE)
    intent_matches = list(intent_header_pattern.finditer(relevant_section))

    for i, current_match in enumerate(intent_matches):
        _, intent_name, intent_desc = current_match.groups()
        intent_name = intent_name.strip()
        intent_desc = intent_desc.strip()
        start_pos = current_match.end()
        end_pos = intent_matches[i + 1].start() if (i + 1) < len(intent_matches) else len(relevant_section)
        slots_text_block = relevant_section[start_pos:end_pos]
        current_slots = {}
        for slot_match in slot_pattern.finditer(slots_text_block):
            slot_name, slot_desc = slot_match.groups()
            current_slots[slot_name.strip()] = slot_desc.strip()
        intents_data[intent_name] = {"description": intent_desc, "slots": current_slots}
    return intents_data


def generate_tools_definitions(parsed_intent_info: dict, current_schema_with_examples: dict,
                               current_required_slots_map: dict,
                               enum_threshold=10) -> list:  # Renamed schema_with_examples to current_schema_with_examples
    tools_defs = []
    for intent_name, intent_data in parsed_intent_info.items():
        properties = {}
        if "slots" not in intent_data:
            print(f"Warning: No slots found for intent {intent_name} in parsed_intent_info.")
            continue
        for slot_name, slot_desc_from_instr in intent_data["slots"].items():
            slot_type = "string"
            enum_values = None
            if intent_name in current_schema_with_examples and slot_name in current_schema_with_examples[intent_name]:
                example_values = list(set(current_schema_with_examples[intent_name][slot_name]))
                if 0 < len(example_values) <= enum_threshold:
                    enum_values = example_values
            properties[slot_name] = {"type": slot_type, "description": slot_desc_from_instr}
            if enum_values:
                properties[slot_name]["enum"] = enum_values
        tool_def = {
            "type": "function",
            "function": {
                "name": intent_name,
                "description": intent_data["description"],
                "parameters": {"type": "object", "properties": properties,
                               "required": current_required_slots_map.get(intent_name, [])}
            }
        }
        tools_defs.append(tool_def)
    return tools_defs


# Updated function to include example_schema_dict for prompt augmentation
def get_gpt4_intent_calls_from_query(
        user_query: str,
        extracted_items_dict: dict,
        example_schema_dict: dict,  # New parameter for the full example schema
        system_prompt_for_gpt4: str,
        current_tools_definitions: list,
        current_all_intent_slots: dict,
        openai_client: OpenAI
) -> list[str]:
    # --- Create hint from extracted_items_dict ---
    items_hint_str = ""
    if extracted_items_dict:
        formatted_items_list = []
        for key, values in extracted_items_dict.items():
            if values and isinstance(values, list) and all(isinstance(v, str) for v in values):
                quoted_values = [f"'{v}'" if ' ' in v or not v.isalnum() else v for v in values]
                formatted_items_list.append(f"- {key}: {', '.join(quoted_values)}")
            elif values and isinstance(values, str):
                formatted_items_list.append(
                    f"- {key}: '{values}'" if ' ' in values or not values.isalnum() else f"- {key}: {values}")
        if formatted_items_list:
            items_hint_str = "\n\nHint: Consider these pre-extracted items to help fill slots if relevant:\n" + "\n".join(
                formatted_items_list)

    # --- Create hint from example_schema_dict ---
    schema_hint_str = ""
    if example_schema_dict:
        schema_hint_parts = [
            "\n\nHint: Here are some known example values for slots (these are illustrative examples, not exhaustive lists; prioritize values from the query or pre-extracted items if they conflict):"]
        for intent_name, slots_data in example_schema_dict.items():
            if intent_name in current_all_intent_slots:  # Only show examples for known intents
                schema_hint_parts.append(f"  For Intent '{intent_name}':")
                for slot_name, ex_values in slots_data.items():
                    if ex_values and isinstance(ex_values, list):  # Show only first few examples for brevity
                        display_examples = [f"'{v}'" for v in ex_values[:3]]  # Show up to 3 examples
                        schema_hint_parts.append(
                            f"    - {slot_name} examples: {', '.join(display_examples)}{'...' if len(ex_values) > 3 else ''}")
        if len(schema_hint_parts) > 1:  # Only add if there's actual schema content
            schema_hint_str = "\n" + "\n".join(schema_hint_parts)

    augmented_user_query = user_query + items_hint_str + schema_hint_str

    messages = [
        {"role": "system", "content": system_prompt_for_gpt4},
        {"role": "user", "content": augmented_user_query}
    ]
    formatted_intent_strings = []

    if not current_tools_definitions:
        print("Error: tools_definitions is empty. Cannot make API call.")
        return [f"ConfigurationError(error='tools_definitions is empty')"]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",  # Changed model to gpt-4o from user's gpt-4.1-mini
            messages=messages,
            tools=current_tools_definitions,
            tool_choice="auto",
        )
        response_message = response.choices[0].message

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(
                        f"Error: Could not decode arguments for {function_name}: {tool_call.function.arguments}. Error: {e}")
                    error_arg_str = f"arguments_error='Could not decode: {tool_call.function.arguments[:50]}...'"
                    formatted_intent_strings.append(f"{function_name}({error_arg_str})")
                    continue

                if function_name in current_all_intent_slots:
                    slots_for_this_intent = current_all_intent_slots[function_name]
                    arg_strings_list = []
                    for slot_name in slots_for_this_intent:
                        value = arguments.get(slot_name)
                        if value is None:
                            arg_strings_list.append(f"{slot_name}=None")
                        else:
                            if isinstance(value, str):
                                escaped_value = value.replace("'", "\\'")
                                arg_strings_list.append(f"{slot_name}='{escaped_value}'")
                            else:
                                arg_strings_list.append(f"{slot_name}={value}")
                    formatted_intent_strings.append(f"{function_name}({', '.join(arg_strings_list)})")
                else:
                    print(f"Warning: GPT-4 called an unknown function '{function_name}'. Original args: {arguments}")
                    formatted_intent_strings.append(
                        f"UnknownIntentError(name='{function_name}', args='{str(arguments)}')")

        elif response_message.content:
            print(f"Warning: GPT-4 provided a text response instead of a tool call: {response_message.content}")

    except Exception as e:
        print(f"Error calling OpenAI API or processing response: {e}")
        formatted_intent_strings.append(f"APIError(error='{str(e)[:100]}...')")

    return formatted_intent_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MixATIS baseline with GPT-4 using a command-line API key, pre-extracted items, and schema examples in prompt.")
    parser.add_argument("--api_key", type=str, required=True, help="Your OpenAI API key (e.g., sk-...).")
    parser.add_argument("--input_file", type=str, default="atis_data_augmented.json",
                        help="Path to the input JSONL data file.")
    # Changed default output file name to reflect new prompt structure
    parser.add_argument("--output_file", type=str, default="gpt4_baseline_with_all_hints_atis.json",
                        help="Path to save the output JSON results.")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # Renamed user's global 'schema' to 'user_provided_schema_with_examples' for clarity when passing to functions
    # This is the same object the user provided globally and named 'schema'.
    parsed_snips_info = parse_system_instruction_for_intents(system_instruction)
    if not parsed_snips_info:
        print("ERROR: Could not parse system instruction. Exiting.")
        exit()

    tools_definitions = generate_tools_definitions(
        parsed_snips_info,
        schema,  # This is the user's global 'schema' variable
        REQUIRED_SLOTS_MAP
    )
    if not tools_definitions:
        print("ERROR: Tools definitions were not generated. Exiting.")
        exit()

    ALL_INTENT_SLOTS = {
        intent: list(data["slots"].keys())
        for intent, data in parsed_snips_info.items() if "slots" in data
    }
    if not ALL_INTENT_SLOTS:
        print("ERROR: ALL_INTENT_SLOTS is empty. Exiting.")
        exit()

    # Updated system instruction to mention both hints
    system_instruction_for_gpt4 = (
        "You are an expert task-oriented model for the MixATIS domain. Parse user utterances into intent calls using the provided functions/tools. "
        "Extract slot values accurately. "
        "Two types of hints might be provided with the user query:\n"
        "1. 'Hint: Consider these pre-extracted items: ...' - If present, prioritize using these values for slots where appropriate.\n"
        "2. 'Hint: Here are some known example values for slots: ...' - This provides illustrative examples for slot values. Use this for guidance on the type of values expected but prioritize information from the user's query or pre-extracted items if there's a conflict. These examples are not exhaustive.\n"
        "Always ensure the final arguments are consistent with the user's full request. "
        "Since no slots are strictly required by the functions, only include arguments for slots explicitly mentioned, clearly implied by the user, or suggested by the hints. "
        "Post-processing will handle mapping any known but unmentioned slots for an intent to 'None', so you don't need to include slots with no value in the function call arguments."
    )

    results_log = []

    print(f"Loading data from: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            actual_data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(actual_data)} entries.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input_file}")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from file: {args.input_file}. Ensure it's a valid JSONL file.")
        exit()

    # Processing a slice of data for example: actual_data[5:10]
    # Replace with `actual_data` to process the whole file
    # data_to_process = actual_data
    data_to_process = actual_data if len(actual_data) >= 10 else actual_data[:min(5, len(actual_data))]

    for i, entry in enumerate(data_to_process):
        user_query = entry.get("input")
        expected_output = entry.get("output")
        extracted_items_from_file = entry.get("input_items", {})
        if not isinstance(extracted_items_from_file, dict):
            print(
                f"Warning: Entry {i + 1} 'input_items' is not a dictionary. Treating as empty. Value: {extracted_items_from_file}")
            extracted_items_from_file = {}

        if user_query is None:
            print(f"Warning: Entry {i + 1} is missing 'input' query. Skipping.")
            continue

        print(
            f"\n--- Processing Entry {i + 1}/{len(data_to_process)} (Original Index: {actual_data.index(entry) if entry in actual_data else 'N/A'}) ---")
        print(f"User Query: {user_query}")
        # The hints will be printed by get_gpt4_intent_calls_from_query if items_hint_str or schema_hint_str are generated
        if expected_output is not None:
            print(f"Expected Output: {expected_output}")

        gpt4_output_list = get_gpt4_intent_calls_from_query(
            user_query,
            extracted_items_from_file,
            schema,  # Pass your global schema object here
            system_instruction_for_gpt4,
            tools_definitions,
            ALL_INTENT_SLOTS,
            client
        )

        print(f"GPT-4 Formatted Output (Answer): {gpt4_output_list}")

        results_log.append({
            "input": user_query,
            "expected": expected_output,
            "answer": gpt4_output_list,
            "extracted_items": extracted_items_from_file
        })
        print("-" * 70)

    output_filename = args.output_file
    # Update default filename if still an old one
    if args.output_file in ("gpt4_baseline_from_actual_data.json", "gpt4_baseline_with_extracted_items.json",
                            "gpt4_baseline_with_extracted_items_hint.json", "gpt4_baseline_no_ner_hint.json"):
        output_filename = "gpt4_baseline_with_all_hints_atis.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nFull results log saved to {output_filename}")

    print("\n--- Evaluation Guidance ---")
    print(f"Compare 'answer' (GPT-4's output) with 'expected' in '{output_filename}'.")
    print("This version provided both pre-extracted NER items AND schema examples as HINTS to GPT-4.")
