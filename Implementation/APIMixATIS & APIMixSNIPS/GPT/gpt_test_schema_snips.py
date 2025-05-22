import argparse
import os
import json
import re
from openai import OpenAI

system_instruction = (
    "You are an expert task-oriented conversational model for the MixSNIPS domain. "
    "Your job is to parse each user utterance into one or more MixSNIPS intent calls, using the exact intent names and slot parameters defined in the MixSNIPS schema below.\n\n"
    "### Output Format:\n"
    "- Return a Python list of strings, each string representing one intent invocation.\n"
    "- Each string must follow this pattern:\n"
    "    \"[IntentName(slot1='value1', slot2='value2', …), ... ]\"\n"
    "- If multiple intents appear in the utterance, include each as its own string, in the order they appear.\n"

    "### MixSNIPS Intents & Slots:\n\n"
    "1. AddToPlaylist – Add an item (song/album) to a specified playlist\n"
    "   • playlist          – target playlist name\n"
    "   • playlist_owner    – owner identifier (e.g., 'my', 'betsy s')\n"
    "   • entity_name       – title of the song/book/etc. to add\n"
    "   • artist            – artist name\n"
    "   • music_item        – specific track or album identifier\n\n"
    "2. PlayMusic – Play or queue up music according to user preferences\n"
    "   • artist            – performing artist\n"
    "   • sort              – ordering preference ('newest', 'last')\n"
    "   • year              – year or era filter\n"
    "   • music_item        – specific song or album\n"
    "   • playlist          – playlist to play from\n"
    "   • service           – streaming service ('spotify', 'itunes', etc.)\n"
    "   • genre             – music genre\n"
    "   • album             – album name\n\n"
    "3. BookRestaurant – Find or book a restaurant reservation\n"
    "   • spatial_relation  – proximity qualifier ('close', 'around', 'distant')\n"
    "   • city              – city name\n"
    "   • time              – named day or vague time ('monday', etc.)\n"
    "   • party_size_number – number of people ('five', '3', etc.)\n"
    "   • restaurant_type   – type of venue ('steakhouse', 'food truck', etc.)\n"
    "   • served_dish       – desired dish\n"
    "   • country           – country name\n"
    "   • restaurant_name   – specific restaurant\n"
    "   • timeRange         – precise date or time range\n"
    "   • state             – state abbreviation\n"
    "   • cuisine           – cuisine type\n\n"
    "4. SearchCreativeWork – Search for a creative work (movie, song, book, etc.)\n"
    "   • object_type       – kind of work ('movie', 'song', 'video game', etc.)\n"
    "   • object_name       – title or name of the work\n\n"
    "5. SearchScreeningEvent – Look up showtimes or screening events\n"
    "   • movie_name        – film title\n"
    "   • location_name     – venue name\n"
    "   • movie_type        – category ('animated movies', 'films', etc.)\n"
    "   • spatial_relation  – proximity qualifier\n"
    "   • timeRange         – numeric or descriptive time token\n\n"
    "6. RateBook – Rate or review a book or text\n"
    "   • object_name               – title of the book/text\n"
    "   • rating_value              – rating given ('five', '3', etc.)\n"
    "   • rating_unit               – unit ('stars', 'points')\n"
    "   • best_rating               – maximum on the scale\n"
    "   • object_type               – type of object ('textbook', 'album')\n"
    "   • object_part_of_series_type – series descriptor ('chronicle')\n\n"
    "7. GetWeather – Request weather information for a location and time\n"
    "   • city               – city name\n"
    "   • geographic_poi     – landmark or point of interest\n"
    "   • timeRange          – time or date expression\n"
    "   • condition_description – weather feature ('snow', 'humidity', etc.)\n"
    "   • state              – state abbreviation\n"
    "   • country            – country name\n"
    "   • condition_temperature – qualitative temperature descriptor ('hotter', 'chilly')\n\n"
    "### Processing Steps:\n"
    "1. Identify all intents mentioned by the user.\n"
    "2. For each intent, extract slot values exactly as they appear.\n"
    "4. Emit the list of intent-call strings, ensuring valid Python syntax.\n"
)

schema = {
    "AddToPlaylist": {
        "playlist": [
            "the refugee playlist",
            "nu metal",
            "songs to sing in the car",
            "wild country",
            "evening commute",
            "white noise",
            "epic gaming",
            "el mejor rock en espanol",
            "jazz classics",
            "just dance to afterclub",
            "edna st vincent millay",
            "gold edition",
            "electro sur",
            "indie hipster",
            "rock save the queen",
            "chill",
            "crate diggers anonymous",
            "i love neo soul",
            "this is mozart",
            "dinnertime acoustics",
            "fresh finds fire emoji",
            "evening acoustic",
            "we everywhere",
            "soundscapes for gaming",
            "post-grunge",
            "dance workout"
        ],
        "playlist_owner": [
            "dorothea s",
            "my",
            "betsy s"
        ],
        "entity_name": [
            "stairway to heaven",
            "beautiful world",
            "star light star bright"
        ],
        "artist": [
            "george thorogood",
            "coldplay",
            "matt garrison",
            "patti page",
            "frank ferrer",
            "troy van leeuwen",
            "jency anthony",
            "odesza",
            "rob tyner",
            "gackt camui",
            "andreas johnson",
            "fabri fibra"
        ],
        "music_item": [
            "pangaea",
            "the blurred crusade",
            "mary wells sings",
            "heart of the world",
            "even serpents",
            "hanging on"
        ]
    },
    "PlayMusic": {
        "artist": [
            "evil jared hasselhoff",
            "the beatles",
            "soko",
            "frank beard",
            "vlada divljan",
            "wellman braud",
            "coldplay",
            "cinder block",
            "richard kruspe",
            "jawad ahmad",
            "rob mills",
            "odesza",
            "arjen anthony lucassen",
            "gackt camui",
            "chris cunningham",
            "panda bear",
            "clark kent"
        ],
        "sort": [
            "newest",
            "latest",
            "last"
        ],
        "year": [
            "1975",
            "1987",
            "1974",
            "twenties"
        ],
        "music_item": [
            "a moment apart",
            "sugar baby",
            "ep",
            "i get ideas",
            "this is colour",
            "movement",
            "asleep in the deep",
            "album",
            "tiny tim ep"
        ],
        "playlist": [
            "tgif"
        ],
        "service": [
            "spotify",
            "itunes",
            "iheart"
        ],
        "genre": [
            "jungle",
            "theme",
            "symphonic rock"
        ],
        "album": [
            "the red room sessions",
            "the golden archipelago"
        ]
    },
    "BookRestaurant": {
        "spatial_relation": [
            "distant",
            "close",
            "around"
        ],
        "city": [
            "chicago",
            "laneville",
            "juliff",
            "bowlegs",
            "london",
            "virginia city",
            "denver",
            "foley"
        ],
        "time": [
            "monday"
        ],
        "party_size_number": [
            "three",
            "nine",
            "seven",
            "four",
            "eight",
            "five",
            "9",
            "7"
        ],
        "restaurant_type": [
            "internet restaurant",
            "tea house",
            "food truck",
            "steakhouse",
            "brasserie",
            "restaurant"
        ],
        "served_dish": [
            "fettucine",
            "sashimi",
            "gougere"
        ],
        "country": [
            "seychelles",
            "panama",
            "norway",
            "samoa"
        ],
        "restaurant_name": [
            "coon chicken inn",
            "grecian coffee house"
        ],
        "timeRange": [
            "apr 7th 2024",
            "1 am"
        ],
        "state": [
            "ma"
        ],
        "cuisine": [
            "churrascaria",
            "gluten free"
        ]
    },
    "SearchCreativeWork": {
        "object_type": [
            "television show",
            "tv series",
            "video game",
            "saga",
            "game",
            "movie",
            "picture",
            "soundtrack",
            "book",
            "song",
            "tv show"
        ],
        "object_name": [
            "family dog",
            "heart of gold",
            "operetta for the theatre organ",
            "magic hour",
            "living in america",
            "set sail the prairie",
            "espn major league soccer",
            "heavenly sword",
            "balance and timing",
            "shaun the sheep",
            "twin husbands",
            "ruthless",
            "pax warrior",
            "the young warriors",
            "love will tear us apart",
            "young sheldon"
        ]
    },
    "SearchScreeningEvent": {
        "movie_name": [
            "romulus and the sabines",
            "interstellar",
            "holiday heart",
            "goodbye mothers",
            "dear old girl",
            "the clutching hand",
            "the trouble with girls",
            "paranormal activity 3",
            "now and forever"
        ],
        "location_name": [
            "amco entertainment",
            "moviehouse",
            "harkins theatres",
            "plitt theatres",
            "cooper foundation",
            "southern theatres",
            "imax corporation"
        ],
        "movie_type": [
            "films",
            "animated movies",
            "movies"
        ],
        "spatial_relation": [
            "nearby",
            "in the neighbourhood",
            "nearest",
            "in the neighborhood",
            "closest",
            "local"
        ],
        "timeRange": [
            "ten"
        ]
    },
    "RateBook": {
        "object_name": [
            "lie tree",
            "the improvisatore",
            "suribachi",
            "charlie peace",
            "taste of blackberries",
            "history by contract"
        ],
        "rating_value": [
            "three",
            "5",
            "four",
            "4",
            "zero",
            "two",
            "five",
            "3"
        ],
        "rating_unit": [
            "points",
            "stars"
        ],
        "object_type": [
            "textbook",
            "album",
            "book"
        ],
        "best_rating": [
            "6",
            "six"
        ],
        "object_part_of_series_type": [
            "chronicle"
        ]
    },
    "GetWeather": {
        "city": [
            "faraway",
            "leisure knoll",
            "wyomissing hills",
            "amy",
            "getzville",
            "varnado",
            "lago vista",
            "lost creek",
            "spencer",
            "niceville",
            "grand coteau",
            "waverly city",
            "oakdale"
        ],
        "geographic_poi": [
            "stelvio national park"
        ],
        "timeRange": [
            "october fourteenth 2022",
            "next year",
            "10 pm",
            "sunset",
            "this winter",
            "twenty one minutes",
            "1 hour",
            "six pm",
            "purple heart day",
            "one am",
            "march 13th",
            "12 am"
        ],
        "condition_description": [
            "humidity",
            "foggy",
            "snow",
            "wind",
            "snowfall"
        ],
        "state": [
            "wv",
            "ak",
            "texas",
            "ok",
            "ut",
            "hi",
            "georgia",
            "california",
            "south carolina",
            "delaware",
            "minnesota",
            "fm"
        ],
        "country": [
            "canada",
            "samoa",
            "british virgin islands",
            "bahrain",
            "brazil"
        ],
        "condition_temperature": [
            "freezing",
            "chillier",
            "chilly",
            "hotter"
        ]
    }
}

REQUIRED_SLOTS_MAP = {
    "AddToPlaylist": [], "PlayMusic": [], "BookRestaurant": [],
    "SearchCreativeWork": [], "SearchScreeningEvent": [], "RateBook": [],
    "GetWeather": [],
}

def parse_system_instruction_for_intents(instruction_string: str) -> dict:
    intents_data = {}
    try:
        relevant_section = instruction_string.split("### MixSNIPS Intents & Slots:\n")[1]
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
                               enum_threshold=10) -> list:
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


def get_gpt4_intent_calls_from_query(
        user_query: str,
        extracted_items_dict: dict,
        example_schema_dict: dict,
        system_prompt_for_gpt4: str,
        current_tools_definitions: list,
        current_all_intent_slots: dict,
        openai_client: OpenAI
) -> list[str]:
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
            model="gpt-4.1-mini",
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
        description="Run MixSNIPS baseline with GPT-4 using a command-line API key, pre-extracted items, and schema examples in prompt.")
    parser.add_argument("--api_key", type=str, required=True, help="Your OpenAI API key (e.g., sk-...).")
    parser.add_argument("--input_file", type=str, default="snips_data_augmented_2.json",
                        help="Path to the input JSONL data file.")
    # Changed default output file name to reflect new prompt structure
    parser.add_argument("--output_file", type=str, default="gpt4_baseline_with_all_hints.json",
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
        "You are an expert task-oriented model for the MixSNIPS domain. Parse user utterances into intent calls using the provided functions/tools. "
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
            schema,
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
        output_filename = "gpt4_baseline_with_all_hints.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nFull results log saved to {output_filename}")

    print("\n--- Evaluation Guidance ---")
    print(f"Compare 'answer' (GPT-4's output) with 'expected' in '{output_filename}'.")
    print("This version provided both pre-extracted NER items AND schema examples as HINTS to GPT-4.")
