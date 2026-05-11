import requests
import json
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://pokeapi.co/api/v2"

def get_pokemon_data(pokemon_id):
    response = requests.get(f"{BASE_URL}/pokemon/{pokemon_id}")
    if response.status_code != 200:
        return None
    return response.json()

def get_species_data(pokemon_id):
    response = requests.get(f"{BASE_URL}/pokemon-species/{pokemon_id}")
    if response.status_code != 200:
        return None
    return response.json()

def build_dataset(num_pokemon=251):
    dataset = []
    type_lookup = {}

    for i in range(1, num_pokemon + 1):
        print(f"Fetching Pokemon {i}/{num_pokemon}...")

        poke_data = get_pokemon_data(i)
        species_data = get_species_data(i)

        if not poke_data or not species_data:
            continue

        name = poke_data["name"].capitalize()
        types = [t["type"]["name"].capitalize() for t in poke_data["types"]]
        type_str = "/".join(types)

        # Save to type lookup
        type_lookup[poke_data["name"]] = type_str

        flavor_texts = [
            entry["flavor_text"].replace("\n", " ").replace("\f", " ")
            for entry in species_data["flavor_text_entries"]
            if entry["language"]["name"] == "en"
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_flavors = []
        for text in flavor_texts:
            if text not in seen:
                seen.add(text)
                unique_flavors.append(text)

        # Filter out very short entries
        unique_flavors = [f for f in unique_flavors if len(f) > 40]

        for flavor in unique_flavors:
            dataset.append({
                "input": f"Name: {name}. Type: {type_str}.",
                "output": flavor
            })

        time.sleep(0.5)

    # Save type lookup separately
    with open("pokemon_types.json", "w") as f:
        json.dump(type_lookup, f, indent=2)
    print(f"Saved type lookup for {len(type_lookup)} Pokemon to pokemon_types.json")

    return dataset

dataset = build_dataset(251)

with open("pokemon_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nDone! Collected {len(dataset)} training pairs.")