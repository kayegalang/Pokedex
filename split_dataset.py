import json
import random

random.seed(42)

with open("pokemon_dataset.json", "r") as f:
    dataset = json.load(f)

# Group entries by Pokemon name
pokemon_groups = {}
for entry in dataset:
    name = entry["input"].replace("Name: ", "")
    if name not in pokemon_groups:
        pokemon_groups[name] = []
    pokemon_groups[name].append(entry)

# Shuffle Pokemon names
names = list(pokemon_groups.keys())
random.shuffle(names)

# Split 80/10/10
n = len(names)
train_names = names[:int(n * 0.8)]
val_names = names[int(n * 0.8):int(n * 0.9)]
test_names = names[int(n * 0.9):]

train = [e for name in train_names for e in pokemon_groups[name]]
val = [e for name in val_names for e in pokemon_groups[name]]
test = [e for name in test_names for e in pokemon_groups[name]]

with open("train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("val.json", "w") as f:
    json.dump(val, f, indent=2)

with open("test.json", "w") as f:
    json.dump(test, f, indent=2)

print(f"Train: {len(train)} entries ({len(train_names)} Pokemon)")
print(f"Val: {len(val)} entries ({len(val_names)} Pokemon)")
print(f"Test: {len(test)} entries ({len(test_names)} Pokemon)")