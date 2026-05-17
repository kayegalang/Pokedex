import time
import random
import requests
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

KNOWN_POKEMON = {
    # Gen 1
    "bulbasaur", "ivysaur", "venusaur", "charmander", "charmeleon", "charizard",
    "squirtle", "wartortle", "blastoise", "caterpie", "metapod", "butterfree",
    "weedle", "kakuna", "beedrill", "pidgey", "pidgeotto", "pidgeot", "rattata",
    "raticate", "spearow", "fearow", "ekans", "arbok", "pikachu", "raichu",
    "sandshrew", "sandslash", "nidoran-f", "nidorina", "nidoqueen", "nidoran-m",
    "nidorino", "nidoking", "clefairy", "clefable", "vulpix", "ninetales",
    "jigglypuff", "wigglytuff", "zubat", "golbat", "oddish", "gloom", "vileplume",
    "paras", "parasect", "venonat", "venomoth", "diglett", "dugtrio", "meowth",
    "persian", "psyduck", "golduck", "mankey", "primeape", "growlithe", "arcanine",
    "poliwag", "poliwhirl", "poliwrath", "abra", "kadabra", "alakazam", "machop",
    "machoke", "machamp", "bellsprout", "weepinbell", "victreebel", "tentacool",
    "tentacruel", "geodude", "graveler", "golem", "ponyta", "rapidash", "slowpoke",
    "slowbro", "magnemite", "magneton", "farfetchd", "doduo", "dodrio", "seel",
    "dewgong", "grimer", "muk", "shellder", "cloyster", "gastly", "haunter",
    "gengar", "onix", "drowzee", "hypno", "krabby", "kingler", "voltorb",
    "electrode", "exeggcute", "exeggutor", "cubone", "marowak", "hitmonlee",
    "hitmonchan", "lickitung", "koffing", "weezing", "rhyhorn", "rhydon",
    "chansey", "tangela", "kangaskhan", "horsea", "seadra", "goldeen", "seaking",
    "staryu", "starmie", "mrmime", "scyther", "jynx", "electabuzz", "magmar",
    "pinsir", "tauros", "magikarp", "gyarados", "lapras", "ditto", "eevee",
    "vaporeon", "jolteon", "flareon", "porygon", "omanyte", "omastar", "kabuto",
    "kabutops", "aerodactyl", "snorlax", "articuno", "zapdos", "moltres",
    "dratini", "dragonair", "dragonite", "mewtwo", "mew",
    # Gen 2
    "chikorita", "bayleef", "meganium", "cyndaquil", "quilava", "typhlosion",
    "totodile", "croconaw", "feraligatr", "sentret", "furret", "hoothoot",
    "noctowl", "ledyba", "ledian", "spinarak", "ariados", "crobat", "chinchou",
    "lanturn", "pichu", "cleffa", "igglybuff", "togepi", "togetic", "natu",
    "xatu", "mareep", "flaaffy", "ampharos", "bellossom", "marill", "azumarill",
    "sudowoodo", "politoed", "hoppip", "skiploom", "jumpluff", "aipom", "sunkern",
    "sunflora", "yanma", "wooper", "quagsire", "espeon", "umbreon", "murkrow",
    "slowking", "misdreavus", "unown", "wobbuffet", "girafarig", "pineco",
    "forretress", "dunsparce", "gligar", "steelix", "snubbull", "granbull",
    "qwilfish", "scizor", "shuckle", "heracross", "sneasel", "teddiursa",
    "ursaring", "slugma", "magcargo", "swinub", "piloswine", "corsola",
    "remoraid", "octillery", "delibird", "mantine", "skarmory", "houndour",
    "houndoom", "kingdra", "phanpy", "donphan", "porygon2", "stantler",
    "smeargle", "tyrogue", "hitmontop", "smoochum", "elekid", "magby",
    "miltank", "blissey", "raikou", "entei", "suicune", "larvitar", "pupitar",
    "tyranitar", "lugia", "ho-oh", "celebi"
}

POKEMON_IDS = {
    # Gen 1
    "bulbasaur": 1, "ivysaur": 2, "venusaur": 3, "charmander": 4, "charmeleon": 5,
    "charizard": 6, "squirtle": 7, "wartortle": 8, "blastoise": 9, "caterpie": 10,
    "metapod": 11, "butterfree": 12, "weedle": 13, "kakuna": 14, "beedrill": 15,
    "pidgey": 16, "pidgeotto": 17, "pidgeot": 18, "rattata": 19, "raticate": 20,
    "spearow": 21, "fearow": 22, "ekans": 23, "arbok": 24, "pikachu": 25,
    "raichu": 26, "sandshrew": 27, "sandslash": 28, "nidoran-f": 29, "nidorina": 30,
    "nidoqueen": 31, "nidoran-m": 32, "nidorino": 33, "nidoking": 34, "clefairy": 35,
    "clefable": 36, "vulpix": 37, "ninetales": 38, "jigglypuff": 39, "wigglytuff": 40,
    "zubat": 41, "golbat": 42, "oddish": 43, "gloom": 44, "vileplume": 45,
    "paras": 46, "parasect": 47, "venonat": 48, "venomoth": 49, "diglett": 50,
    "dugtrio": 51, "meowth": 52, "persian": 53, "psyduck": 54, "golduck": 55,
    "mankey": 56, "primeape": 57, "growlithe": 58, "arcanine": 59, "poliwag": 60,
    "poliwhirl": 61, "poliwrath": 62, "abra": 63, "kadabra": 64, "alakazam": 65,
    "machop": 66, "machoke": 67, "machamp": 68, "bellsprout": 69, "weepinbell": 70,
    "victreebel": 71, "tentacool": 72, "tentacruel": 73, "geodude": 74, "graveler": 75,
    "golem": 76, "ponyta": 77, "rapidash": 78, "slowpoke": 79, "slowbro": 80,
    "magnemite": 81, "magneton": 82, "farfetchd": 83, "doduo": 84, "dodrio": 85,
    "seel": 86, "dewgong": 87, "grimer": 88, "muk": 89, "shellder": 90,
    "cloyster": 91, "gastly": 92, "haunter": 93, "gengar": 94, "onix": 95,
    "drowzee": 96, "hypno": 97, "krabby": 98, "kingler": 99, "voltorb": 100,
    "electrode": 101, "exeggcute": 102, "exeggutor": 103, "cubone": 104, "marowak": 105,
    "hitmonlee": 106, "hitmonchan": 107, "lickitung": 108, "koffing": 109, "weezing": 110,
    "rhyhorn": 111, "rhydon": 112, "chansey": 113, "tangela": 114, "kangaskhan": 115,
    "horsea": 116, "seadra": 117, "goldeen": 118, "seaking": 119, "staryu": 120,
    "starmie": 121, "mrmime": 122, "scyther": 123, "jynx": 124, "electabuzz": 125,
    "magmar": 126, "pinsir": 127, "tauros": 128, "magikarp": 129, "gyarados": 130,
    "lapras": 131, "ditto": 132, "eevee": 133, "vaporeon": 134, "jolteon": 135,
    "flareon": 136, "porygon": 137, "omanyte": 138, "omastar": 139, "kabuto": 140,
    "kabutops": 141, "aerodactyl": 142, "snorlax": 143, "articuno": 144, "zapdos": 145,
    "moltres": 146, "dratini": 147, "dragonair": 148, "dragonite": 149, "mewtwo": 150,
    "mew": 151,
    # Gen 2
    "chikorita": 152, "bayleef": 153, "meganium": 154, "cyndaquil": 155,
    "quilava": 156, "typhlosion": 157, "totodile": 158, "croconaw": 159,
    "feraligatr": 160, "sentret": 161, "furret": 162, "hoothoot": 163,
    "noctowl": 164, "ledyba": 165, "ledian": 166, "spinarak": 167,
    "ariados": 168, "crobat": 169, "chinchou": 170, "lanturn": 171,
    "pichu": 172, "cleffa": 173, "igglybuff": 174, "togepi": 175,
    "togetic": 176, "natu": 177, "xatu": 178, "mareep": 179,
    "flaaffy": 180, "ampharos": 181, "bellossom": 182, "marill": 183,
    "azumarill": 184, "sudowoodo": 185, "politoed": 186, "hoppip": 187,
    "skiploom": 188, "jumpluff": 189, "aipom": 190, "sunkern": 191,
    "sunflora": 192, "yanma": 193, "wooper": 194, "quagsire": 195,
    "espeon": 196, "umbreon": 197, "murkrow": 198, "slowking": 199,
    "misdreavus": 200, "unown": 201, "wobbuffet": 202, "girafarig": 203,
    "pineco": 204, "forretress": 205, "dunsparce": 206, "gligar": 207,
    "steelix": 208, "snubbull": 209, "granbull": 210, "qwilfish": 211,
    "scizor": 212, "shuckle": 213, "heracross": 214, "sneasel": 215,
    "teddiursa": 216, "ursaring": 217, "slugma": 218, "magcargo": 219,
    "swinub": 220, "piloswine": 221, "corsola": 222, "remoraid": 223,
    "octillery": 224, "delibird": 225, "mantine": 226, "skarmory": 227,
    "houndour": 228, "houndoom": 229, "kingdra": 230, "phanpy": 231,
    "donphan": 232, "porygon2": 233, "stantler": 234, "smeargle": 235,
    "tyrogue": 236, "hitmontop": 237, "smoochum": 238, "elekid": 239,
    "magby": 240, "miltank": 241, "blissey": 242, "raikou": 243,
    "entei": 244, "suicune": 245, "larvitar": 246, "pupitar": 247,
    "tyranitar": 248, "lugia": 249, "ho-oh": 250, "celebi": 251
}

TYPE_CHART = {
    "Normal":   {"weak": ["Fighting"], "strong": []},
    "Fire":     {"weak": ["Water", "Rock", "Ground"], "strong": ["Grass", "Ice", "Bug", "Steel"]},
    "Water":    {"weak": ["Electric", "Grass"], "strong": ["Fire", "Ground", "Rock"]},
    "Electric": {"weak": ["Ground"], "strong": ["Water", "Flying"]},
    "Grass":    {"weak": ["Fire", "Ice", "Poison", "Flying", "Bug"], "strong": ["Water", "Ground", "Rock"]},
    "Ice":      {"weak": ["Fire", "Fighting", "Rock", "Steel"], "strong": ["Grass", "Ground", "Flying", "Dragon"]},
    "Fighting": {"weak": ["Flying", "Psychic", "Fairy"], "strong": ["Normal", "Ice", "Rock", "Dark", "Steel"]},
    "Poison":   {"weak": ["Ground", "Psychic"], "strong": ["Grass", "Fairy"]},
    "Ground":   {"weak": ["Water", "Grass", "Ice"], "strong": ["Fire", "Electric", "Poison", "Rock", "Steel"]},
    "Flying":   {"weak": ["Electric", "Ice", "Rock"], "strong": ["Grass", "Fighting", "Bug"]},
    "Psychic":  {"weak": ["Bug", "Ghost", "Dark"], "strong": ["Fighting", "Poison"]},
    "Bug":      {"weak": ["Fire", "Flying", "Rock"], "strong": ["Grass", "Psychic", "Dark"]},
    "Rock":     {"weak": ["Water", "Grass", "Fighting", "Ground", "Steel"], "strong": ["Fire", "Ice", "Flying", "Bug"]},
    "Ghost":    {"weak": ["Ghost", "Dark"], "strong": ["Psychic", "Ghost"]},
    "Dragon":   {"weak": ["Ice", "Dragon", "Fairy"], "strong": ["Dragon"]},
    "Dark":     {"weak": ["Fighting", "Bug", "Fairy"], "strong": ["Psychic", "Ghost"]},
    "Steel":    {"weak": ["Fire", "Fighting", "Ground"], "strong": ["Ice", "Rock", "Fairy"]},
    "Fairy":    {"weak": ["Poison", "Steel"], "strong": ["Fighting", "Dragon", "Dark"]},
}

TYPE_COLORS = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0",
    "Electric": "#F8D030", "Grass": "#78C850", "Ice": "#98D8D8",
    "Fighting": "#C03028", "Poison": "#A040A0", "Ground": "#E0C068",
    "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8",
    "Dark": "#705848", "Steel": "#B8B8D0", "Fairy": "#EE99AC",
}

def get_type_matchups(type_str):
    types = [t.strip() for t in type_str.replace("Type:", "").strip().split("/")]
    all_weak, all_strong = set(), set()
    for t in types:
        t_cap = t.capitalize()
        if t_cap in TYPE_CHART:
            all_weak.update(TYPE_CHART[t_cap]["weak"])
            all_strong.update(TYPE_CHART[t_cap]["strong"])
    all_weak -= all_strong
    return sorted(all_weak), sorted(all_strong), types

def make_type_badges(types):
    badges = ""
    for t in types:
        t_cap = t.capitalize()
        color = TYPE_COLORS.get(t_cap, "#888")
        badges += f'<span style="background:{color};color:white;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:bold;margin:3px;display:inline-block;text-shadow:0 1px 2px rgba(0,0,0,0.4);">{t_cap}</span>'
    return badges

def make_matchup_badges(types):
    if not types:
        return '<span style="color:#888;font-style:italic;font-size:13px;">None</span>'
    badges = ""
    for t in types:
        t_cap = t.capitalize()
        tc = TYPE_COLORS.get(t_cap, "#888")
        badges += f'<span style="background:{tc};color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:bold;margin:2px;display:inline-block;text-shadow:0 1px 2px rgba(0,0,0,0.4);">{t_cap}</span>'
    return badges

def get_sprite_url(name_lower, shiny=False):
    pokemon_id = POKEMON_IDS.get(name_lower)
    if not pokemon_id:
        return None
    if shiny:
        return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/shiny/{pokemon_id}.gif"
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-v/black-white/animated/{pokemon_id}.gif"

def get_cry_url(name_lower):
    pokemon_id = POKEMON_IDS.get(name_lower)
    if not pokemon_id:
        return None
    return f"https://raw.githubusercontent.com/PokeAPI/cries/main/cries/pokemon/latest/{pokemon_id}.ogg"

def get_evolution_chain(name_lower):
    try:
        species_url = f"https://pokeapi.co/api/v2/pokemon-species/{name_lower}/"
        species_resp = requests.get(species_url, timeout=5)
        if species_resp.status_code != 200:
            return None
        evolution_url = species_resp.json()["evolution_chain"]["url"]
        evo_resp = requests.get(evolution_url, timeout=5)
        if evo_resp.status_code != 200:
            return None
        chain = evo_resp.json()["chain"]
        evolutions = []
        def parse_chain(node):
            evolutions.append(node["species"]["name"])
            for next_evo in node["evolves_to"]:
                parse_chain(next_evo)
        parse_chain(chain)
        return evolutions
    except:
        return None

print("Loading Pokedex model...")
tokenizer = AutoTokenizer.from_pretrained("./pokedex-lora")
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, "./pokedex-lora")
model.eval()
print("Model ready!")

POKEBALL_LOADER = """
<style>
@keyframes pokespin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
</style>
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:48px;gap:16px;">
    <div style="
        width:60px;
        height:60px;
        border-radius:50%;
        border:5px solid #333;
        background:linear-gradient(180deg, #CC0000 0%, #CC0000 50%, #ffffff 50%, #ffffff 100%);
        animation:pokespin 0.9s linear infinite;
        position:relative;
        box-shadow:0 2px 12px rgba(0,0,0,0.15);
    ">
        <div style="
            position:absolute;
            top:50%;
            left:0;
            right:0;
            height:5px;
            background:#333;
            transform:translateY(-50%);
        "></div>
        <div style="
            position:absolute;
            top:50%;
            left:50%;
            width:16px;
            height:16px;
            background:white;
            border:4px solid #333;
            border-radius:50%;
            transform:translate(-50%,-50%);
        "></div>
    </div>
    <span style="font-family:'Courier New',monospace;color:#CC0000;font-size:13px;letter-spacing:2px;">SCANNING...</span>
    <span style="font-family:'Courier New',monospace;color:#888;font-size:11px;letter-spacing:1px;">If this takes a while, another trainer is ahead of you in line!</span>
</div>
"""

WELCOME_SCREEN = """
<div style="text-align:center;padding:32px;font-family:'Courier New',monospace;">
    <div style="font-size:56px;margin-bottom:16px;">🔴</div>
    <div style="font-size:16px;font-weight:900;color:#CC0000;letter-spacing:3px;margin-bottom:12px;">WELCOME TO POKAIDEX</div>
    <div style="font-size:13px;color:#888;line-height:2;">
        Enter a <strong style="color:#555;">Gen 1 or Gen 2</strong> Pokémon name to scan it.<br>
        Try <strong style="color:#555;">Pikachu</strong>, <strong style="color:#555;">Lugia</strong>, or <strong style="color:#555;">Charizard</strong>.<br>
        Or hit <strong style="color:#555;">🎲 RANDOM</strong> for a surprise!<br><br>
        <span style="color:#B8860B;">✨ Pro tip: type <strong>shiny pikachu</strong> for a surprise!</span>
    </div>
</div>
"""

CONFETTI_HTML = """
<style>
@keyframes confetti-fall {
    0% { transform: translateY(-100px) rotate(0deg); opacity: 1; }
    100% { transform: translateY(600px) rotate(720deg); opacity: 0; }
}
.confetti-piece {
    position: fixed;
    width: 10px;
    height: 10px;
    top: 0;
    animation: confetti-fall 1.5s ease-in forwards;
    z-index: 9999;
    pointer-events: none;
}
</style>
<div id="confetti-container"></div>
<script>
(function() {
    var container = document.getElementById('confetti-container');
    var colors = ['#FFD700', '#FFA500', '#FF69B4', '#00CED1', '#FF6347', '#7B68EE'];
    for (var i = 0; i < 60; i++) {
        var piece = document.createElement('div');
        piece.className = 'confetti-piece';
        piece.style.left = Math.random() * 100 + 'vw';
        piece.style.background = colors[Math.floor(Math.random() * colors.length)];
        piece.style.animationDelay = Math.random() * 1.5 + 's';
        piece.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
        container.appendChild(piece);
    }
    setTimeout(function() {
        if (container) container.remove();
    }, 3000);
})();
</script>
"""

def generate_entry(pokemon_name):
    if not pokemon_name.strip():
        return WELCOME_SCREEN

    raw_input = pokemon_name.strip().lower()

    shiny = False
    if raw_input.startswith("shiny "):
        shiny = True
        name_lower = raw_input[6:].strip()
    else:
        name_lower = raw_input

    if name_lower not in KNOWN_POKEMON:
        suggestions = random.sample(list(KNOWN_POKEMON), 3)
        suggestions_html = ", ".join(f"<strong style='color:#555;'>{s.capitalize()}</strong>" for s in suggestions)
        return f"""
        <div style="text-align:center;padding:32px;font-family:'Courier New',monospace;">
            <div style="font-size:48px;margin-bottom:12px;">❓</div>
            <div style="font-size:18px;font-weight:bold;color:#CC0000;letter-spacing:2px;margin-bottom:8px;">NOT FOUND</div>
            <div style="font-size:13px;color:#888;line-height:1.6;">
                <strong style="color:#555;">{pokemon_name.capitalize()}</strong> is not in this Pokedex.<br>
                This model covers <strong style="color:#555;">Generation 1 and 2</strong> Pokemon only.<br><br>
                How about trying: {suggestions_html}?<br><br>
                Or try: <strong style="color:#B8860B;">shiny pikachu</strong> ✨
            </div>
        </div>
        """

    prompt = f"<|user|>\nName: {name_lower.capitalize()}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = decoded.split("<|assistant|>")[-1].strip()

    if raw.startswith("Type:"):
        parts = raw.split(".", 1)
        type_part = parts[0].strip()
        flavor = parts[1].strip() if len(parts) > 1 else ""
    else:
        type_part = "Type: Unknown"
        flavor = raw

    weak, strong, types = get_type_matchups(type_part)
    type_badges = make_type_badges(types)
    weak_badges = make_matchup_badges(weak)
    strong_badges = make_matchup_badges(strong)

    sprite_url = get_sprite_url(name_lower, shiny=shiny)
    cry_url = get_cry_url(name_lower)
    dex_number = POKEMON_IDS.get(name_lower, "?")
    gen = "GEN 1" if dex_number <= 151 else "GEN 2"

    evo_chain = get_evolution_chain(name_lower)
    if evo_chain and len(evo_chain) > 1:
        evo_parts = []
        for evo in evo_chain:
            evo_id = POKEMON_IDS.get(evo)
            if evo_id:
                evo_sprite = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{evo_id}.png"
                is_current = evo == name_lower
                border = "2px solid #CC0000" if is_current else "2px solid transparent"
                evo_parts.append(f"""
                    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
                        <img src="{evo_sprite}" style="width:48px;height:48px;object-fit:contain;image-rendering:pixelated;border:{border};border-radius:8px;"/>
                        <span style="font-size:10px;color:{'#CC0000' if is_current else '#888'};font-weight:{'bold' if is_current else 'normal'};">{evo.upper()}</span>
                    </div>
                """)
        arrow = '<span style="color:#aaa;font-size:18px;align-self:center;">→</span>'
        evo_html = f"""
        <div class="entry-box" style="border-radius:8px;padding:12px;margin-bottom:16px;">
            <div style="font-size:11px;color:#CC0000;font-weight:bold;letter-spacing:2px;margin-bottom:10px;">EVOLUTION CHAIN</div>
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                {arrow.join(evo_parts)}
            </div>
        </div>
        """
    else:
        evo_html = ""

    sprite_html = f"""
        <img src="{sprite_url}"
             style="width:96px;height:96px;object-fit:contain;image-rendering:pixelated;"
             onerror="this.style.display='none'"/>
    """ if sprite_url else ""

    cry_html = f"""
        <audio autoplay style="display:none">
            <source src="{cry_url}" type="audio/ogg">
        </audio>
    """ if cry_url else ""

    shiny_badge = '<span style="background:linear-gradient(135deg,#FFD700,#FFA500);color:white;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:bold;margin-left:6px;text-shadow:0 1px 2px rgba(0,0,0,0.3);">✨ SHINY</span>' if shiny else ""

    confetti_html = CONFETTI_HTML if shiny else ""

    sprite_box_style = """
        background: linear-gradient(135deg, #fff9e6, #fff3cc);
        border: 2px solid #FFD700;
        border-radius: 12px;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(255,215,0,0.5);
    """ if shiny else """
        background: #f0f0f0;
        border: 2px solid #ddd;
        border-radius: 12px;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    """

    return f"""
    {cry_html}
    {confetti_html}
    <div style="font-family:'Courier New',monospace;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
            <div style="{sprite_box_style}">
                {sprite_html}
            </div>
            <div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
                    <span style="font-size:11px;color:#aaa;letter-spacing:2px;">#{str(dex_number).zfill(3)}</span>
                    <span style="background:#CC0000;color:white;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:bold;letter-spacing:1px;">{gen}</span>
                </div>
                <div style="display:flex;align-items:center;margin-bottom:8px;">
                    <span style="font-size:20px;font-weight:900;color:#CC0000;letter-spacing:2px;">
                        {name_lower.upper()}
                    </span>
                    {shiny_badge}
                </div>
                <div style="font-size:11px;color:#CC0000;font-weight:bold;letter-spacing:2px;margin-bottom:6px;">TYPE</div>
                {type_badges}
            </div>
        </div>
        {evo_html}
        <div class="entry-box" style="border-radius:8px;padding:16px;margin-bottom:16px;">
            <div style="font-size:11px;color:#CC0000;font-weight:bold;letter-spacing:2px;margin-bottom:8px;">POKEDEX ENTRY</div>
            <p style="font-size:14px;line-height:1.7;margin:0 0 10px 0;">{flavor}</p>
            <div style="font-size:11px;color:#aaa;font-style:italic;">
                ⚠️ This entry is AI-generated and may not be accurate.
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div class="entry-box" style="border-radius:8px;padding:12px;">
                <div style="font-size:11px;color:#CC0000;font-weight:bold;letter-spacing:2px;margin-bottom:8px;">WEAK TO</div>
                {weak_badges}
            </div>
            <div class="entry-box" style="border-radius:8px;padding:12px;">
                <div style="font-size:11px;color:#4CAF50;font-weight:bold;letter-spacing:2px;margin-bottom:8px;">STRONG AGAINST</div>
                {strong_badges}
            </div>
        </div>
    </div>
    """

css = """
:root {
    --bg: #f5f5f5;
    --surface: #ffffff;
    --border: #e0e0e0;
    --text: #1a1a1a;
    --subtext: #555;
    --entry-bg: #f9f9f9;
    --entry-border: #ddd;
}

.dark-mode {
    --bg: #0a0a0a;
    --surface: #111111;
    --border: #333;
    --text: #e0e0e0;
    --subtext: #aaa;
    --entry-bg: #1a1a1a;
    --entry-border: #333;
}

html,
body,
.gradio-container,
#root,
.main,
.wrap,
.app {
    background: var(--bg) !important;
    transition: background 0.3s !important;
}

.gradio-container {
    max-width: 600px !important;
    margin: 0 auto !important;
    font-family: 'Courier New', monospace !important;
}

.progress-bar, .eta-bar, footer,
.generating, [class*="progress"],
.meta-text, .meta-text-center,
.loader, .loading,
svg.loading, .spinner,
[class*="spinner"], [class*="loader"],
.wrap.default.translucent,
.wrap.default.translucent svg,
.svelte-1ed2p3z,
[class*="status"],
.output-class > .wrap,
.pending,
.wrap svg,
.icon-wrap,
.eta-bar-wrap,
gradio-wasm-error,
.upload-container .wrap {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
}

.pokedex-wrap {
    border-radius: 16px;
    overflow: visible;
    border: 3px solid #CC0000;
    box-shadow: 0 8px 32px rgba(204,0,0,0.2);
}

.pokedex-header {
    background: linear-gradient(135deg, #CC0000 0%, #990000 100%);
    padding: 20px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 4px solid #660000;
    border-radius: 13px 13px 0 0;
}

.pokedex-body {
    background: var(--surface);
    padding: 24px;
    transition: background 0.3s;
    border-radius: 0 0 13px 13px;
    overflow: hidden;
}

.entry-box {
    background: var(--entry-bg) !important;
    border: 2px solid var(--entry-border) !important;
    color: var(--text) !important;
    transition: background 0.3s, border 0.3s;
}

p { color: var(--text) !important; transition: color 0.3s; }

input[type="text"] {
    background: var(--entry-bg) !important;
    border: 2px solid #CC0000 !important;
    color: var(--text) !important;
    font-family: 'Courier New', monospace !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    transition: background 0.3s, color 0.3s !important;
}

input[type="text"]:focus {
    border-color: #FF4444 !important;
    box-shadow: 0 0 12px rgba(204,0,0,0.3) !important;
}

button.primary {
    background: linear-gradient(135deg, #CC0000, #990000) !important;
    border: none !important;
    color: white !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #FF2222, #CC0000) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(204,0,0,0.4) !important;
}

button.secondary {
    background: linear-gradient(135deg, #444, #222) !important;
    border: none !important;
    color: white !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

button.secondary:hover {
    background: linear-gradient(135deg, #666, #444) !important;
    transform: translateY(-1px) !important;
}

footer { display: none !important; }
"""

with gr.Blocks(title="PokAIdex") as app:

    gr.HTML("""
    <div class="pokedex-wrap">
        <div class="pokedex-header">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="width:32px;height:32px;background:white;border-radius:50%;border:4px solid #ddd;box-shadow:0 0 0 3px #bbb;"></div>
                <div style="display:flex;gap:6px;">
                    <div style="width:12px;height:12px;background:#FF4444;border-radius:50%;"></div>
                    <div style="width:12px;height:12px;background:#FFFF44;border-radius:50%;"></div>
                    <div style="width:12px;height:12px;background:#44FF44;border-radius:50%;"></div>
                </div>
            </div>
            <div style="text-align:center;">
                <h1 style="color:#ffffff;font-family:'Courier New',monospace;font-size:26px;font-weight:900;letter-spacing:6px;margin:0;text-shadow:0 2px 8px rgba(0,0,0,0.4);">POKAIDEX</h1>
            </div>
            <button onclick="
                document.body.classList.toggle('dark-mode');
                this.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
            " style="background:rgba(255,255,255,0.15);border:2px solid rgba(255,255,255,0.4);color:white;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:18px;transition:all 0.2s;">🌙</button>
        </div>
        <div class="pokedex-body">
    """)

    name_input = gr.Textbox(
        placeholder="Enter a Gen 1 or 2 Pokemon... or try 'shiny lugia' ✨",
        label="",
        show_label=False
    )

    with gr.Row():
        search_btn = gr.Button("SCAN POKEMON", variant="primary")
        random_btn = gr.Button("🎲 RANDOM", variant="secondary")

    queue_status = gr.HTML(value="")
    output = gr.HTML(value=WELCOME_SCREEN)

    gr.HTML("</div></div>")

    def generate_with_loader(pokemon_name):
        cache_buster = random.randint(0, 999999)
        yield "", '<div style="text-align:center;font-family:\'Courier New\',monospace;font-size:12px;color:#888;padding:4px;">⏳ Waiting in queue...</div>'
        time.sleep(0.05)
        yield POKEBALL_LOADER + f"<!-- {cache_buster} -->", '<div style="text-align:center;font-family:\'Courier New\',monospace;font-size:12px;color:#CC0000;padding:4px;">🔴 Scanning now...</div>'
        time.sleep(1.5)
        yield generate_entry(pokemon_name), ""

    def pick_random():
        return random.choice(list(KNOWN_POKEMON))

    search_btn.click(
        fn=generate_with_loader,
        inputs=name_input,
        outputs=[output, queue_status],
        concurrency_limit=1
    )

    name_input.submit(
        fn=generate_with_loader,
        inputs=name_input,
        outputs=[output, queue_status],
        concurrency_limit=1
    )

    random_btn.click(
        fn=lambda: gr.update(interactive=False),
        outputs=random_btn
    ).then(
        fn=pick_random,
        outputs=name_input
    ).then(
        fn=generate_with_loader,
        inputs=name_input,
        outputs=[output, queue_status],
        concurrency_limit=1
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=random_btn
    )

app.launch(
    css=css,
    share=True,
    max_threads=1
)