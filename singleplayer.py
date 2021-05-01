"""
Copyright 2021 AdamantPenguin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


print("Loading, please wait...")  # it takes a while to load the AI

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os
import random
import sys

MODEL_NAME = 'EleutherAI/gpt-neo-125M'
MODEL = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)

safemode = False  # TODO: Make safe mode actually do something
temperature = 1.0

# help message for /help
HELP = f"""
[AIC help text]
/help - show this help
/remember <thing> - add something to memory
/forget <thing> - remove something from memory
/delete <thing> - erase something from the story
/rename <old> <new> - rename a character or object
/safemode - toggle safe mode (censor things you might not want to see)
          | Safe mode is currently not implemented.
/temperature <new_temperature> - change the AI's 'inventiveness'
""".strip()

# things that trigger death
DEATH_KWS = [
    "You die", "you die",
    "You get killed", "you get killed",
    "You got killed", "you got killed",
    "You aren't alive", "you aren't alive",
    "You are not alive", "you are not alive",
    "You're dead", "you're dead",
    "You are dead", "you are dead"
]

# things which trigger remembering
IMPORTANT_KWS = [
    "You are", "you are",
    "You're", "you're",
    "You have", "you have",
    "You can", "you can"
]


# generation function for making text exist
def generate_text(history, temp):
    input_ids = TOKENIZER(history, return_tensors='pt')['input_ids']
    hist_length = len(input_ids)
    gen_tokens = MODEL.generate(
        input_ids,
        do_sample=True,
        temperature=temp,
        max_length=hist_length + 200,
        pad_token_id=50256
    )
    gen_text = TOKENIZER.batch_decode(gen_tokens)[0]
    return gen_text


# remove unfinished sentences from the end of text
def chop_sentences(text: str):
    # TODO: Make this work with other punctuation
    return '.'.join(text.split('.')[:-1]) + '.'


# clear screen function
def clear():
    os.system("cls||clear")  # super neat clear terminal thing right here


def game():
    global safemode, temperature
    remember = ""  # things to remember
    # get prompt (starting text/context)
    clear()
    print("Enter the prompt text below:")
    text_history = input()
    # text_history = generate_text(text_history, temperature)  # initial generation
    remember = text_history  # put the prompt in memory to keep it relevant
    clear()
    print(text_history)

    while True:
        action = input("> ").strip()
        # commands
        if action.startswith("/remember"):
            remember += " " + action.replace("/remember", "").strip()
            print("[AIC] Added that to my memory.")
            continue

        elif action.startswith("/forget"):
            forget_this = action.replace("/forget", "").strip()
            remember = remember.replace(forget_this, "")
            print("[AIC] Removed that from my memory.")
            continue
            
        elif action.startswith("/delete"):
            delete_this = action.replace("/delete", "").strip()
            text_history = text_history.replace(delete_this, "")
            print("[AIC] Deleted that from the story.")
            continue

        elif action.startswith("/rename"):
            parts = action.split(" ")
            try:
                text_history = text_history.replace(parts[1], parts[2])
                remember = remember.replace(parts[1], parts[2])
                print(f"[AIC] Renamed {parts[1]} to {parts[2]}.")
            except IndexError:
                print("[AIC] Invalid syntax.")
            continue

        elif action.startswith("/safemode"):
            safemode = not safemode
            print(f"[AIC] Set safe mode to {safemode}.")
            continue

        elif action.startswith("/temperature"):
            new_temp = float(action.replace("/temperature", "").strip())
            if new_temp < 0 or new_temp > 3:
                print("[AIC] Temperature must be between 0 and 3")
                continue
            temperature = new_temp
            print(f"[AIC] Set the temperature to {temperature}")
            continue

        elif action.startswith("/help"):
            print(HELP)
            continue

        elif action.startswith("/"):  # unrecognised command
            print("[AIC] Unknown command. Use /help for help.")
            continue

        # not commands (normal generation)
        text_history += " " + action
        gen_input = remember + '\n' + text_history[-100:]  # provide 100 chars
        generated = generate_text(gen_input, temperature)
        new_text = chop_sentences(
                generated.replace(gen_input, '')
            ).strip()
        print(new_text)
        text_history += new_text

        # death checking
        for death_kw in DEATH_KWS:
            if death_kw in new_text:
                print("[AIC] Game over! You died.")
                sys.exit()
        
        # auto-remember important things
        for important_kw in IMPORTANT_KWS:
            if important_kw in new_text:
                important_start = new_text.find(important_kw)
                important_end = new_text[important_start:].find(".")\
                    + important_start + 1
                new_remember = new_text[important_start:important_end]
                remember += new_remember

                # debugging
                print(f"[AIC Debug] Auto-remembered '{new_remember}'.")


if __name__ == '__main__':
    goodbye_messages = [
        "You quit the game.",
        "You got bored and left.",
        "You said bye to the AI.",
        "You said AI Dungeon was better and got ejected.",
        "You quit.",
        "You left.",
        "You used your keyboard to exit the program.",
        "You forgot how to play the game.",
        "You vanished from existence.",
        "You walked away.",
        "You said goodbye."
    ]

    try:
        game()
    except EOFError:  # CTRL-D
        print(random.choice(goodbye_messages))
    except KeyboardInterrupt:  # CTRL-C
        print(random.choice(goodbye_messages))
