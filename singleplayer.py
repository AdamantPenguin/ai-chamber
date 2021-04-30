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


print("Loading, please wait...")

from transformers import pipeline
import os
GENERATOR = pipeline('text-generation', model='gpt2')


# generation function for ease of use
def generate_text(history):
    return GENERATOR(
            history,
            num_return_sequences=1,
            pad_token_id=50256,
            max_length=len(history)/3  # note to self: fix soon
        )[0]['generated_text']


# clear screen function
def clear():
    os.system("cls||clear")  # super neat clear terminal thing right here


# get prompt (starting text/context)
clear()
print("Enter the prompt text below:")
text_history = input()
# text_history = generate_text(text_history)  # initial generation
clear()
print(text_history)

while True:
    action = input("> ")
    text_history += f' {action}\n'
    generated = generate_text(text_history)
    new_text = generated.replace(text_history, '')
    print(new_text)
    text_history = generated
