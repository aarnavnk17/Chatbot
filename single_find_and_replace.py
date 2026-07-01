# single_find_and_replace.py
import re

def single_find_and_replace(filepath, old_string, new_string):
    with open(filepath, 'r') as file:
        content = file.read()

    content = re.sub(old_string, new_string, content)

    with open(filepath, 'w') as file:
        file.write(content)