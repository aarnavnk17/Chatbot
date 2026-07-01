# grep_search.py
import subprocess

def grep_search(query):
    result = subprocess.run(['rg', query], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')