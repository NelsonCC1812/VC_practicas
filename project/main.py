import sys
import subprocess

res = subprocess.run(['python', '_main.py', *sys.argv[1::]], capture_output=True, text=True)
print(res.stdout.split('\\#')[1].strip())