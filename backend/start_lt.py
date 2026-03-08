import re
import time
import subprocess

print("Starting localtunnel...")

process = subprocess.Popen(
    ['npx', 'localtunnel', '--port', '8000'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

url = None
start_time = time.time()

for line in iter(process.stdout.readline, ''):
    print(line, end='')
    match = re.search(r'http[s]?://[a-zA-Z0-9.-]+\.loca\.lt', line)
    if match:
        url = match.group(0)
        break
    if time.time() - start_time > 15:
        break

if url:
    print(f"\n[+] Found Localtunnel URL: {url}")
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'BASE_URL=' in content:
        content = re.sub(r'BASE_URL=.*', f'BASE_URL={url}', content)
    else:
        content += f"\nBASE_URL={url}\n"
        
    with open('.env', 'w') as f:
        f.write(content)
    print("[+] Updated .env with new BASE_URL")
else:
    print("\n[-] Failed to find URL")
