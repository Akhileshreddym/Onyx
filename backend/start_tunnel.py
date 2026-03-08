import pty
import os
import re
import time
import subprocess

print("Starting localhost.run SSH tunnel...")

master, slave = pty.openpty()
process = subprocess.Popen(
    ['ssh', '-o', 'StrictHostKeyChecking=no', '-R', '80:localhost:8000', 'nokey@localhost.run'],
    stdin=slave, stdout=slave, stderr=slave, close_fds=True
)
os.close(slave)

url = None
start_time = time.time()
while time.time() - start_time < 15:
    try:
        data = os.read(master, 1024).decode('utf-8', errors='ignore')
        print(data, end='')
        match = re.search(r'http[s]?://[a-zA-Z0-9.-]+\.lhr\.life', data)
        if match:
            url = match.group(0)
            break
    except Exception:
        time.sleep(0.1)

if url:
    print(f"\n[+] Found localhost.run URL: {url}")
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
