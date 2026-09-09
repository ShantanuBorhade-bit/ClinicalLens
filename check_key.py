"""
Check whether the GEMINI_API_KEY in .env actually works with Google.

Run from the project root:
    venv\\Scripts\\python.exe check_key.py

Never prints the key itself — only a verdict.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252 and crash on emoji — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

key = (os.getenv("GEMINI_API_KEY") or "").strip()

if not key:
    print("❌ NO KEY: GEMINI_API_KEY is missing or empty in .env")
    sys.exit(1)

if "your_gemini_api_key_here" in key:
    print("❌ PLACEHOLDER: .env still contains the template value — paste your real key.")
    sys.exit(1)

if not key.startswith("AIza"):
    print("⚠️  WARNING: Google API keys usually start with 'AIza' — double-check what you pasted.")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=1"
try:
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.load(resp)
    print("✅ KEY VALID — Google accepted it. Restart uvicorn and you're good to go.")
    sys.exit(0)
except urllib.error.HTTPError as exc:
    body = exc.read().decode(errors="replace")
    if exc.code == 400 or "API key not valid" in body:
        print("❌ KEY INVALID: Google rejected it (API key not valid).")
        print("   → Create a fresh key at https://aistudio.google.com/apikey")
        print("   → Make sure you copied the whole 'AIza...' string with no spaces/quotes.")
    elif exc.code == 403:
        print("❌ KEY REJECTED (403): Generative Language API not enabled for this")
        print("   project, key restrictions block this machine, or your region is")
        print("   unsupported. In AI Studio, create the key in a NEW project, and")
        print("   check 'API restrictions' in the key settings.")
    else:
        print(f"❌ HTTP {exc.code}: {body[:300]}")
    sys.exit(2)
except urllib.error.URLError as exc:
    print(f"❌ NETWORK ERROR reaching Google: {exc.reason}")
    sys.exit(3)
