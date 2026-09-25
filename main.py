import uvicorn
import webbrowser
import threading
import time
import sys

def open_browser():
    time.sleep(1.5)
    print("\n[+] Opening MEXC Scalper Dashboard in your default browser...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 65)
    print("      MEXC SCALPER PRO 25% - AUTONOMOUS TRADING BOT")
    print("      Target: 20-25% Monthly Return (~1% Daily)")
    print("      URL:    http://localhost:8000")
    print("=" * 65)

    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI server
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
