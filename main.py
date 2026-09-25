import os
import sys
import uvicorn
import webbrowser
import threading
import time

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

    # Launch browser automatically if graphical display exists
    try:
        if "DISPLAY" in os.environ or sys.platform.startswith("win"):
            threading.Thread(target=open_browser, daemon=True).start()
    except Exception:
        pass

    # Run FastAPI server accessible both locally and over VPS IP
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")
