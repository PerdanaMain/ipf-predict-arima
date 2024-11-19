from datetime import datetime
import pytz

def print_log(message):
    time = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

    # cetak ke logs/log.txt
    with open("logs/log.txt", "a") as f:
        f.write(f"[{time}] {message}\n")
        f.close()