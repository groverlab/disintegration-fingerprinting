# /// script
# dependencies = ["pyserial"]
# ///

### df-acquire.py by Grover Lab, University of California, Riverside

import sys
import datetime
import time
import platform
import json
import gzip
import argparse

try:
    import serial
    from serial.tools.list_ports import comports
except ImportError as err:
    sys.exit(
        "❌ Can't find pyserial module - install it by running 'pip install pyserial'"
    )

parser = argparse.ArgumentParser()
parser.add_argument("comments", nargs="?", help="Sample description")
parser.add_argument(
    "-d", "--duration", type=int, help="Run duration [min] or -1 for forever"
)
args = parser.parse_args()


r = {}
if args.comments:
    comments = args.comments.replace(" ", "_")
    print("sample =", comments)
else:
    comments = input("Sample?  ").replace(" ", "_")
r["filename"] = (
    datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + comments + ".json.gz"
)

ports = 0
port = ""
arduino_ports = []
for p in comports():
    print(str(p))
    if "USB" in str(p) or "usb" in str(p) or "Arduino" in str(p):
        port = p.name
        ports += 1
        arduino_ports.append(port)
if ports == 0:
    sys.exit("❌ No port found - is the Arduino plugged in?")
if ports >= 2:
    print(f"Found Arduinos on these ports:  {arduino_ports}")
    port = input("Which port?  ")
if platform.system() == "Darwin":  # if MacOS...
    port = "/dev/" + port  # ...prepend /dev/
print("✅ Found an Arduino at " + port)

r["port"] = port

r["data"] = []
ser = serial.Serial(port, 115200, timeout=5)

start_time = time.time()
r["start_time"] = str(start_time)

t1 = 300  # duration of baseline phase in seconds; currently 5 minutes or 300 seconds
if args.duration:
    t2 = args.duration * 60 + t1  # set custom dissolution time from command option
else:
    t2 = (
        3600 + t1
    )  # duration of dissolution phase in seconds; currently 60 minutes or 3600 seconds

pill_in = False
errors = 0
ser.flush()
ser.read_all()
while True:
    try:
        s = ser.readline().decode("utf-8", "ignore")
        waiting = ser.inWaiting()
        et = time.time() - start_time
        if 0 <= et < t1:  # baseline phase
            print(
                f"TimeToPill: {str(datetime.timedelta(seconds=t1 - et))[:9]}", end="\t"
            )
        elif t1 <= et < t2:  # dissolution phase
            if not pill_in:
                ser.write(b"2")
                pill_in = True
            print(
                f"TimeToEnd: {str(datetime.timedelta(seconds=t2 - et))[:9]}", end="\t"
            )
        elif et >= t2:  # experiment over
            ser.write(b"1")
            break
        print(f"Port: {port}\tBacklog: {waiting}\tErrors: {errors}\tData: {s}", end="")
        try:
            r["data"].append(int(s))
        except ValueError:
            print("Value error")
            errors += 1
    except KeyboardInterrupt:
        print("quitting now")
        ser.write(b"1")
        break
r["stop_time"] = str(time.time())
f = gzip.open(r["filename"], "wt")
json.dump(r, f)
f.close()
print("DONE")
