# Disintegration Fingerprinting:  A low-cost and easy-to-use tool for identifying substandard and falsified medicines


If it's not already installed, install *uv* 

MacOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Program Arduino using the Arduino IDE and the file in df-arduino/df-arduino.ino 

Run acquisition code using

```bash
uv run df-acquire.py
```

Results will be safed in comments.json.gz.

Analyze results using

```base
uv run df-analyze path_to_directory
```

where path_to_directory is the path to the directory containing the data files to be analyzed.

To recreate all data analysis in the paper, run

```bash
./build.sh
```