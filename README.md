# Disintegration Fingerprinting: A Low-Cost and User-Friendly Tool for Identifying Substandard and Falsified Solid-Dosage Medicines

Ishmam Fatima, Oscar Fajardo, Canhui Liu, Harshith Sadhu, and William H. Grover
*Analytical Chemistry* 98 (12), 8871-8892 (2026)
https://doi.org/10.1021/acs.analchem.5c05418

## Usage

If it's not already installed, install *uv* 

MacOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Program Arduino using the [Arduino IDE](https://www.arduino.cc/en/software/#ide) and the file in df-arduino/df-arduino.ino 

Run acquisition code using

```bash
uv run df-acquire.py
```

Results will be safed in comments.json.gz.

Analyze results using

```base
uv run df-analyze.py path_to_directory
```

where `path_to_directory` is the path to the directory containing the data files to be analyzed.

To recreate all data analysis in the paper, run

```bash
./build.sh
```