# Energy Dispatch Optimization

A Python project for simulating baseline electricity dispatch and optimizing
generator schedules against hourly demand and renewable generation data.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib scipy ortools
```

## Run

From the project root:

```bash
python main.py
```

The script reads hourly demand, wind, and solar profiles from `data/`, runs
baseline and optimized dispatch, and writes the optimized schedule to
`results/optimized_dispatch.csv`.