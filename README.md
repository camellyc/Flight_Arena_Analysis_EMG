# Flight_Arena_Analysis_EMG

Analysis code for tethered-flight arena experiments in *Drosophila*, combining wingbeat
measurements, DLMn (EMG), and optogenetic stimulation.

A rigidly tethered fly flies in an LED flight arena. An optical wingbeat analyzer reports
left and right wing stroke amplitude and wingbeat frequency; a tungsten sharp electrode in the thorax
records muscle spikes; a 625 nm LED delivers optogenetic stimulation, with the LED driver
signal split back into the acquisition system as ground truth. All channels are digitized
together on a single 9-channel DAQ at 20 kHz and saved as MATLAB `.mat` files.

This repository takes those `.mat` files and produces per-stimulus ("bout") data tables,
quality-control figures, detected EMG spike trains, and per-animal and cross-animal averages
of wingbeat frequency, wingbeat amplitude, spike rate, PSTHs, and interspike-interval (ISI)
statistics.

---

## Repository contents

| File | Role |
|---|---|
| `Step1_WBF_EMG_preprocessing.ipynb` | Raw `.mat` → per-bout CSVs (the main preprocessing stage) |
| `Step2_WBF_EMG_quality_check_plotting.ipynb` | Per-bout quality-control figures |
| `Step3_spike_filtering.ipynb` | Spike-amplitude gating to isolate the target motor unit |
| `Step4_averaging_and_plot.ipynb` | Per-animal and cross-animal averaging, PSTH, ISI analysis |
| `Step4_no_EMG_plotting.ipynb` | Cross-animal wingbeat plotting for experiments without EMG |
| `step_mat_to_summary_no_EMG.ipynb` | Condensed Step 1 + Step 4 in one pass, EMG skipped |
| `analyze_paired_6col.ipynb` | Generic paired statistics (paired *t*, Wilcoxon, Cohen's *dz*) |
| `append_stimulus_sine.ipynb` | Utility to join a stimulus column between two CSVs |
| `tiff_trace_overlay.ipynb` | Composites a high-speed TIFF movie with a data trace overlay |
| `spiracle_helper_functions.py` | Shared library used by all notebooks |

The numbered notebooks are the main pipeline and are meant to be run in order. In `Step1` and
`Step2`, only the **first code cell** is the active analysis path; later cells are marked
`# Legacy` or `# DEFUNCT` and are kept for reference only.

---

## System requirements

### Operating systems

The code is plain Python in Jupyter notebooks and has been run on:

- **macOS** (Apple Silicon)
- **Windows 10 Pro** (22H2)

### Software dependencies

Python **3.11–3.13** (the notebooks carry 3.11, 3.12.5, and 3.13.0 kernels; any version in
that range works).

| Package | Version used |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.14.0 |
| pandas | 3.0.1 |
| matplotlib | 3.9.2 |
| statsmodels | any recent |
| natsort | any recent |
| tqdm | 4.67.3 |
| tifffile | 2026.3.3 |
| imageio + imageio-ffmpeg | any recent |
| pillow | 10.1.0 |
| openpyxl | any recent |
| jupyter / notebook | 7.0.8 |

Version numbers above are the ones the analysis was run with. The code does not pin or check
versions, and no unusual APIs are used, so nearby versions are expected to work.

Notes on the less obvious dependencies:

- **natsort** is used only by `Step3`.
- **statsmodels** is used only by one cell of `Step2`.
- **openpyxl** is needed only if you point `tiff_trace_overlay.ipynb` at an `.xlsx` file
  instead of a `.csv`.
- **imageio-ffmpeg** supplies the FFmpeg binary and is needed only for the optional MP4
  export in `tiff_trace_overlay.ipynb`.

### Non-standard hardware

**None.** This is CPU-only analysis code:

- No GPU or CUDA is required.
- No MATLAB licence is required — the `.mat` files are read directly by `scipy.io.loadmat`.
- No data-acquisition hardware, camera SDK, or instrument driver is required.

---

## Installation guide

With [conda](https://docs.conda.io/) (recommended):

```bash
git clone https://github.com/camellyc/Flight_Arena_Analysis_EMG.git
cd Flight_Arena_Analysis_EMG

conda create -n flight_arena python=3.12
conda activate flight_arena

pip install numpy scipy pandas matplotlib statsmodels natsort tqdm \
            tifffile imageio imageio-ffmpeg pillow openpyxl jupyter
```

Or with `venv` and pip only:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy pandas matplotlib statsmodels natsort tqdm \
            tifffile imageio imageio-ffmpeg pillow openpyxl jupyter
```

Then start Jupyter:

```bash
jupyter notebook
```

**Typical install time on a normal desktop computer: about 3–5 minutes**, most of it spent
downloading packages.

---

## Demo

### Demo data

Example data are deposited at Dryad:

**<https://doi.org/10.5061/dryad.zkh1893s4>**

Download the archive and unpack it to a folder of your choice. It contains raw `.mat` trial
files of the kind the pipeline expects.

### Running the demo

1. Open `Step1_WBF_EMG_preprocessing.ipynb`.
2. In the first code cell, set `input_directory` to the folder containing the demo `.mat`
   files, and `output_directory` to a folder for results (conventionally an `output_csv`
   subfolder).
3. Change the `%run` line at the top of the cell to point at your local copy of
   `spiracle_helper_functions.py`, or replace it with
   `from spiracle_helper_functions import *` if you launched Jupyter from the repository
   folder.
4. Run the cell.
5. Open `Step2_WBF_EMG_quality_check_plotting.ipynb`, point `folder_path` at the
   `output_csv` folder, and run the first cell to generate quality-control figures.
6. Optionally continue with `Step3` and `Step4` as described below.

### Expected output

**Step 1** writes one CSV per detected stimulus presentation:

```
output_csv/<trial_name>_bout<N>_stim<duration>ms.csv
```

Each file covers a window from 5 s before to 15 s after stimulus onset, with time re-zeroed
to onset, and has these columns:

```
Time (s), Wingbeat Frequency, Wing Beat Amplitude, X Position, Raw EMG,
Spike Rate, Spike Count, Spike Amplitude, Opto Stimulus, hutchens, Stim Duration (s)
```

While running, Step 1 prints the loaded data shape, the number of stimulus blocks detected
and their measured lengths in seconds, the planned stimulus orders read from the file, and
the matched stimulus onset times. Trials that include unstimulated control presentations
produce `stim0ms` bouts alongside the stimulated ones.

**Step 2** writes one three-panel SVG per bout next to each CSV: change in wingbeat frequency
from baseline with wingbeat amplitude and spike rate overlaid, the raw EMG trace with
detected spikes marked, and a spike raster, with the stimulus window shaded.

**Step 4** writes, per animal, a `summary_<duration>ms.csv` containing trial-averaged traces
(mean and SEM of change in wingbeat frequency, wingbeat amplitude, normalized wingbeat
amplitude, and spike rate, plus one raster column per trial) and a matching
`visualization_<duration>ms.png`. Across animals it writes
`summary_across_animals_<duration>ms.svg`/`.csv`, a PSTH-only variant, and a set of ISI
figures and tables.

### Expected run time

On a normal desktop computer:

- **Step 1: a few minutes per trial** (roughly 3–4 minutes for a several-minute recording).
  This is the slow stage; nearly all of the time goes into the adaptive spike-detection
  threshold, which loops over every sample in Python. Very long recordings scale accordingly
  — a 30-minute trial takes upwards of 20 minutes.
- **Steps 2–4: seconds to about a minute per animal**, since they work on the much smaller
  CSV files.

---

## Instructions for use

### Running on your own data

Run the numbered notebooks in order. Every notebook has its input and output paths hardcoded
at the top of the relevant cell — **edit these to point at your own data before running.**
In addition, `Step1` through `Step4` load the shared library with an absolute `%run` path;
change it to your local path, or replace it with a plain
`from spiracle_helper_functions import *` when Jupyter is started from the repository folder.

**Step 1 — preprocessing.** Point `input_directory` at a folder of `.mat` trials and
`output_directory` at an `output_csv` folder. Step 1 extracts the traces, converts wingbeat
frequency to Hz, reconstructs the stimulus epochs, detects EMG spikes, and cuts one CSV per
stimulus presentation.

**Step 2 — quality control.** Point `folder_path` at the `output_csv` folder and run. Inspect
the resulting figures and discard bouts in which the fly stopped flying or the EMG was lost.

**Step 3 — spike filtering.** Run the first block to see spike-amplitude histograms for every
bout. **Read the amplitude cutoffs off these histograms and type them into the second
block.** These values depend on the electrode and the preparation and must be chosen for each
dataset — there is no universal setting. The second block zeroes spikes outside the chosen
amplitude window, recomputes spike rate, and re-emits the CSVs into
`output_csv_spikefiltered/<duration>ms/`. The third block re-plots the filtered data so you
can confirm the result. An optional final cell tabulates per-bout summary statistics into
`bout_quality_summary.csv` with a blank `keep` column you can fill in by hand as a record of
which bouts you accepted.

**Step 4 — averaging and figures.** Point `base_dir` at an animal's `output_csv_spikefiltered`
folder to build that animal's summary CSV and figure. The cross-animal blocks take a parent
directory and pool across animals. Use `Step4_no_EMG_plotting.ipynb` instead when the
experiment has no EMG channel.

### Manual steps

The pipeline is deliberately not fully automatic. Three points require human judgement:

1. **Spike-amplitude cutoffs** are chosen by eye from the Step 3 histograms, per dataset.
2. **Bad bouts are removed by hand** — inspect the Step 2 and Step 3 figures and delete the
   rejected bout CSVs before running Step 4.
3. **Vetted animal folders are marked with a `_Y` suffix.** The cross-animal blocks in Step 4
   only pick up folders whose names contain both `EMG` and `Y`, so an animal is included in
   group averages only once you have renamed its folder.

### Standalone notebooks

`analyze_paired_6col.ipynb`, `append_stimulus_sine.ipynb`, and `tiff_trace_overlay.ipynb` are
independent utilities and are not part of the Step 1–4 chain. `analyze_paired_6col.ipynb`
runs paired statistics on any CSV with one identifier column and six numeric columns
(three paired comparisons).

---

## Pipeline overview

```
<trial>.mat  (9 channels x N samples @ 20 kHz)
   |
   |  Step 1   extract traces, detect spikes, cut bouts (-5 s to +15 s)
   v
output_csv/<trial>_bout<N>_stim<duration>ms.csv
   |                        |
   |  Step 2                |  Step 3   amplitude gating (cutoffs chosen by eye)
   v                        v
per-bout QC figures    output_csv_spikefiltered/<duration>ms/*.csv
                            |
                            |  Step 4   align, average, resample
                            v
                       summary_<duration>ms.csv   +   visualization_<duration>ms.png
                            |
        +-------------------+--------------------+
        v                   v                    v
 summary_across_      PSTH-only figures     ISI figures and tables
 animals_*.svg/csv    and tables            (histograms, ECDFs, statistics)
```

---

## Input data format

Each trial is a MATLAB `.mat` file containing:

- a **9 × N** (or N × 9) double array of raw channel data sampled at **20 kHz**, and
- `allRandomizedStimOrders`, the planned stimulus durations for each block.

Save these as MAT-file **v7 or older**. Version 7.3 files are HDF5-based and cannot be read
by this code.

### Channel map

| Row | Signal | Used as |
|---|---|---|
| `Data[0]` | Acquisition time | Not used; the time axis is regenerated |
| `Data[1]`, `Data[2]` | Wing stroke amplitude, left and right | Averaged into `Wing Beat Amplitude` |
| `Data[3]` | Wingbeat frequency | Smoothed and scaled to Hz |
| `Data[4]` | Arena X position | `X Position` |
| `Data[5]` | Raw EMG | Spike detection |
| `Data[6]` | 625 nm LED driver signal | Binarized into `Opto Stimulus` |
| `Data[7]` | Right wingbeat analyzer channel | Carried through, not analyzed |
| `Data[8]` | Camera trigger | Not used in this repository |

The channel order is assumed, not verified at runtime. If the acquisition wiring changes, this
mapping must be updated in `extract_traces()` in `spiracle_helper_functions.py`.

### Processing parameters

| Quantity | Value |
|---|---|
| Sampling rate | 20 kHz |
| Wingbeat smoothing | 100-sample (5 ms) moving average |
| Stimulus reconstruction | LED pulses separated by < 30 ms merged into one epoch |
| Spike threshold | Adaptive: 3 × local SD in a ±0.5 s window |
| Spike detection | Derivative peaks, 10 ms refractory period, realigned ±2.5 ms to the absolute peak |
| Spike amplitude | Peak-to-trough within 5 ms |
| Spike rate | Gaussian smoothing of the spike train, σ = 50 ms |
| Bout window | −5 s to +15 s relative to stimulus onset |
| Baseline window | −2 s to 0 s |
| PSTH | 60 ms bins, Gaussian σ = 60 ms |
| ISI analysis | −2 s to 10 s, 200 ms sliding window, 50 ms step, 1 ms histogram bins |

Stimulus epochs are matched against the planned stimulus orders stored in the file, and
unstimulated control epochs are inserted at their planned times so that control bouts are
produced even though no LED fired.

---

## Limitations and known issues

- **Hardcoded paths.** Every notebook has absolute input/output paths, and Steps 1–4 load the
  shared library with an absolute `%run` path. All of these must be edited for your machine.
  `step_mat_to_summary_no_EMG.ipynb` shows the more portable pattern
  (`from spiracle_helper_functions import ...`).
- **The sampling rate is hardcoded to 20 kHz** and is not read from the file, even though the
  acquisition parameters are stored in it. Recordings at another rate need the constant
  changed.
- **The `.mat` loader finds the data array by shape**, taking the first two-dimensional array
  with a dimension of 9. A file containing another 9 × N array could be misread.
- `detect_spikes` is defined twice in `spiracle_helper_functions.py`; the second definition is
  the one that takes effect.
- One legacy cell in `Step2` calls helper functions that no longer exist and will not run.
- Figures are written with editable text (`svg.fonttype = 'none'`) at 300 dpi so they can be
  edited in vector graphics software. This requires Arial to be installed for the text to
  render as intended.

---

## Contact

- Yichen Luo — <yichenl@uw.edu>
- John Tuthill — <tuthill@uw.edu>
