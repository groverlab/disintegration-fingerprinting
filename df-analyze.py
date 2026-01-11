# /// script
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///

### df-analyze.py by Grover Lab, University of California, Riverside

import os
import gzip
import json
import numpy
import scipy
import sys
import argparse
import matplotlib.pyplot as plt
import time
from matplotlib import use
import matplotlib.ticker as mticker
from matplotlib.colors import TABLEAU_COLORS

use("MacOSX")
plt.rcParams["font.family"] = "Arial"

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory containing DF data files")
parser.add_argument("--figure", default="none", help="figure to generate", choices=["aspirin", "beads", "variety", "long", "condensed", "bayer_tylenol", "bayer_tylenol_only", "bayer_tylenol_only_subdirs"],)
args = parser.parse_args()


def list_dir(directory):
    runfiles = []
    if os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(".gz"):
                    runfiles.append(os.path.join(root, f))
    else:
        if directory.endswith(".gz"):
            runfiles.append(directory)
    return runfiles


def window_sum(x, w):
    c = numpy.cumsum(x)
    s = c[w - 1 :]
    s[1:] -= c[:-w]
    return s


def window_lin_reg(x, y, w):
    sx = window_sum(x, w)
    sy = window_sum(y, w)
    sx2 = window_sum(x**2, w)
    sxy = window_sum(x * y, w)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept


discards = open(f"{args.figure} discards.txt", "w")
log = open(f"{args.figure} log.txt", "w")


def plog(text):
    print(text)
    log.write(f"{text}\n")


summary_peak_counts = []
summary_samples = []
summary_filenames = []
summary_peak_times = []
summary_baselines = []

oodr_count = 0
noise_peak_count = 0
measurement_count = 0

bin_width = 75  # seconds

for filename in sorted(list_dir(args.directory)):
    plog("\n  " + filename)
    sample = os.path.normpath(filename).split(os.path.sep)[-2]
    plog(f"  💊 {sample}")
    discard = False

    # plots of individual samples, saved alongside data files:
    sample_fig, sample_axs = plt.subplots(3, figsize=(4.3, 5))

    # load data from gzip file
    gf = gzip.open(filename, "rb")
    r = json.load(gf)
    start = float(r["start_time"])
    plog(f"  Start time: {start}")
    stop = float(r["stop_time"])
    plog(f"  Stop time: {stop}")
    duration = stop - start
    plog(f"  Duration: {duration} s")
    points = len(r["data"])
    plog(f"  Measurements: {points}")
    measurement_count += points
    data = numpy.array(r["data"])
    measurements_per_second = points / duration
    plog(f"  Measurements per second: {measurements_per_second}")

    # reconstruct times from byte counts:
    byte_count = 0
    elapsed_bytes = []
    for d in data:
        elapsed_bytes.append(byte_count)
        byte_count += len(str(d)) + 2  # include carriage return and newline
    plog(f"  Byte count: {byte_count}")
    seconds_per_byte = duration / byte_count
    plog(f"  Seconds per byte: {seconds_per_byte}")
    times = numpy.array(elapsed_bytes) * seconds_per_byte

    # remove out-of-dynamic-range measurements:
    for i in range(len(data)):
        if data[i] < 0 or data[i] > 1023:
            data[i] = data[i - 1]  # replace OODM data with previous measurement
            plog(f"  ⚠️  Out-of-dynamic-range measurement removed at index {i}")
            oodr_count += 1

    # invert data so that peaks point upward:
    data = 1023 - data

    # Calculate baseline in 200-290 seconds for later use
    pre_pill_baseline_start_index = numpy.searchsorted(times, 200)
    pre_pill_baseline_stop_index = numpy.searchsorted(times, 290)
    pre_pill_baseline_mean = numpy.mean(
        data[pre_pill_baseline_start_index:pre_pill_baseline_stop_index]
    )
    plog(f"  Baseline mean: {pre_pill_baseline_mean}")

    # remove all data before 300 seconds
    pill_added_index = numpy.searchsorted(times, 300)
    times = times[pill_added_index:] - times[pill_added_index]
    data = data[pill_added_index:]

    # Noise peak detection (using threshold algorithm) and removal:
    noise_peaks, noise_properties = scipy.signal.find_peaks(data, threshold=20)

    # Noise peak removal
    for p in noise_peaks:
        data[p] = (data[p - 1] + data[p + 1]) / 2  # testing noise peak removal
        plog(f"  ⚠️  Noise peak removed at index {p}")
        noise_peak_count += 1

    # plot the raw data after noise removal
    sample_axs[1].plot(
        numpy.array(times) / 60, 5 * numpy.array(data) / 1023, "-k", linewidth=0.2
    )
    # sample_axs[1].set_xlim(-3, 63)
    sample_axs[1].set_ylim(-0.1, 1.7)
    sample_axs[1].set_xlabel("Time (minutes)")
    sample_axs[1].set_ylabel("Sensor output (volts)")

    # plot a closeup of some of the raw data
    sample_axs[0].plot(
        numpy.array(times), 5 * numpy.array(data) / 1023, "-k", linewidth=1
    )
    sample_axs[0].set_xlim(152.95, 154.05)
    sample_axs[0].set_ylim(0.05, 0.7)
    sample_axs[0].set_xlabel("Time (seconds)")
    sample_axs[0].set_ylabel("Sensor output (volts)")






##### DISCARD TESTING:

# Original aspirins:
# aspirin/bayer/20241121T124250_Bayer_Aspirin.json.gz     -0.4234769589340449
# aspirin/generic/20241114T201703_generic_aspirin.json.gz -0.5492949490815137
# aspirin/generic/20241115T080902_generic_aspirin.json.gz -1.180565397378209
# aspirin/generic/20241119T123836_Generic_Aspirin.json.gz -0.3501627837618819
# aspirin/generic/20241121T150322_generic_aspirin.json.gz -0.22782928989961843

# New Tylenol and Bayer:
# bayer_tylenol/tylenol-cold/20251210T135456_T06COLDONE.json.gz   -0.2052484326550232
# bayer_tylenol/tylenol-cold/20251210T160747_T06COLDTHREE.json.gz -0.20948676231470195
# bayer_tylenol/tylenol-rt/20251113T154922_T01one.json.gz         -0.21095045032875986
# bayer_tylenol/tylenol-rt/20251117T112834_T04one.json.gz         -0.24732790108080935
# bayer_tylenol/tylenol-rt/20251119T164358_T13one.json.gz         -0.6429757147389821
# bayer_tylenol/tylenol-rt/20251124T135557_T17.json.gz            -0.22988615287523775

# min_slope discard threshold was originally < -0.2
# Changing this to -0.25 would admit all new tylenols except one.
# It will also admit one old generic aspirin.



    ########################################################
    # DISABLE BUBBLE DISCARDS FOR LONG DRUG RUNS:
    ########################################################
    if args.figure != "long":
        # Calculate moving linear regression to find bubbles:
        w = int(measurements_per_second * 50.0)  # 50 seconds of measurements
        # w = 100_000  # was 100_000.  At 2000 meas/sec, this is about 50 seconds
        slopes, yints = window_lin_reg(times, data, w)
        min_slope = numpy.min(slopes)
        plog(f"  Minimum slope: {min_slope}")
        log.write(f"{filename},{min_slope},\n")
        if min_slope < -0.25:
            discards.write(f"{filename}\t{min_slope}\n")
            discard = True










    # calculate peak counts and average prominence
    peak_times = []
    peak_counts = []
    baselines = []
    for i in range(0, int(times[-1]), bin_width):
        peak_times.append(i)
        start_index = numpy.searchsorted(times, i)
        stop_index = numpy.searchsorted(times, i + bin_width)
        peak_indices, peak_properties = scipy.signal.find_peaks(
            data[start_index:stop_index], prominence=10
        )
        peak_counts.append(len(peak_indices))
        baselines.append(
            numpy.median(data[start_index:stop_index]) - pre_pill_baseline_mean
        )

    # Sample plot of peak locations
    sample_axs[0].plot(
        times[peak_indices],
        5 * numpy.array(data[peak_indices]) / 1023,
        "go",
        markersize=2,
    )

    # Sample plot of peak counts
    sample_axs[2].plot(
        numpy.array(peak_times) / 60.0,
        60.0 * numpy.array(peak_counts) / bin_width + 1,
        "k",
        linewidth=1,
    )
    sample_axs[2].set_yscale("log")
    sample_axs[2].set_ylim(0.5, 2000)
    sample_axs[2].set_yticks([1, 10, 100, 1000])
    sample_axs[2].minorticks_off()
    sample_axs[2].yaxis.set_major_formatter(mticker.ScalarFormatter())
    a = sample_axs[2].get_yticks().tolist()
    a[0] = "0"
    sample_axs[2].set_yticklabels(a)
    sample_axs[2].set_xlabel("Time (minutes)")
    sample_axs[2].set_ylabel("Particles/minute")
    sample_axs[2].yaxis.set_label_coords(-0.095, 0.40)

    # save results for summary plots
    if not discard:
        summary_peak_times.append(numpy.array(peak_times))
        summary_peak_counts.append(numpy.array(peak_counts))
        summary_baselines.append(numpy.array(baselines))
        summary_samples.append(sample)
        summary_filenames.append(filename)

    # save summary plot:
    sample_fig.subplots_adjust(
        left=0.13, right=0.97, bottom=0.10, top=0.98, hspace=0.45
    )
    sample_fig.savefig(filename + ".pdf")
    sample_fig.clf()


########### Aspirin log plot
if args.figure == "aspirin":
    summary_fig, summary_axs = plt.subplots(figsize=(4, 3))
    for sample, peak_times, peak_counts in zip(
        summary_samples, summary_peak_times, summary_peak_counts
    ):
        if "ayer" in sample:
            summary_axs.plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:orange",
                linewidth=1,
            )
        else:
            summary_axs.plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:blue",
                linewidth=1,
            )
    summary_axs.set_yscale("log")
    summary_axs.set_ylim(0.7, 1500)
    summary_axs.set_yticks([1, 10, 100, 1000])
    summary_axs.minorticks_off()
    summary_axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    a = summary_axs.get_yticks().tolist()
    a[0] = "0"
    summary_axs.set_yticklabels(a)
    summary_axs.set_xlabel("Time (minutes)")
    summary_axs.set_ylabel("Particles per minute")
    summary_axs.legend(("Aspirin (name brand)", "Aspirin (generic)"), frameon=False)
    leg = summary_axs.get_legend()
    if len(leg.legend_handles) > 1:  # only do this if there are at least 2 sample types
        leg.legend_handles[0].set_color("tab:orange")
        leg.legend_handles[1].set_color("tab:blue")
    summary_axs.yaxis.set_label_coords(-0.1, 0.50)
    summary_fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.99)
    summary_fig.savefig("aspirin.pdf")
    summary_fig.clf()



########### Bayer-Tylenol_only log plot
if args.figure == "bayer_tylenol_only_subdirs":
    summary_fig, summary_axs = plt.subplots(1, 2, figsize=(8, 3))
    for sample, peak_times, peak_counts in zip(
        summary_samples, summary_peak_times, summary_peak_counts
    ):
        if "ACOLD" in sample:
            summary_axs[0].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:blue",
                linewidth=1, zorder=10
            )
        elif "AHOT" in sample:
            summary_axs[0].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:pink",
                linewidth=1, zorder=10
            )
        elif "ART" in sample:
            summary_axs[0].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="lightgray",
                linewidth=1,
            )
        elif "ACANADA" in sample:
            summary_axs[0].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:red",
                linewidth=1, zorder=10
            )
        elif "TCOLD" in sample:
            summary_axs[1].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:blue",
                linewidth=1, zorder=10
            )
        elif "THOT" in sample:
            summary_axs[1].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:pink",
                linewidth=1, zorder=10
            )
        elif "TRT" in sample:
            summary_axs[1].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="lightgray",
                linewidth=1,
            )
        elif "TCANADA" in sample:
            summary_axs[1].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                color="tab:red",
                linewidth=1, zorder=10
            )
        else:
            sys.exit(f"ERROR {sample} run can't be categorized")
    summary_axs[0].set_yscale("log")
    summary_axs[1].set_yscale("log")
    summary_axs[0].set_ylim(0.7, 1500)
    summary_axs[1].set_ylim(0.7, 1500)
    summary_axs[0].set_yticks([1, 10, 100, 1000])
    summary_axs[1].set_yticks([1, 10, 100, 1000])
    summary_axs[0].minorticks_off()
    summary_axs[1].minorticks_off()
    summary_axs[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
    summary_axs[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
    a = summary_axs[0].get_yticks().tolist()
    b = summary_axs[0].get_yticks().tolist()
    a[0] = "0"
    b[0] = "0"
    summary_axs[0].set_yticklabels(a)
    summary_axs[1].set_yticklabels(b)
    summary_axs[0].set_xlabel("Time (minutes)")
    summary_axs[1].set_xlabel("Time (minutes)")
    summary_axs[0].set_ylabel("Particles per minute")
    summary_axs[1].set_ylabel("Particles per minute")
    summary_axs[0].legend(("bayer-cold", "bayer-hot", "bayer-rt", "bayer-ca"), frameon=False)
    summary_axs[1].legend(("tylenol-cold", "tylenol-hot", "tylenol-rt", "tylenol-ca"), frameon=False)
    lega = summary_axs[0].get_legend()
    legb = summary_axs[1].get_legend()
    if len(lega.legend_handles) > 1:  # only do this if there are at least 2 sample types
        lega.legend_handles[0].set_color("tab:blue")
        lega.legend_handles[1].set_color("tab:pink")
        lega.legend_handles[2].set_color("lightgray")
        lega.legend_handles[3].set_color("tab:red")
        legb.legend_handles[0].set_color("tab:blue")
        legb.legend_handles[1].set_color("tab:pink")
        legb.legend_handles[2].set_color("lightgray")
        legb.legend_handles[3].set_color("tab:red")
    summary_axs[0].yaxis.set_label_coords(-0.1, 0.50)
    summary_axs[1].yaxis.set_label_coords(-0.1, 0.50)
    summary_fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.99)
    summary_fig.savefig("bayer_tylenol_only_subdirs.pdf")
    summary_fig.clf()



###### bayer-tylenol big plot
if args.figure == "bayer_tylenol_only_subdirs":
    plot_numbers = {
        '101': 0,
        '102': 1,
        '103': 2,
        '104': 3,
        '105': 4,
        '106': 5,
        '107': 6,
        '108': 7,
        '109': 8,
        '110': 9,
        '111': 10,
        '112': 11,
        '113': 12,
        '114': 13,
        '115': 14,
        '116': 15,
        '117': 16,
        '150': 18,
        '151': 19,
        '152': 20,
        '153': 21,
        '154': 22,
        '155': 23,
        '501': 24,
        '502': 25,
        '503': 26,
        '504': 27,
        '505': 28,
        '506': 29,
        '507': 30,
        '508': 31,
        '509': 32,
        '510': 33,
        '511': 34,
        '512': 35,
        '513': 36,
        '514': 37,
        '515': 38,
        '516': 39,
        '517': 40,
        '518': 41,
        '519': 42,
        '550': 48,
        '551': 49,
        '552': 50,
        '553': 51,
        '554': 52,
        '555': 53,
    }
    bayer_tylenol_data = {
    '101': 'Tylenol 2027/03 BC',
    '102': 'Tylenol 2028/04 CA',
    '103': 'Tylenol 2028/04 WA',
    '104': 'Tylenol 2028/04 MD',
    '105': 'Tylenol 2028/04 TN',
    '106': 'Tylenol 2028/04 CO',
    '107': 'Tylenol 2028/05 DC',
    '108': 'Tylenol 2028/07 BC',
    '109': 'Tylenol 2028/10 VA',
    '110': 'Tylenol 2028/12 CO',
    '111': 'Tylenol 2029/01 TN',
    '112': 'Tylenol 2029/01 CO',
    '113': 'Tylenol 2029/01 CA',
    '114': 'Tylenol 2029/02 CA',
    '115': 'Tylenol 2029/03 VA',
    '116': 'Tylenol 2029/03 CA',
    '117': 'Tylenol 2029/07 CO',
    '150': 'Tylenol HOT 1',
    '151': 'Tylenol HOT 2',
    '152': 'Tylenol HOT 3',
    '153': 'Tylenol COLD 1',
    '154': 'Tylenol COLD 2',
    '155': 'Tylenol COLD 3',
    '501': 'Bayer 2026/10 CO',
    '502': 'Bayer 2027/01 CO',
    '503': 'Bayer 2027/06 BC',
    '504': 'Bayer 2027/09 TN',
    '505': 'Bayer 2027/09 CA',
    '506': 'Bayer 2027/10 MD',
    '507': 'Bayer 2027/11 CO',
    '508': 'Bayer 2027/12 BC',
    '509': 'Bayer 2028/01 CA',
    '510': 'Bayer 2028/01 CO',
    '511': 'Bayer 2028/01 TN',
    '512': 'Bayer 2028/01 BC',
    '513': 'Bayer 2028/02 CA',
    '514': 'Bayer 2028/02 DC',
    '515': 'Bayer 2028/02 MN',
    '516': 'Bayer 2028/04 CA',
    '517': 'Bayer 2028/04 TN',
    '518': 'Bayer 2028/05 VA',
    '519': 'Bayer 2028/06 CA',
    '550': 'Bayer HOT 1',
    '551': 'Bayer HOT 2',
    '552': 'Bayer HOT 3',
    '553': 'Bayer COLD 1',
    '554': 'Bayer COLD 2',
    '555': 'Bayer COLD 3',
    }
    collage_fig, collage_axs = plt.subplots(
        9, 6, sharex=True, sharey=True, figsize=(6.5, 8)
    )
    for filename, sample, peak_times, peak_counts in zip(summary_filenames, summary_samples, summary_peak_times, summary_peak_counts):
        plot_number = plot_numbers[filename.split(" ")[0][-3:]]
        label = bayer_tylenol_data[filename.split(" ")[0][-3:]]
        collage_axs.flat[plot_number].text(
            0.02,
            0.90,
            # sample,
            # filename,
            label,
            size=8,
            horizontalalignment="left",
            verticalalignment="center",
            transform=collage_axs.flat[plot_number].transAxes,
        )
        collage_axs.flat[plot_number].plot(
            numpy.array(peak_times) / 60.0,
            60.0 * numpy.array(peak_counts) / bin_width + 1,
        )  # log
        collage_axs.flat[plot_number].set_yscale("log")
        collage_axs.flat[plot_number].set_ylim(0.5, 5000)
        collage_axs.flat[plot_number].set_yticks([1, 10, 100, 1000])
        collage_axs.flat[plot_number].yaxis.set_major_formatter(
            mticker.ScalarFormatter()
        )
        a = collage_axs.flat[plot_number].get_yticks().tolist()
        a[0] = "0"
        collage_axs.flat[plot_number].set_yticklabels(a)
        collage_axs.flat[plot_number].set_xticks([0, 30, 60])
    collage_fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.06, top=0.99, hspace=0.10, wspace=0.08
    )
    collage_fig.text(0.5, 0.01, "Time (minutes)", ha="center")
    collage_fig.text(
        0.01, 0.5, "Particles per minute", va="center", rotation="vertical"
    )
    collage_fig.savefig("BIGPLOT.pdf")
    collage_fig.clf()

















############## Bead plot
if args.figure == "beads":
    summary_fig, summary_axs = plt.subplots(figsize=(4, 3))
    for sample, peak_times, peak_counts in zip(
        summary_samples, summary_peak_times, summary_peak_counts
    ):
        summary_axs.plot(
            numpy.array(peak_times) / 60.0,
            60.0 * numpy.array(peak_counts) / bin_width + 1,
            color="k",
            linewidth=1,
        )
    summary_axs.set_yscale("log")
    summary_axs.set_ylim(0.7, 1500)
    summary_axs.set_yticks([1, 10, 100, 1000])
    summary_axs.minorticks_off()
    summary_axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    a = summary_axs.get_yticks().tolist()
    a[0] = "0"
    summary_axs.set_yticklabels(a)
    summary_axs.set_xlabel("Time (minutes)")
    summary_axs.set_ylabel("Particles per minute")
    summary_axs.legend(("Polyethylene beads",), frameon=False)  # for beads
    leg = summary_axs.get_legend()
    if len(leg.legend_handles) > 1:  # only do this if there are at least 2 sample types
        leg.legend_handles[0].set_color("tab:orange")
        leg.legend_handles[1].set_color("tab:blue")
    summary_axs.yaxis.set_label_coords(-0.1, 0.50)
    summary_fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.99)
    summary_fig.savefig("beads.pdf")
    summary_fig.clf()


discards.close()

###### drug variety collage
if args.figure == "variety":
    sample_plot_numbers = {}
    most_recent_plot_number = 0
    collage_fig, collage_axs = plt.subplots(
        8, 4, sharex=True, sharey=True, figsize=(6.5, 8)
    )
    for filename, sample, peak_times, peak_counts in zip(
        summary_filenames, summary_samples, summary_peak_times, summary_peak_counts
    ):
        if (
            sample not in sample_plot_numbers
        ):  # if this is the first time we've plotted this sample type, start a new plot
            sample_plot_numbers[sample] = most_recent_plot_number
            most_recent_plot_number += 1
            collage_axs.flat[sample_plot_numbers[sample]].text(
                0.02,
                0.90,
                sample,
                size=8,
                horizontalalignment="left",
                verticalalignment="center",
                transform=collage_axs.flat[sample_plot_numbers[sample]].transAxes,
            )
        collage_axs.flat[sample_plot_numbers[sample]].plot(
            numpy.array(peak_times) / 60.0,
            60.0 * numpy.array(peak_counts) / bin_width + 1,
        )  # log
        collage_axs.flat[sample_plot_numbers[sample]].set_yscale("log")
        collage_axs.flat[sample_plot_numbers[sample]].set_ylim(0.5, 5000)
        collage_axs.flat[sample_plot_numbers[sample]].set_yticks([1, 10, 100, 1000])
        collage_axs.flat[sample_plot_numbers[sample]].yaxis.set_major_formatter(
            mticker.ScalarFormatter()
        )
        a = collage_axs.flat[sample_plot_numbers[sample]].get_yticks().tolist()
        a[0] = "0"
        collage_axs.flat[sample_plot_numbers[sample]].set_yticklabels(a)
        collage_axs.flat[sample_plot_numbers[sample]].set_xticks([0, 30, 60])
    collage_fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.06, top=0.99, hspace=0.10, wspace=0.08
    )
    collage_fig.text(0.5, 0.01, "Time (minutes)", ha="center")
    collage_fig.text(
        0.01, 0.5, "Particles per minute", va="center", rotation="vertical"
    )
    collage_fig.savefig("variety.pdf")
    collage_fig.clf()


######## LONG drug variety collage
if args.figure == "long":
    # find the length of the shortest run
    smallest_peak_times_length = len(summary_peak_times[0])
    for peak_times in summary_peak_times:
        if len(peak_times) < smallest_peak_times_length:
            smallest_peak_times_length = len(peak_times)
    # truncate all the data to the length of the shortest run:
    for i in range(len(summary_peak_times)):
        summary_peak_times[i] = summary_peak_times[i][:smallest_peak_times_length]
        summary_peak_counts[i] = summary_peak_counts[i][:smallest_peak_times_length]
        summary_baselines[i] = summary_baselines[i][:smallest_peak_times_length]
    # now make the plot
    sample_plot_numbers = {}
    most_recent_plot_number = 0
    collage_fig, collage_axs = plt.subplots(4, 1, sharex=True, figsize=(6.5, 4))
    for filename, sample, peak_times, peak_counts in zip(
        summary_filenames, summary_samples, summary_peak_times, summary_peak_counts
    ):
        if (
            sample not in sample_plot_numbers
        ):  # if this is the first time we've plotted this sample type, start a new plot
            sample_plot_numbers[sample] = most_recent_plot_number
            most_recent_plot_number += 1
            collage_axs.flat[sample_plot_numbers[sample]].text(
                0.02,
                0.90,
                sample,
                size=8,
                horizontalalignment="left",
                verticalalignment="center",
                transform=collage_axs.flat[sample_plot_numbers[sample]].transAxes,
            )
        collage_axs.flat[sample_plot_numbers[sample]].plot(
            numpy.array(peak_times) / 60.0 / 60.0,
            60.0 * numpy.array(peak_counts) / bin_width + 1,
        )  # log
        collage_axs.flat[sample_plot_numbers[sample]].set_yscale("log")
        collage_axs.flat[sample_plot_numbers[sample]].set_ylim(0.5, 5000)
        collage_axs.flat[sample_plot_numbers[sample]].set_yticks([1, 10, 100, 1000])
        collage_axs.flat[sample_plot_numbers[sample]].yaxis.set_major_formatter(
            mticker.ScalarFormatter()
        )
        a = collage_axs.flat[sample_plot_numbers[sample]].get_yticks().tolist()
        a[0] = "0"
        collage_axs.flat[sample_plot_numbers[sample]].set_yticklabels(a)
        # collage_axs.flat[sample_plot_numbers[sample]].set_xticks([0, 30, 60])
    collage_fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.10, top=0.99, hspace=0.10, wspace=0.08
    )
    collage_fig.text(0.55, 0.01, "Time (hours)", ha="center")
    collage_fig.text(
        0.01, 0.5, "Particles per minute", va="center", rotation="vertical"
    )
    collage_fig.savefig("long.pdf")
    collage_fig.clf()


########## condensed drug variety collage
if args.figure == "condensed":
    sample_plot_numbers = {}
    most_recent_plot_number = 0
    collage_fig, collage_axs = plt.subplots(
        5, 6, sharex=True, sharey=True, figsize=(6.5, 2)
    )  # for ACS grant
    for filename, sample, peak_times, peak_counts in zip(
        summary_filenames, summary_samples, summary_peak_times, summary_peak_counts
    ):
        if sample not in [
            "Powdered milk",
            "Ibuprofen methocarbamol",
        ]:  # remove two to save space in grant figure
            if (
                sample not in sample_plot_numbers
            ):  # if this is the first time we've plotted this sample type, start a new plot
                sample_plot_numbers[sample] = most_recent_plot_number
                most_recent_plot_number += 1
                collage_axs.flat[sample_plot_numbers[sample]].text(
                    0.015,
                    0.83,
                    sample,
                    size=7,
                    weight="bold",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=collage_axs.flat[sample_plot_numbers[sample]].transAxes,
                )
            collage_axs.flat[sample_plot_numbers[sample]].plot(
                numpy.array(peak_times) / 60.0,
                60.0 * numpy.array(peak_counts) / bin_width + 1,
                linewidth=0.5,
            )
            collage_axs.flat[sample_plot_numbers[sample]].set_yscale("log")
            collage_axs.flat[sample_plot_numbers[sample]].set_ylim(0.5, 10000)
            collage_axs.flat[sample_plot_numbers[sample]].set_xticks([])  # for ACS
            collage_axs.flat[sample_plot_numbers[sample]].set_yticks([])  # for ACS
            collage_axs.flat[sample_plot_numbers[sample]].set_facecolor("0.9")
            [
                x.set_linewidth(0)
                for x in collage_axs.flat[sample_plot_numbers[sample]].spines.values()
            ]
    collage_fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.10, wspace=0.03
    )  # for ACS
    collage_fig.savefig("collage_condensed.png", dpi=600)
    collage_fig.clf()


################
# Comparisons: #
################

peak_match_comparisons = []
peak_match_comparisons_with_self_comparisons = []
peak_mismatch_comparisons = []
peak_mismatch_comparisons_with_self_comparisons = []
peak_comparison_dict = {}
peak_comparison_dict_with_self_comparisons = {}

for a, (filename_a, sample_a, peak_times_a, peak_counts_a, baselines_a) in enumerate(zip(summary_filenames, summary_samples, summary_peak_times, summary_peak_counts, summary_baselines)):
    for_peak_comparison_dict = []
    for_peak_comparison_dict_with_self_comparisons = []
    for b, (filename_b, sample_b, peak_times_b, peak_counts_b, baselines_b) in enumerate(zip(summary_filenames, summary_samples, summary_peak_times, summary_peak_counts, summary_baselines)):
        peak_difference = int((numpy.absolute(peak_counts_a - peak_counts_b)).sum())  # ORIGINAL

        # DON'T include self-comparisons:
        if filename_a != filename_b:  # don't include statistics from self-comparisons
            if sample_a == sample_b:
                peak_match_comparisons.append(peak_difference)
            else:
                peak_mismatch_comparisons.append(peak_difference)
            for_peak_comparison_dict.append((filename_a, sample_a, filename_b, sample_b, peak_difference))

        # DO include self-comparisons:
        if sample_a == sample_b:
            peak_match_comparisons_with_self_comparisons.append(peak_difference)
        else:
            peak_mismatch_comparisons_with_self_comparisons.append(peak_difference)
        for_peak_comparison_dict_with_self_comparisons.append((filename_a, sample_a, filename_b, sample_b, peak_difference))

    peak_comparison_dict[filename_a] = for_peak_comparison_dict
    peak_comparison_dict_with_self_comparisons[filename_a] = for_peak_comparison_dict_with_self_comparisons

with open(args.figure + " comparisons.json", "w") as f:
    json.dump(peak_comparison_dict, f, indent=4)
with open(args.figure + " comparisons self.json", "w") as f:
    json.dump(peak_comparison_dict_with_self_comparisons, f, indent=4)

peak_match_failures = []
peak_fail_count = 0
sample_count = 0

for c in peak_comparison_dict:  # the peak and baseline dicts have the same set of keys
    plog(c)
    sample_count += 1
    peak_comparison_dict[c].sort(key=lambda tup: tup[4])
    plog("  PEAK COMPARISONS")
    for rank, comparison in enumerate(peak_comparison_dict[c][:3]):  # print the top 3 matches
        result = "❌"
        if comparison[1] == comparison[3]:
            result = "✅"
        if result == "❌" and rank == 0:
            plog("  ^^^^^PEAK FAIL^^^^^")
            peak_fail_count += 1
            peak_match_failures.append(comparison[0])
        plog(f"    {result} {comparison[1]} == {comparison[3]}   {comparison[4]}")
plog("PEAK MATCH FAILURES")
for f in peak_match_failures:
    plog(f"  {f}")

plog(f"{peak_fail_count} peak failures out of {sample_count} samples")
plog(f"Peak failure rate: {peak_fail_count / sample_count}")


# calculate and plot average fingerprints for each variety sample:
if args.figure == "variety":
    average_dict = {}
    for filename, sample, peak_times, peak_counts in zip(summary_filenames, summary_samples, summary_peak_times, summary_peak_counts):
        if sample not in average_dict:
            average_dict[sample] = numpy.array(peak_counts)
        else:
            average_dict[sample] = average_dict[sample] + numpy.array(peak_counts)
    for sample in average_dict:
        average_dict[sample] = average_dict[sample] / 3.0  # calculate average of 3 runs
    average_fig, average_axs = plt.subplots(1, 1, figsize=(7.5, 3))
    styles = []
    TABLEAU_11 = TABLEAU_COLORS
    TABLEAU_11["k"] = "k"
    for color in TABLEAU_11:  #  was TABLEAU_COLORS
        for line in ["solid", "dashed", "dotted"]:
            styles.append((color, line))
    for sample, style in zip(average_dict, styles):
        lines = average_axs.plot(numpy.array(peak_times) / 60.0, 60.0 * average_dict[sample] / bin_width + 1, label=str(sample), linewidth=1.5)
        lines[0].set_color(style[0])
        lines[0].set_linestyle(style[1])
    average_axs.set_yscale("log")
    average_axs.set_ylim(0.7, 1500)
    average_axs.set_yticks([1, 10, 100, 1000])
    average_axs.minorticks_off()
    average_axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    a = average_axs.get_yticks().tolist()
    a[0] = "0"
    average_axs.set_yticklabels(a)
    average_axs.set_xlabel("Time (minutes)")
    average_axs.set_ylabel("Particles per minute")
    average_axs.legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.43),
        fancybox=True,
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    average_axs.yaxis.set_label_coords(-0.12, 0.50)
    average_fig.subplots_adjust(left=0.065, right=0.44, bottom=0.15, top=0.99)
    average_fig.savefig("variety averages.pdf")
    average_fig.clf()


plog(f"Total number of measurements:  {measurement_count}")
plog(f"Number of out-of-dynamic-range measurements removed:  {oodr_count}")
plog(f"Number of noise peaks removed:  {noise_peak_count}")
plog(f"Done!  Elapsed time {(time.time() - start_time):.1f} seconds")
