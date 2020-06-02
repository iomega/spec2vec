"""Plotting functions for spec2vec"""
import numpy as np
from matplotlib import pyplot as plt


def plot_precentile(arr_ref, arr_sim, num_bins=1000, show_top_percentile=1.0,
                    ignore_diagonal=False):
    """ Plot top percentile (as specified by show_top_percentile) of best restults
    in arr_sim and compare against reference values in arr_ref.

    Args:
    -------
    arr_ref: numpy array
        Array of reference values to evaluate the quality of arr_sim.
    arr_sim: numpy array
        Array of similarity values to evaluate.
    num_bins: int
        Number of bins to divide data (default = 1000)
    show_top_percentile
        Choose which part to plot. Will plot the top 'show_top_percentile' part of
        all similarity values given in arr_sim. Default = 1.0
    """
    def _ignore_reference_nans(arr_ref, arr_sim):
        assert arr_ref.shape == arr_sim.shape, "Expected two arrays of identical shape."
        idx_not_nans = np.where(np.isnan(arr_ref) == False)
        arr_sim = arr_sim[idx_not_nans]
        arr_ref = arr_ref[idx_not_nans]
        return arr_ref, arr_sim

    if ignore_diagonal:
        np.fill_diagonal(arr_ref, np.nan)

    arr_ref, arr_sim = _ignore_reference_nans(arr_ref, arr_sim)

    start = int(arr_sim.shape[0] * show_top_percentile / 100)
    idx = np.argpartition(arr_sim, -start)
    starting_point = arr_sim[idx[-start]]
    if starting_point == 0:
        print("not enough datapoints != 0 above given top-precentile")

    # Remove all data below show_top_percentile
    low_as = np.where(arr_sim < starting_point)[0]

    length_selected = arr_sim.shape[0] - low_as.shape[0]  # start+1

    data = np.zeros((2, length_selected))
    data[0, :] = np.delete(arr_sim, low_as)
    data[1, :] = np.delete(arr_ref, low_as)
    data = data[:, np.lexsort((data[1, :], data[0, :]))]

    ref_score_cum = []

    for i in range(num_bins):
        low = int(i * length_selected / num_bins)
        # high = int((i+1) * length_selected/num_bins)
        ref_score_cum.append(np.mean(data[1, low:]))
    ref_score_cum = np.array(ref_score_cum)
    x_percentiles = (show_top_percentile / num_bins * (1 + np.arange(num_bins)))[::-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        x_percentiles,
        ref_score_cum,
        color='black')
    plt.xticks(np.linspace(0, show_top_percentile, 5),
               ["{:.2f}%".format(x) for x in np.linspace(0, show_top_percentile, 5)])
    plt.xlabel("Top percentile of spectral similarity score g(s,s')")
    plt.ylabel("Mean molecular similarity (f(t,t') within that percentile)")

    return ref_score_cum
