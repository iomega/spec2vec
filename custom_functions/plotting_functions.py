"""Plotting functions for spec2vec"""
import numpy as np
from matchms.similarity.collect_peak_pairs import collect_peak_pairs
from matplotlib import pyplot as plt
from scipy import spatial
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG, display
from spec2vec import SpectrumDocument
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity


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


def plot_spectra_comparison(spectrum1_in, spectrum2_in,
                            model,
                            intensity_weighting_power=0.5,
                            num_decimals=2,
                            min_mz=5,
                            max_mz=500,
                            intensity_threshold=0.01,
                            method='cosine',
                            tolerance=0.005,
                            wordsim_cutoff=0.5,
                            circle_size=5,
                            circle_scaling='wordsim',
                            padding=10,
                            display_molecules=False,
                            figsize=(12, 12),
                            filename=None):
    """ In-depth visual comparison of spectral similarity scores,
    calculated based on cosine/mod.cosine and Spev2Vec.

    Parameters
    ----------
    method: str
        'cosine' or 'modcos' (modified cosine score)
    circle_scaling: str
        Scale circles based on 'wordsim' or 'peak_product'
    """

    def apply_filters(s):
        s = normalize_intensities(s)
        s = select_by_mz(s, mz_from=min_mz, mz_to=max_mz)
        s = select_by_relative_intensity(s, intensity_from=intensity_threshold)
        s.losses = None
        return s

    spectrum1 = apply_filters(spectrum1_in)
    spectrum2 = apply_filters(spectrum2_in)

    plt.style.use('ggplot')
    plot_colors = ['darkcyan', 'purple']

    # Definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_wordsim = [left, bottom, width, height]
    rect_specx = [left, bottom + height + spacing, width, 0.2]
    rect_specy = [left + width, bottom, 0.2, height]

    document1 = SpectrumDocument(spectrum1, n_decimals=num_decimals)
    document2 = SpectrumDocument(spectrum2, n_decimals=num_decimals)

    # Remove words/peaks that are not in dictionary
    select1 = np.asarray([i for i, word in enumerate(document1.words) if word in model.wv.vocab])
    select2 = np.asarray([i for i, word in enumerate(document2.words) if word in model.wv.vocab])
    peaks1 = np.asarray(spectrum1.peaks[:]).T
    peaks2 = np.asarray(spectrum2.peaks[:]).T
    peaks1 = peaks1[select1, :]
    peaks2 = peaks2[select2, :]
    min_peaks1 = np.min(peaks1[:, 0])
    min_peaks2 = np.min(peaks2[:, 0])
    max_peaks1 = np.max(peaks1[:, 0])
    max_peaks2 = np.max(peaks2[:, 0])

    word_vectors1 = model.wv[[document1.words[x] for x in select1]]
    word_vectors2 = model.wv[[document2.words[x] for x in select2]]

    csim_words = 1 - spatial.distance.cdist(word_vectors1, word_vectors2, 'cosine')
    csim_words[csim_words < wordsim_cutoff] = 0  # Remove values below cutoff
    print(np.min(csim_words), np.max(csim_words))

    # Plot spectra
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    # Word similariy plot (central)
    ax_wordsim = plt.axes(rect_wordsim)
    ax_wordsim.tick_params(direction='in', top=True, right=True)
    # Spectra plot (top)
    ax_specx = plt.axes(rect_specx)
    ax_specx.tick_params(direction='in', labelbottom=False)
    # Spectra plot 2 (right)
    ax_specy = plt.axes(rect_specy)
    ax_specy.tick_params(direction='in', labelleft=False)

    # Spec2Vec similarity plot:
    # -------------------------------------------------------------------------
    data_x = []
    data_y = []
    data_z = []
    data_peak_product = []
    for i in range(len(select1)):
        for j in range(len(select2)):
            data_x.append(peaks1[i, 0])
            data_y.append(peaks2[j, 0])
            data_z.append(csim_words[i, j])
            data_peak_product.append(peaks1[i, 1] * peaks2[j, 1])

    # Sort by word similarity
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)
    data_peak_product = np.array(data_peak_product)
    idx = np.lexsort((data_x, data_y, data_z))

    cm = plt.cm.get_cmap('RdYlBu_r')  # 'YlOrRd') #'RdBu_r')

    # Plot word similarities
    if circle_scaling == 'peak_product':
        wordsimplot = ax_wordsim.scatter(data_x[idx],
                                         data_y[idx],
                                         s=100 * circle_size *
                                         (0.01 + data_peak_product[idx]**2),
                                         c=data_z[idx],
                                         cmap=cm,
                                         alpha=0.6)
    elif circle_scaling == 'wordsim':
        wordsimplot = ax_wordsim.scatter(data_x[idx],
                                         data_y[idx],
                                         s=100 * circle_size *
                                         (0.01 + data_z[idx]**2),
                                         c=data_z[idx],
                                         cmap=cm,
                                         alpha=0.6)

    # (Modified) Cosine similarity plot:
    # -------------------------------------------------------------------------
    if method == 'cosine':
        score_classical, used_matches = cosine_score(spectrum1, spectrum2, tolerance, modified_cosine=False)
    elif method == 'modcos':
        score_classical, used_matches = cosine_score(spectrum1, spectrum2, tolerance, modified_cosine=True)
    else:
        print("Given method unkown.")

    idx1, idx2, _ = zip(*used_matches)
    cosine_x = []
    cosine_y = []
    for i in range(len(idx1)):
        if idx1[i] in select1 and idx2[i] in select2:
            cosine_x.append(peaks1[idx1[i], 0])
            cosine_y.append(peaks2[idx2[i], 0])

    # Plot (mod.) cosine similarities
    ax_wordsim.scatter(cosine_x, cosine_y, s=100, c='black', marker=(5, 2))
    ax_wordsim.set_xlim(min_peaks1 - padding, max_peaks1 + padding)
    ax_wordsim.set_ylim(min_peaks2 - padding, max_peaks2 + padding)
    ax_wordsim.set_xlabel('spectrum 1 - fragment mz', fontsize=16)
    ax_wordsim.set_ylabel('spectrum 2 - fragment mz', fontsize=16)
    ax_wordsim.tick_params(labelsize=13)

    # Plot spectra 1
    ax_specx.vlines(peaks1[:, 0], [0], peaks1[:, 1], color=plot_colors[0])
    ax_specx.plot(peaks1[:, 0], peaks1[:, 1], '.')  # Stem ends
    ax_specx.plot([peaks1[:, 0].max(), peaks1[:, 0].min()], [0, 0],
                  '--')  # Middle bar
    ax_specx.set_xlim(min_peaks1 - padding, max_peaks1 + padding)
    ax_specx.set_ylabel('peak intensity (relative)', fontsize=16)
    ax_specx.tick_params(labelsize=13)

    # Plot spectra 2
    ax_specy.hlines(peaks2[:, 0], [0], peaks2[:, 1], color=plot_colors[1])
    ax_specy.plot(peaks2[:, 1], peaks2[:, 0], '.')  # Stem ends
    ax_specy.plot([0, 0], [peaks2[:, 0].min(), peaks2[:, 0].max()],
                  '--')  # Middle bar
    ax_specy.set_ylim(min_peaks2 - padding, max_peaks2 + padding)
    ax_specy.set_xlabel('peak intensity (relative)', fontsize=16)
    ax_specy.tick_params(labelsize=13)

    fig.colorbar(wordsimplot, ax=ax_specy)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    # Plot molecules
    # -------------------------------------------------------------------------
    if display_molecules:
        smiles = [spectrum1.get("smiles"), spectrum2.get("smiles")]
        molecules = [Chem.MolFromSmiles(x) for x in smiles]
        display(Draw.MolsToGridImage(molecules, molsPerRow=2, subImgSize=(400, 400)))


# def scour_svg_cleaning(target, source, env=[]):
#     """ Use scour to clean an svg file.

#     """
#     options = scour.generateDefaultOptions()

#     # override defaults for max cleansing
#     options.enable_viewboxing = True
#     options.strip_comments = True
#     options.strip_ids = True
#     options.remove_metadata = True
#     options.indent_type = None
#     options.shorten_ids = True

#     if 'SCOUR_OPTIONS' in env:
#         options.__dict__.update(env['SCOUR_OPTIONS'])

#     instream = open(source, 'rb')
#     outstream = open(target, 'wb')

#     scour.start(options, instream, outstream)


# def plot_molecules(smiles_lst, filename=None):
#     """ Plot molecule from smile(s).
#     Uses Scour to clean rdkit svg.

#     filename: str
#         If filename is given, molecules will be saved to filename.
#     """
#     if not isinstance(smiles_lst, list):
#         smiles_lst = [smiles_lst]
#     for i, smiles in enumerate(smiles_lst):
#         temp_file = "draw_mols_temp.svg"
#         mol = Chem.MolFromSmiles(smiles)
#         Draw.MolToFile(mol, temp_file)

#         # Clean svg using scour
#         if filename is not None:
#             file = filename.split('.svg')[0] + str(i) + '.svg'
#         else:
#             file = "draw_mols_temp_corr.svg"
#         scour(file, temp_file, [])

#         # Display cleaned svg
#         display(SVG(filename=temp_file))


def cosine_score(spectrum1, spectrum2, tolerance, modified_cosine=False):
    """

    Parameters
    ----------
    spectrum1 : TYPE
        DESCRIPTION.
    spectrum2 : TYPE
        DESCRIPTION.
    tolerance : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def get_peaks_arrays():
        """Get peaks mz and intensities as numpy array."""
        spec1 = np.vstack((spectrum1.peaks.mz, spectrum1.peaks.intensities)).T
        spec2 = np.vstack((spectrum2.peaks.mz, spectrum2.peaks.intensities)).T
        assert max(spec1[:, 1]) <= 1, ("Input spectrum1 is not normalized. ",
                                       "Apply 'normalize_intensities' filter first.")
        assert max(spec2[:, 1]) <= 1, ("Input spectrum2 is not normalized. ",
                                       "Apply 'normalize_intensities' filter first.")
        return spec1, spec2

    def get_matching_pairs():
        """Get pairs of peaks that match within the given tolerance."""
        zero_pairs = collect_peak_pairs(spec1, spec2, tolerance, shift=0.0)
        if modified_cosine:
            message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            assert spectrum1.get("precursor_mz") and spectrum2.get("precursor_mz"), message
            mass_shift = spectrum1.get("precursor_mz") - spectrum2.get("precursor_mz")
            nonzero_pairs = collect_peak_pairs(spec1, spec2, tolerance, shift=mass_shift)
            unsorted_matching_pairs = zero_pairs + nonzero_pairs
        else:
            unsorted_matching_pairs = zero_pairs
        return sorted(unsorted_matching_pairs, key=lambda x: x[2], reverse=True)

    def calc_score():
        """Calculate cosine similarity score."""
        used1 = set()
        used2 = set()
        score = 0.0
        used_matches = []
        for match in matching_pairs:
            if not match[0] in used1 and not match[1] in used2:
                score += match[2]
                used1.add(match[0])  # Every peak can only be paired once
                used2.add(match[1])  # Every peak can only be paired once
                used_matches.append(match)
        # Normalize score:
        score = score/max(np.sum(spec1[:, 1]**2), np.sum(spec2[:, 1]**2))
        return score, used_matches

    spec1, spec2 = get_peaks_arrays()
    matching_pairs = get_matching_pairs()
    return calc_score()
