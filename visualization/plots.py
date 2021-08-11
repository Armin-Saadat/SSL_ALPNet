import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os


def read_nii_bysitk(input_fid, peel_info=False):
    """
     reads nii to numpy through simpleitk
     peel info: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    if peel_info:
        info_obj = {
            "spacing": img_obj.GetSpacing(),
            "origin": img_obj.GetOrigin(),
            "direction": img_obj.GetDirection(),
            "array_size": img_np.shape
        }
        return img_np, info_obj
    else:
        return img_np


def plot_intersection(seq, depth, pid1, pid2, score, dstep=0, save=False, rank=0):
    """
    Args:
        seq: sequence (list) with the form [slice_a, slice_b, supix_a, supix_b]
        depth: number (depth) of first slice_a
        pid1: slice_a patient ID
        pid2: slice_b patient ID (both could be the same)
        score: matching IOU score
        dstep: depth distance between slice_a and slice_b
        save: if True plot will be saved
        rank: rank of matching (determined by score)

    Does:
        creates a fig containing three plots (slice_a and supix_a, slice_b and supix_b, matching intersection)
    """
    rank_letter = [None, 'first', 'second', 'third', 'fourth', 'fifth', 'sixth']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    for i, pic in enumerate(seq):
        if i == 0:
            ax1.imshow(pic + (100 * seq[i + 2]));
            ax1.set_title(f'Patient {pid1}   slice {depth}', fontsize=22);
        elif i == 1:
            ax2.imshow(pic + (100 * seq[i + 2]));
            ax2.set_title(f'Patient {pid2}   slice {depth + dstep}', fontsize=22);
        elif i == 4:
            ax3.imshow(pic)
            ax3.set_title(
                f'superpixels intersection\n{rank_letter[rank]} best match - IOU score: {round(score, 2)}', fontsize=22);
    if save:
        if rank == 1:
            os.mkdir(f'./supix_matching_visualization/P{pid1}_S{depth}')
        fig.savefig(f'./supix_matching_visualization/P{pid1}_S{depth}/P{pid1}_S{depth}_R{rank}_S{str(round(score, 2))[2:]}.png', transparent=True, bbox_inches='tight');
    return fig


def plot_best_matches_by_pid_slice_rank(pid, slice, rank, supix_matches):
    """
    Args:
        pid:  patient ID
        slice: slice number
        rank: number of matches to plot (eg. first best, second best, ...)
        supix_matches: self.supix_matches of SuperpixelDataset class

    Does:
        finds matches sorted by score and calls plot_intersection to create and save the plot.
    """
    label_path = f'../data/CHAOST2/chaos_MR_T2_normalized/superpix-MIDDLE_{pid}.nii.gz'
    obj = read_nii_bysitk(label_path)
    pseudo_label_a = obj[slice]
    best_ones = []
    for i, supix_value in enumerate(np.unique(pseudo_label_a)):
        if supix_value == 0:
            continue
        match, score = supix_matches.get(str(pid))[slice].get(supix_value)
        if i < rank + 1:
            best_ones.append((score, supix_value, match))
        else:
            best_ones.append((score, supix_value, match))
            best_ones.sort(reverse=True)
            best_ones.pop()
    for i, (score, supix_value, match) in enumerate(best_ones):
        supix_a = 10 * (pseudo_label_a == supix_value)
        supix_b = 10 * match.transpose(2, 0, 1)[0]
        intersect = supix_a + supix_b
        img_3d = read_nii_bysitk(f'../data/CHAOST2/chaos_MR_T2_normalized/image_{pid}.nii.gz')
        slice_a, slice_b = img_3d[slice], img_3d[slice + 1]
        seq = [slice_a, slice_b, supix_a, supix_b, intersect]
        plot_intersection(seq, depth=slice, pid1=pid, pid2=pid, score=score, dstep=1, save=True, rank=i + 1)

