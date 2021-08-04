import numpy as np

from GenericSuperDatasetv2 import get_matched_supix


class SupixMatch:
    def __init__(self, scan_id, z_id, pseudo_label_a, pseudo_label_b):
        self.scan_id = scan_id
        self.z_id = z_id
        self.match_map = {}

    def get_matches(self, scan_id, z_id, pseudo_label_a, pseudo_label_b):
        unique = np.unique(pseudo_label_a)
        for supix_value in unique:
            supix_binary = pseudo_label_a == supix_value
            match, score = get_matched_supix(supix_binary, pseudo_label_b)
            self.match_map[supix_value] = (match, score)




