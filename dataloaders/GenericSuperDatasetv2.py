"""
Dataset for training with pseudolabels
TODO:
1. Merge with manual annotated dataset
2. superpixel_scale -> superpix_config, feed like a dict
"""
import glob
import torch
import copy
import json
import re
import pickle

from dataloaders.common import BaseDataset
from dataloaders.dataset_utils import *
from util.utils import CircularList
from matplotlib import pyplot as plt


class SuperpixelDataset(BaseDataset):
    def __init__(self, which_dataset, base_dir, idx_split, mode, transforms, scan_per_load, num_rep=2, min_fg='',
                 nsup=1, fix_length=None, tile_z_dim=3, exclude_list=[], superpix_scale='SMALL', figPath=None,
                 supix_matching_threshold=0.7, create_supix_matching_prep_file=False, use_supix_matching=False,
                 exclude_testing_objs=True, root_of_supix_matches_file='/HDD/SSL_ALPNet_models', **kwargs):
        """
        Pseudolabel dataset
        Args:
            which_dataset:      name of the dataset to use
            base_dir:           directory of dataset
            idx_split:          index of data split as we will do cross validation
            mode:               'train', 'val'.
            nsup:               number of scans used as support. currently idle for superpixel dataset
            transforms:         data transform (augmentation) function
            scan_per_load:      loading a portion of the entire dataset, in case that the dataset is too large to fit into the memory. Set to -1 if loading the entire dataset at one time
            num_rep:            Number of augmentation applied for a same pseudolabel
            tile_z_dim:         number of identical slices to tile along channel dimension, for fitting 2D single-channel medical images into off-the-shelf networks designed for RGB natural images
            fix_length:         fix the length of dataset
            exclude_list:       Labels to be excluded
            superpix_scale:     config of superpixels
        """
        super(SuperpixelDataset, self).__init__(base_dir)

        self.figPath = figPath
        self.supix_matching_threshold = supix_matching_threshold
        self.exclude_testing_objs = exclude_testing_objs
        self.use_supix_matching = use_supix_matching
        self.matches_num = 0
        self.all_batches = 0

        self.img_modality = DATASET_INFO[which_dataset]['MODALITY']
        self.sep = DATASET_INFO[which_dataset]['_SEP']
        self.pseu_label_name = DATASET_INFO[which_dataset]['PSEU_LABEL_NAME']
        self.real_label_name = DATASET_INFO[which_dataset]['REAL_LABEL_NAME']

        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        assert mode == 'train'
        self.fix_length = fix_length
        self.nclass = len(self.pseu_label_name)
        self.num_rep = num_rep
        self.tile_z_dim = tile_z_dim

        # find scans in the data folder
        self.nsup = nsup
        self.base_dir = base_dir
        self.img_pids = [re.findall('\d+', fid)[-1] for fid in glob.glob(self.base_dir + "/image_*.nii.gz")]
        self.img_pids = CircularList(sorted(self.img_pids, key=lambda x: int(x)))

        # experiment configs
        self.exclude_lbs = exclude_list
        self.superpix_scale = superpix_scale
        if len(exclude_list) > 0:
            print(f'Dataset: the following classes has been excluded {exclude_list}')
        self.idx_split = idx_split
        self.scan_ids = self.get_scanids(mode, idx_split)  # patient ids of the entire fold
        self.min_fg = min_fg if isinstance(min_fg, str) else str(min_fg)
        self.scan_per_load = scan_per_load

        self.info_by_scan = None
        self.img_lb_fids = self.organize_sample_fids()  # information of scans of the entire fold
        self.norm_func = get_normalize_op(self.img_modality,
                                          [fid_pair['img_fid'] for _, fid_pair in self.img_lb_fids.items()])

        if self.is_train:
            if scan_per_load > 0:  # if the dataset is too large, only reload a subset in each sub-epoch
                self.pid_curr_load = np.random.choice(self.scan_ids, replace=False, size=self.scan_per_load)
            else:  # load the entire set without a buffer
                self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        else:
            raise Exception
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.overall_slice_by_cls = self.read_classfiles()

        # supix matches preprocess
        if create_supix_matching_prep_file:
            print("\n--- start creating and saving supix matches ---\n")
            self.save_all_supix_matches()
        if use_supix_matching:
            print('\n----- TRAINING MODE: SUPIX MATCHING -----\n')
            print("--- trying to load supix matches ---")
            try:
                with open(root_of_supix_matches_file + 'supix_matches/supix_matches.pkl', 'rb') as f:
                    self.supix_matches = pickle.load(f)
                print("\n--- supix matches loaded completelty ---\n")
            except:
                print(
                    '\n------ "use_supix_matching" is true but no preprocessed file is available. Will find matches on fly. ------\n')
                self.supix_matches = None
        else:
            self.supix_matches = None

        print("Initial scans loaded: ")
        print(self.pid_curr_load)

    def get_scanids(self, mode, idx_split):
        """
        Load scans by train-test split
        leaving one additional scan as the support scan. if the last fold, taking scan 0 as the additional one
        Args:
            idx_split: index for spliting cross-validation folds
        """
        val_ids = copy.deepcopy(self.img_pids[self.sep[idx_split]: self.sep[idx_split + 1] + self.nsup])
        if mode == 'train':
            return [ii for ii in self.img_pids if ii not in val_ids]
        elif mode == 'val':
            return val_ids

    def reload_buffer(self):
        """
        Reload a only portion of the entire dataset, if the dataset is too large
        1. delete original buffer
        2. update self.ids_this_batch
        3. update other internel variables like __len__
        """
        if self.scan_per_load <= 0:
            print("We are not using the reload buffer, doing nothing")
            return -1

        del self.actual_dataset
        del self.info_by_scan

        self.pid_curr_load = np.random.choice(self.scan_ids, size=self.scan_per_load, replace=False)
        self.actual_dataset = self.read_dataset()
        self.size = len(self.actual_dataset)
        self.update_subclass_lookup()
        print(f'Loader buffer reloaded with a new size of {self.size} slices')

    def organize_sample_fids(self):
        out_list = {}
        for curr_id in self.scan_ids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid = os.path.join(self.base_dir, f'superpix-{self.superpix_scale}_{curr_id}.nii.gz')

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def organize_all_fids(self):
        out_list = {}
        for curr_id in self.img_pids:
            curr_dict = {}

            _img_fid = os.path.join(self.base_dir, f'image_{curr_id}.nii.gz')
            _lb_fid = os.path.join(self.base_dir, f'superpix-{self.superpix_scale}_{curr_id}.nii.gz')

            curr_dict["img_fid"] = _img_fid
            curr_dict["lbs_fid"] = _lb_fid
            out_list[str(curr_id)] = curr_dict
        return out_list

    def get_matches(self, pseudo_label_a, pseudo_label_b):
        match_map = {}
        unique = np.unique(pseudo_label_a)
        for supix_value in unique:
            supix_binary = pseudo_label_a == supix_value
            match, score = self.get_matched_supix(supix_binary, pseudo_label_b)
            match_map[supix_value] = (match, score)
        return match_map

    def save_all_supix_matches(self):
        all_fids = self.organize_all_fids()
        supix_matches = {}
        for scan_id, itm in all_fids.items():
            print("scan_id", scan_id)
            img, _info = read_nii_bysitk(itm["img_fid"], peel_info=True)  # get the meta information out
            img = img.transpose(1, 2, 0)

            supix_matches[scan_id] = [None for _ in range(img.shape[-1] - 1)]

            img = np.float32(img)
            img = self.norm_func(img)

            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1, 2, 0)
            lb = np.int32(lb)

            img = img[:256, :256, :]
            lb = lb[:256, :256, :]

            assert img.shape[-1] == lb.shape[-1]

            lb_a = lb[..., 0: 1]

            for ii in range(1, img.shape[-1]):
                supix_matches.get(scan_id)[ii - 1] = self.get_matches(lb_a, lb[..., ii: ii + 1])
                lb_a = lb[..., ii: ii + 1]

        with open('./supix_matches/supix_matches.pkl', 'wb') as f:
            pickle.dump(supix_matches, f)

    def read_dataset(self):
        """
        Read images into memory and store them in 2D
        Build tables for the position of an individual 2D slice in the entire dataset
        """
        out_list = []
        self.info_by_scan = {}  # meta data of each scan

        for scan_id, itm in self.img_lb_fids.items():
            if scan_id not in self.pid_curr_load:
                continue

            img, _info = read_nii_bysitk(itm["img_fid"], peel_info=True)  # get the meta information out
            img = img.transpose(1, 2, 0)
            self.info_by_scan[scan_id] = _info

            img = np.float32(img)
            img = self.norm_func(img)

            lb = read_nii_bysitk(itm["lbs_fid"])
            lb = lb.transpose(1, 2, 0)
            lb = np.int32(lb)

            img = img[:256, :256, :]
            lb = lb[:256, :256, :]

            # format of slices: [axial_H x axial_W x Z]
            assert img.shape[-1] == lb.shape[-1]

            # re-organize 3D images into 2D slices and record essential information for each slice
            out_list.append({"img": img[..., 0:1],
                             "lb": lb[..., 0:1],
                             "supix_values": np.unique(lb[..., 0:1]),
                             "is_start": True,
                             "is_end": False,
                             "nframe": img.shape[-1],
                             "scan_id": scan_id,
                             "z_id": 0})

            for ii in range(1, img.shape[-1] - 1):
                out_list.append({"img": img[..., ii: ii + 1],
                                 "lb": lb[..., ii: ii + 1],
                                 "is_start": False,
                                 "is_end": False,
                                 "supix_values": np.unique(lb[..., ii: ii + 1]),
                                 "nframe": -1,
                                 "scan_id": scan_id,
                                 "z_id": ii
                                 })

            ii = img.shape[-1] - 1  # last slice of a 3D volume
            out_list.append({"img": img[..., ii: ii + 1],
                             "lb": lb[..., ii: ii + 1],
                             "is_start": False,
                             "is_end": True,
                             "supix_values": np.unique(lb[..., ii: ii + 1]),
                             "nframe": -1,
                             "scan_id": scan_id,
                             "z_id": ii
                             })

        return out_list

    def read_classfiles(self):
        """
        Load the scan-slice-class indexing file
        """
        with open(os.path.join(self.base_dir, f'classmap_{self.min_fg}.json'), 'r') as fopen:
            cls_map = json.load(fopen)
            fopen.close()

        with open(os.path.join(self.base_dir, 'classmap_1.json'), 'r') as fopen:
            self.tp1_cls_map = json.load(fopen)
            fopen.close()

        return cls_map

    @staticmethod
    def get_random_supix_mask(super_map, supix_values, supix_value=None):
        """
        pick up a certain super-pixel class or multiple classes, and binarize it into segmentation target
        Args:
            super_map:      super-pixel map
            supix_value:         if given, pick up a certain superpixel. Otherwise, draw a random one
            supix_values:    superpixel's values e.g: [1, 2, ... 5, 7]
        """
        if supix_value is None:
            supix_value = int(supix_values[torch.randint(len(supix_values), (1,))])

        return np.float32(super_map == supix_value), supix_value

    @staticmethod
    def get_matched_supix(input_supix, pseudo_lable):
        assert input_supix.shape == pseudo_lable.shape
        supix_values = np.unique(pseudo_lable)
        intersections = dict((supix_value, 0) for supix_value in supix_values)
        sizes = dict((supix_value, 0) for supix_value in supix_values)
        for i in range(pseudo_lable.shape[0]):
            for j in range(pseudo_lable.shape[1]):
                supix_value = pseudo_lable[i][j][0]
                sizes[supix_value] = sizes.get(supix_value) + 1
                if input_supix[i][j][0] == 1:
                    intersections[supix_value] = intersections.get(supix_value) + 1
        best_value = None
        best_score = 0
        input_supix_size = input_supix.sum()
        for supix_value in supix_values:
            intersection = intersections.get(supix_value)
            size = sizes.get(supix_value)
            union = input_supix_size + size - intersection
            if union == 0:
                continue
            score = intersection / union
            if score >= best_score:
                best_score = score
                best_value = supix_value
        if best_value is None:
            return None, -1
        else:
            return np.float32(pseudo_lable == best_value), best_score

    @staticmethod
    def pair_plot(seq: list, score: int, saving_path: str, curr_scan_id, curr_z_id, next_scan_id, next_z_id):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        for i, pic in enumerate(seq):
            if i > 1: break;
            if i == 0:
                ax1.imshow(pic + seq[i + 2]);
                ax1.set(title=f'CurrScanID: {curr_scan_id}, z_id: {curr_z_id} score: {score}')
            else:
                ax2.imshow(pic + seq[i + 2]);
                ax2.set(title=f'NextScanID: {next_scan_id}, z_id: {next_z_id} score: {score}')
        if saving_path:
            fig.savefig(saving_path, transparent=True, bbox_inches='tight')

    def create_sample(self, comp, sample_dict):
        img, lb = self.transforms(comp, c_img=1, c_label=1, nclass=self.nclass, is_train=True, use_onehot=False)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        lb = torch.from_numpy(lb.squeeze(-1))

        if self.tile_z_dim:
            img = img.repeat([self.tile_z_dim, 1, 1])
            assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

        is_start = sample_dict["is_start"]
        is_end = sample_dict["is_end"]
        nframe = np.int32(sample_dict["nframe"])
        scan_id = sample_dict["scan_id"]
        z_id = sample_dict["z_id"]

        sample = {"image": img,
                  "label": lb,
                  "is_start": is_start,
                  "is_end": is_end,
                  "nframe": nframe,
                  "scan_id": scan_id,
                  "z_id": z_id
                  }

        # Add auxiliary attributes
        if self.aux_attrib is not None:
            for key_prefix in self.aux_attrib:
                # Process the data sample, create new attributes and save them in a dictionary
                aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
                for key_suffix in aux_attrib_val:
                    # one function may create multiple attributes, so we need suffix to distinguish them
                    sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        slice_a = self.actual_dataset[index]
        supix_values = slice_a['supix_values']
        if len(supix_values) < 1 or slice_a["is_end"]:
            return self.__getitem__(index + 1)

        if self.exclude_testing_objs:
            # if using setting 1, this slice need to be excluded since it contains label which is supposed to be unseen
            for _ex_cls in self.exclude_lbs:
                if slice_a["z_id"] in self.tp1_cls_map[self.real_label_name[_ex_cls]][slice_a["scan_id"]]:
                    return self.__getitem__(torch.randint(low=0, high=self.__len__() - 1, size=(1,)))

        if self.use_supix_matching:
            slice_b = self.actual_dataset[index + 1]
            assert slice_a["scan_id"] == slice_b["scan_id"]
            assert slice_a["z_id"] + 1 == slice_b["z_id"]
            if self.exclude_testing_objs:
                for _ex_cls in self.exclude_lbs:
                    if slice_b["z_id"] in self.tp1_cls_map[self.real_label_name[_ex_cls]][slice_b["scan_id"]]:
                        return self.__getitem__(torch.randint(low=0, high=self.__len__() - 1, size=(1,)))

        image_a = slice_a["img"]
        pseudo_label_a = slice_a["lb"]
        supix_a, supix_value_a = self.get_random_supix_mask(pseudo_label_a, supix_values)
        comp_a = np.concatenate([image_a, supix_a], axis=-1)

        if self.use_supix_matching:
            image_b = slice_b["img"]
            pseudo_label_b = slice_b["lb"]
            if self.supix_matches is not None:
                # read match from preprocessed file
                supix_b, matching_score = self.supix_matches.get(slice_a["scan_id"])[slice_a["z_id"]].get(supix_value_a)
            else:
                # find match on fly
                supix_b, matching_score = self.get_matched_supix(supix_a, pseudo_label_b)

            if matching_score < self.supix_matching_threshold:
                self.all_batches += 1
                sample_b = self.create_sample(comp_a, slice_a)
            else:
                self.matches_num += 1
                self.all_batches += 1
                comp_b = np.concatenate([image_b, supix_b], axis=-1)
                sample_b = self.create_sample(comp_b, slice_b)
        else:
            sample_b = self.create_sample(comp_a, slice_a)

        sample_a = self.create_sample(comp_a, slice_a)

        r = np.random.uniform()
        if r > 0.5:
            pair_buffer = [sample_a, sample_b]
        else:
            pair_buffer = [sample_b, sample_a]
        # if self.use_supix_matching and r < 0.005:
        #     print(
        #         f'\n======== (estimation) num_used_matches: {self.matches_num},   num_all: {self.all_batches} ========\n')

        support_images = []
        support_mask = []
        support_class = []

        query_images = []
        query_labels = []
        query_class = []

        for idx, itm in enumerate(pair_buffer):
            if idx % 2 == 0:
                support_images.append(itm["image"])
                support_class.append(1)  # pseudolabel class
                support_mask.append(self.getMaskMedImg(itm["label"], 1, [1]))
            else:
                query_images.append(itm["image"])
                query_class.append(1)
                query_labels.append(itm["label"])

        return {'class_ids': [support_class],
                'support_images': [support_images],
                'support_mask': [support_mask],
                'query_images': query_images,
                'query_labels': query_labels,
                }

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        if self.fix_length is not None:
            assert self.fix_length >= len(self.actual_dataset)
            return self.fix_length
        else:
            return len(self.actual_dataset)

    def print_augment_ratio(self):
        return self.our_idea_num, self.paper_idea_num

    @staticmethod
    def getMaskMedImg(label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
            class_ids:      all class id in this episode
        """
        fg_mask = torch.where(label == class_id, torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id, torch.ones_like(label), torch.zeros_like(label))
        for class_id in class_ids:
            bg_mask[label == class_id] = 0
        return {'fg_mask': fg_mask,
                'bg_mask': bg_mask}
