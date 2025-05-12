########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import glob

import numpy as np
import cv2

from ..dataset_base import DatasetBase
from .streethazards import StreetHazardsBase


class StreetHazards(StreetHazardsBase, DatasetBase):
    def __init__(
        self,
        data_dir=None,
        n_classes=12,
        n_samples=None,
        n_skips=None,
        n_offset=0,
        split="train",
        with_input_orig=False,
        overfit=False,
        classes=19,
    ):
        super(StreetHazards, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        self._n_classes = classes
        self._split = split
        self._with_input_orig = with_input_orig
        self._cameras = ["camera1"]  # just a dummy camera name
        self.overfit = overfit

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            print(split)

            # laoad subdir paths
            if split == "test":
                images_path = os.path.join(data_dir, "test/images")
                annotations_path = os.path.join(data_dir, "test/annotations")
            if split == "train":
                images_path = os.path.join(data_dir, "train/images/training")
                annotations_path = os.path.join(data_dir, "train/annotations/training")
            if split == "valid" or split == "val" or self.overfit:
                images_path = os.path.join(data_dir, "train/images/validation")
                annotations_path = os.path.join(data_dir, "train/annotations/validation")

            self.images = []
            self.labels = []

            # load file lists
            for i, filename in enumerate(glob.iglob(images_path+"/**/*.png", recursive=True)):
                # skip some images
                if (n_skips is not None) and ((i+n_offset) % n_skips) != 0 and split == "train":
                    continue
                self.images.append(filename)
            for i, filename in enumerate(glob.iglob(annotations_path+"/**/*.png", recursive=True)):
                # skip some images
                if (n_skips is not None) and ((i+n_offset) % n_skips) != 0 and split == "train":
                    continue
                self.labels.append(filename)

            # limit the number of samples in training data
            if n_samples is not None and split == "train":
                self.images = self.images[:n_samples] if n_samples < len(self.images) else self.images
                self.labels = self.labels[:n_samples] if n_samples < len(self.labels) else self.labels

            self.images.sort()
            self.labels.sort()

            self._files = {}

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")
        # class names, class colors, and label directory
        self._class_names = self.CLASS_NAMES_FULL
        self._class_colors = np.array(self.CLASS_COLORS_FULL, dtype="uint8")
        self._label_dir = self.LABELS_FULL_DIR

        print(f"DATASET({data_dir}) image len:{len(self.images)}")
        print(f"DATASET({data_dir}) labels len:{len(self.labels)}")

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, filename):
        # all the other files are pngs
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_name(self, idx):
        return self.images[idx]

    def load_image(self, idx):
        return self._load(self.images[idx])

    def load_label(self, idx):
        label = self._load(self.labels[idx])
        label = label - 1 # 1-14 -> 0-13
        return label

    def __len__(self):
        if self.overfit:
            return len(self.images) # 2
        return len(self.images) # 40
