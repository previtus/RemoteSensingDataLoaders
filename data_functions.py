import fsspec
import glob
import numpy as np
import rasterio

def load_all_tile_indices_from_folder(settings_dataset):
    allfiles = glob.glob(settings_dataset["data_base_path"]+"/*.tif")
    allfiles.sort()
    tiles = []

    for idx,filename in enumerate(allfiles):

        tiles_from_file = file_to_tiles_indices(filename, settings_dataset, 
            tile_px_size = settings_dataset["tile_px_size"], tile_overlap_px = settings_dataset["tile_overlap_px"], 
            include_last_row_colum_extra_tile = settings_dataset["include_last_row_colum_extra_tile"])

        tiles += tiles_from_file
        print(idx, filename, "loaded", len(tiles_from_file), "tiles.")


    print("Loaded:", len(tiles), "total tile indices")
    return tiles

def file_to_tiles_indices(filename, settings, tile_px_size = 128, tile_overlap_px = 4, 
                          include_last_row_colum_extra_tile = True):
    """
    Opens one tif file and extracts all tiles (given tile size and overlap).
    Returns list of indices to the tile (to postpone in memory loading).
    """

    with rasterio.open(filename) as src:
        filename_shape = src.height, src.width

    data_h, data_w = filename_shape
    if data_h < tile_px_size or data_w < tile_px_size:
        # print("skipping, too small!")
        return []

    h_tiles_n = int(np.floor((data_h-tile_overlap_px) / (tile_px_size-tile_overlap_px)))
    w_tiles_n = int(np.floor((data_w-tile_overlap_px) / (tile_px_size-tile_overlap_px)))

    tiles = []
    tiles_X = []
    tiles_Y = []
    for h_idx in range(h_tiles_n):
            for w_idx in range(w_tiles_n):
                    tiles.append([w_idx * (tile_px_size-tile_overlap_px), h_idx * (tile_px_size-tile_overlap_px)])
    if include_last_row_colum_extra_tile:
            for w_idx in range(w_tiles_n):
                    tiles.append([w_idx * (tile_px_size-tile_overlap_px), data_h - tile_px_size])
            for h_idx in range(h_tiles_n):
                    tiles.append([data_w - tile_px_size, h_idx * (tile_px_size-tile_overlap_px)])
            tiles.append([data_w - tile_px_size, data_h - tile_px_size])

    # Save file ID + corresponding tiles[]
    tiles_indices = [[filename]+t+[tile_px_size,tile_px_size] for t in tiles]
    return tiles_indices

def load_tile_idx(tile, settings):
    """
    Loads tile data values from the saved indices (file and window locations).
    """
    filename, x, y, w, h = tile

    # load window:
    window = rasterio.windows.Window(row_off=y, col_off=x, width=w, height=h)

    with rasterio.open(filename) as src:
        tile_data = src.read(window=window)

    return tile_data


import numpy as np
import pandas as pd
import math

class DataNormalizer():

    def __init__(self, settings):
        self.settings_dataset = settings.dataset
        self.settings_dataloader = settings.dataloader

        """
        TileDatasets hold descriptors and select only some bands

        class Normalizer
            can normalize, denormalize data
                knows what to do with NaN areas (outside of the swaths)
            can load statistics from samples
            knows precomputed statistics for each band (id from descriptor)

        ^ each TileDataset uses it after loading the tile data ...
        """

        pass

    def setup(self, data_module):
        if self.settings_dataset.normalization_precalculated_file is not "":
            self.load_normalization_parameters(self.settings_dataset.normalization_precalculated_file)

    def normalize_x(self, x, nan_to_zero = True):
        norm_params = self.normalization_parameters
        
        # nan_value = -9999.0
        # x[x <= nan_value] = np.nan

        bands = x.shape[0]
        for band_idx in range(bands):
            a_min, b_max = norm_params[band_idx]
            x[band_idx,:,:] = (x[band_idx,:,:] - a_min) / (b_max - a_min)
            # stats = np.nanmin(x[band_idx,:,:]), np.nanmax(x[band_idx,:,:]), np.nanmean(x[band_idx,:,:])
            # print("band", nanometers[band_idx], "stats:", stats)

        if nan_to_zero:
            x = np.nan_to_num(x, nan=0.0)

        return x # (C, H, W)

    def denormalize_x(self, x):
        norm_params = self.normalization_parameters
        
        bands = x.shape[0]
        for band_idx in range(bands):
            a_min, b_max = norm_params[band_idx]
            # reverting:
            x[band_idx,:,:] = (x[band_idx,:,:] * (b_max - a_min)) + a_min

        return x # (C, H, W)

    # Used to get normalization ranges from a sample of data, to be pre-calculated
    def estimate_from_data(self, x):
        print("estimate_from_data of shape (B, C, H, W):")
        print(x.shape)

        normalization_parameters = {}

        n_bands = x.shape[1]
        for i in range(0,n_bands):
            per_band = x[:,i,:,:]
            all_pixels = per_band.flatten()
            
            stats = np.min(all_pixels), np.max(all_pixels), np.mean(all_pixels)
            print("Band", str(i).zfill(3) , "min, max, mean", stats)
            
            # Proposed processing
            all_pixels = np.clip(all_pixels,np.nanquantile(all_pixels, 0.1), np.nanquantile(all_pixels, 0.9))
            
            a_min = np.nanmin(all_pixels)
            b_max = np.nanmax(all_pixels)
            all_pixels = (all_pixels - a_min) / (b_max - a_min)
            stats = np.nanmin(all_pixels), np.nanmax(all_pixels), np.nanmean(all_pixels)
            print("    ", str(i).zfill(3) , "min, max, mean", stats)

            normalization_parameters[i] = [str(a_min), str(b_max)]

        print("Estimated normalization parameters from a sample of data")
        self.normalization_parameters = normalization_parameters


    def save_normalization_parameters(self, file):
        df = pd.DataFrame.from_dict(self.normalization_parameters, orient="index")
        df.to_csv(file)


    def load_normalization_parameters(self, file):
        df = pd.read_csv(file, index_col=0)
        d = df.to_dict("split")
        normalization_parameters_loaded = dict(zip(d["index"], d["data"]))
        for key in normalization_parameters_loaded:
            values = normalization_parameters_loaded[key]
            normalization_parameters_loaded[key] = [float(values[0]), float(values[1])]
        self.normalization_parameters = normalization_parameters_loaded

    def debug(self, x, normalized=True):
        # if normalized -> show denormalization and then back normalization

        x = x.numpy()

        print("debug for normalizer")
        print("[x]")

        explore_idx = 0
        ex = x[explore_idx].flatten()
        print("before", np.nanmin(ex), np.nanmax(ex), np.nanmean(ex))

        if normalized:
            print("was normalized, will denormalize and go back")
            x_norm = np.copy(x)
            x_denorm = self.denormalize_x(np.copy(x))
            
            ex = x_denorm[explore_idx].flatten()
            print("denormalized", np.nanmin(ex), np.nanmax(ex), np.nanmean(ex))

            x_backnorm = self.normalize_x(np.copy(x_denorm))
            ex = x_backnorm[explore_idx].flatten()
            print("normalized again", np.nanmin(ex), np.nanmax(ex), np.nanmean(ex))

        else:
            print("wasn't normalized, will normalize and go back to denormalized")

            x_denorm = np.copy(x)
            x_norm = self.normalize_x(np.copy(x))
            
            ex = x_norm[explore_idx].flatten()
            print("normalized", np.nanmin(ex), np.nanmax(ex), np.nanmean(ex))

            x_backdenorm = self.denormalize_x(np.copy(x_norm))
            ex = x_backdenorm[explore_idx].flatten()
            print("denormalized again", np.nanmin(ex), np.nanmax(ex), np.nanmean(ex))

        print("")
        
# Torch Dataset:

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TileDataset(Dataset):
    # Main class that holds a dataset with smaller tiles originally extracted from larger geotiff files
    # Minimal impact on memory, loads actual data of x only in __getitem__ (when loading a batch of data)
    # Additional functionality:
    # - Load useful statistics for its tiles (such as the number of plume pixels in the label)
    # - Filter itself using those statistics (example: keep valid tiles, or only tiles with plumes, etc...)
    # - Spawn filtered tiles (to later make train / test / val splits ...)
    def __init__(self, tiles, settings_dataset, data_normalizer:DataNormalizer=None):
        self.tiles = tiles
        self.settings_dataset = settings_dataset
        self.data_normalizer = data_normalizer

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        # Load only when needed:
        # A.)
        x = load_tile_idx(tile, self.settings_dataset)

        if self.data_normalizer is not None:
            x = self.data_normalizer.normalize_x(x)

        x = torch.from_numpy(x)
        return x
    
# Pytorch Lightning Module

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

    def __init__(self, settings, data_normalizer:DataNormalizer):
        super().__init__()
        self.settings = settings
        self.data_normalizer = data_normalizer

        self.batch_size = self.settings["dataloader"]["batch_size"]
        self.num_workers = self.settings["dataloader"]["num_workers"]

        self.train_ratio = self.settings["dataloader"]["train_ratio"]
        self.validation_ratio = self.settings["dataloader"]["validation_ratio"]
        self.test_ratio = self.settings["dataloader"]["test_ratio"]

        self.setup_finished = False

    def prepare_data(self):
        # Could contain data download and unpacking...
        pass

    def setup(self, stage=None):
        if self.setup_finished:
            return True # to prevent double setup
        
        tiles = load_all_tile_indices_from_folder(self.settings["dataset"])
        print("Altogether we have", len(tiles), "tiles.")
        
        tiles_train, tiles_rest = train_test_split(tiles, test_size=1 - self.train_ratio)
        tiles_val, tiles_test = train_test_split(tiles_rest, test_size=self.test_ratio/(self.test_ratio + self.validation_ratio)) 

        print("train, test, val:",len(tiles_train), len(tiles_test), len(tiles_val))

        self.train_dataset = TileDataset(tiles_train, settings["dataset"], self.data_normalizer)
        self.test_dataset = TileDataset(tiles_test, settings["dataset"], self.data_normalizer)
        self.val_dataset = TileDataset(tiles_val, settings["dataset"], self.data_normalizer)

        self.setup_finished = True


    def train_dataloader(self):
        """Initializes and returns the training dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, num_workers=None):
        """Initializes and returns the validation dataloader"""
        num_workers = num_workers or self.num_workers
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=num_workers)

    def test_dataloader(self, num_workers=None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=num_workers)

    def debug(self):
        print("Dataset debug:")
        print("train", self.train_dataset.__len__(), " tiles")
        print("val", self.val_dataset.__len__(), " tiles")
        print("test", self.test_dataset.__len__(), " tiles")
        print("Sample data:")

        if self.debug_normalizer_for_i_samples > 0:
            for i in range(self.debug_normalizer_for_i_samples):
                img = self.train_dataset[i]
                print("x shapes:", img.size())

                self.data_normalizer.debug(img, normalized=True)