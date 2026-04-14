"""
GDAL utilities for safe resource management.

This module provides context managers and utilities for safely working with GDAL
datasets to prevent resource leaks and ensure proper cleanup.
"""

import os
import numpy as np
from typing import Optional, Any, List, Dict, Tuple
from contextlib import contextmanager
from numpy.typing import NDArray

try:
    from osgeo import gdal
    from osgeo import osr

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    # Don't raise error here to allow tests to run


class GDALDatasetManager:
    """
    Context manager for safely managing GDAL datasets.

    This class ensures that GDAL datasets are properly closed even if exceptions
    occur during processing.
    """

    def __init__(self, file_path: str, access_mode: int = 0):  # 0 = gdal.GA_ReadOnly
        """
        Initialize the GDAL dataset manager.

        Parameters
        ----------
        file_path : str
            Path to the GDAL dataset file
        access_mode : int, optional
            GDAL access mode (default: gdal.GA_ReadOnly)
        """
        self.file_path = file_path
        self.access_mode = access_mode
        self.dataset: Optional[gdal.Dataset] = None

    def __enter__(self) -> gdal.Dataset:
        """
        Open the GDAL dataset and return it.

        Returns
        -------
        gdal.Dataset
            The opened GDAL dataset

        Raises
        ------
        RuntimeError
            If the dataset cannot be opened
        """
        if not GDAL_AVAILABLE:
            raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
        self.dataset = gdal.Open(self.file_path, self.access_mode)
        if self.dataset is None:
            raise RuntimeError(f"Failed to open GDAL dataset: {self.file_path}")
        return self.dataset

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Close the GDAL dataset.

        Parameters
        ----------
        exc_type : Any
            Exception type if an exception occurred
        exc_val : Any
            Exception value if an exception occurred
        exc_tb : Any
            Exception traceback if an exception occurred
        """
        if self.dataset is not None:
            self.dataset = (
                None  # GDAL datasets are automatically closed when dereferenced
            )


@contextmanager
def open_gdal_dataset(
    file_path: str, access_mode: int = 0
) -> Any:  # 0 = gdal.GA_ReadOnly
    """
    Context manager for opening GDAL datasets.

    This is a convenience function that wraps GDALDatasetManager.

    Parameters
    ----------
    file_path : str
        Path to the GDAL dataset file
    access_mode : int, optional
        GDAL access mode (default: gdal.GA_ReadOnly)

    Yields
    ------
    gdal.Dataset
        The opened GDAL dataset
    """
    with GDALDatasetManager(file_path, access_mode) as dataset:
        yield dataset


def read_gdal_band_safe(file_path: str, band_number: int = 1) -> Any:
    """
    Safely read a band from a GDAL dataset.

    Parameters
    ----------
    file_path : str
        Path to the GDAL dataset file
    band_number : int, optional
        Band number to read (default: 1)

    Returns
    -------
    Any
        The band data as a numpy array

    Raises
    ------
    RuntimeError
        If the dataset or band cannot be read
    """
    with open_gdal_dataset(file_path) as dataset:
        band = dataset.GetRasterBand(band_number)
        if band is None:
            raise RuntimeError(f"Failed to get band {band_number} from {file_path}")

        data = band.ReadAsArray()
        if data is None:
            raise RuntimeError(f"Failed to read data from band {band_number}")

        return data


def get_gdal_metadata_safe(file_path: str) -> dict:
    """
    Safely get metadata from a GDAL dataset.

    Parameters
    ----------
    file_path : str
        Path to the GDAL dataset file

    Returns
    -------
    dict
        Dictionary containing dataset metadata
    """
    with open_gdal_dataset(file_path) as dataset:
        metadata = {}

        # Get basic metadata
        metadata["driver"] = dataset.GetDriver().ShortName
        metadata["size"] = (dataset.RasterXSize, dataset.RasterYSize)
        metadata["bands"] = dataset.RasterCount

        # Get projection
        projection = dataset.GetProjection()
        if projection:
            metadata["projection"] = projection

        # Get geotransform
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            metadata["geotransform"] = geotransform

        # Get band-specific metadata
        band_metadata = {}
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            band_info = {
                "data_type": gdal.GetDataTypeName(band.DataType) if GDAL_AVAILABLE else "Unknown",
                "no_data_value": band.GetNoDataValue(),
                "scale": band.GetScale(),
                "offset": band.GetOffset(),
            }
            band_metadata[f"band_{i}"] = band_info

        metadata["bands_info"] = band_metadata

        return metadata


def read_raster_band(file_path: str, band_number: int = 1) -> NDArray:
    """
    Read a single band from a raster file with error handling.

    Parameters
    ----------
    file_path : str
        Path to the raster file
    band_number : int, optional
        Band number to read (default: 1)

    Returns
    -------
    NDArray
        Array containing the band data

    Raises
    ------
    RuntimeError
        If the file cannot be opened or band cannot be read
    """
    return read_gdal_band_safe(file_path, band_number)


def read_raster_bands(
    file_path: str, band_numbers: Optional[List[int]] = None
) -> List[NDArray]:
    """
    Read multiple bands from a raster file.

    Parameters
    ----------
    file_path : str
        Path to the raster file
    band_numbers : list of int, optional
        List of band numbers to read (default: all bands)

    Returns
    -------
    list of NDArray
        List of arrays containing the band data

    Raises
    ------
    RuntimeError
        If the file cannot be opened or bands cannot be read
    """
    with open_gdal_dataset(file_path) as dataset:
        if band_numbers is None:
            band_numbers = list(range(1, dataset.RasterCount + 1))

        bands = []
        for band_num in band_numbers:
            band = dataset.GetRasterBand(band_num)
            if band is None:
                raise RuntimeError(f"Failed to get band {band_num} from {file_path}")

            data = band.ReadAsArray()
            if data is None:
                raise RuntimeError(f"Failed to read data from band {band_num}")

            bands.append(data)

        return bands


def write_raster(
    data: NDArray,
    output_path: str,
    source_path: Optional[str] = None,
    geotransform: Optional[Tuple[float, float, float, float, float, float]] = None,
    projection: Optional[str] = None,
    data_type: int = 6,
) -> None:  # 6 = gdal.GDT_Float32
    """
    Write array data to a raster file.

    Parameters
    ----------
    data : NDArray
        Array data to write
    output_path : str
        Path for the output raster file
    source_path : str, optional
        Path to source raster file to copy georeferencing from
    geotransform : tuple, optional
        GDAL geotransform parameters
    projection : str, optional
        Projection string
    data_type : int, optional
        GDAL data type (default: GDT_Float32)

    Raises
    ------
    RuntimeError
        If the raster cannot be created or written
    """
    # If source_path is provided, extract georeferencing from it
    if source_path and os.path.exists(source_path):
        with open_gdal_dataset(source_path) as source_dataset:
            if geotransform is None:
                geotransform = source_dataset.GetGeoTransform()
            if projection is None:
                projection = source_dataset.GetProjection()

    if not GDAL_AVAILABLE:
        raise ImportError("GDAL library is required but not available. Install with: pip install gdal")
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GTiff driver not available")

    if len(data.shape) == 2:
        bands = 1
        height, width = data.shape
    else:
        bands, height, width = data.shape

    dataset = driver.Create(output_path, width, height, bands, data_type)
    if dataset is None:
        raise RuntimeError(f"Failed to create raster file: {output_path}")

    if geotransform:
        dataset.SetGeoTransform(geotransform)
    if projection:
        dataset.SetProjection(projection)

    if len(data.shape) == 2:
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
    else:
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(data[i])

    dataset.FlushCache()
    dataset = None


def get_raster_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a raster file.

    Parameters
    ----------
    file_path : str
        Path to the raster file

    Returns
    -------
    dict
        Dictionary containing raster metadata
    """
    with open_gdal_dataset(file_path) as dataset:
        metadata = {
            "width": dataset.RasterXSize,
            "height": dataset.RasterYSize,
            "bands": dataset.RasterCount,
            "geotransform": dataset.GetGeoTransform(),
            "projection": dataset.GetProjection(),
            "driver": dataset.GetDriver().ShortName,
        }

        # Add band-specific metadata
        band_metadata = {}
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            band_metadata[f"band_{i}"] = {
                "data_type": gdal.GetDataTypeName(band.DataType) if GDAL_AVAILABLE else "Unknown",
                "no_data_value": band.GetNoDataValue(),
                "minimum": band.GetMinimum(),
                "maximum": band.GetMaximum(),
                "statistics": band.GetStatistics(True, True),
            }

        metadata["bands_metadata"] = band_metadata
        return metadata


def create_raster_copy(
    source_path: str, output_path: str, data: Optional[NDArray] = None
) -> None:
    """
    Create a copy of a raster file with the same metadata.

    Parameters
    ----------
    source_path : str
        Path to the source raster file
    output_path : str
        Path for the output raster file
    data : NDArray, optional
        New data to write (if None, copies original data)

    Raises
    ------
    RuntimeError
        If the source file cannot be read or output cannot be written
    """
    with open_gdal_dataset(source_path) as source_dataset:
        metadata = get_raster_metadata(source_path)

        if data is None:
            # Copy original data
            if source_dataset.RasterCount == 1:
                data = read_raster_band(source_path)
            else:
                data = np.stack(read_raster_bands(source_path))

        write_raster(
            data, output_path, metadata["geotransform"], metadata["projection"]
        )


__all__ = [
    "GDALDatasetManager",
    "open_gdal_dataset",
    "read_gdal_band_safe",
    "get_gdal_metadata_safe",
    "read_raster_band",
    "read_raster_bands",
    "write_raster",
    "get_raster_metadata",
    "create_raster_copy",
]
