"""
Orthophoto processing module for creating orthophotos using OpenDroneMap or GDAL.

This module provides functionality to create orthophotos from processed TIFF files
using either OpenDroneMap (preferred) or GDAL as a fallback.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import logging
import warnings
from typing import Dict, Any, List, Optional

from ..core.config import Config, get_config
from ..utils.logger import setup_logger
from ..utils.validators import validate_file_path
from ..utils.exceptions import ValidationError
from ..utils.gdal_utils import open_gdal_dataset
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import gc

# Try to import cv2 with graceful error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Silence harmless TIFF tag warnings
warnings.filterwarnings("ignore", message=".*unknown.*tag.*")
os.environ["GDAL_PAM_ENABLED"] = "NO"


class OrthophotoProcessor:
    """
    Processor for creating orthophotos using OpenDroneMap or GDAL.
    
    This class handles the creation of orthophotos from processed TIFF files,
    with support for both OpenDroneMap (preferred) and GDAL (fallback) methods.
    """

    def __init__(self, config_instance: Optional[Config] = None) -> None:
        """
        Initialize the orthophoto processor.

        Args:
            config_instance: Optional configuration instance for dependency injection
        """
        self.config = get_config(config_instance)
        self.logger = setup_logger("OrthophotoProcessor")  # Revert to default level
        self.odm_path = self._find_odm_path()
        
        # Load orthophoto output configuration with defaults
        self.orthophoto_config = {
            "compression": self.config.get("processing.orthophoto.output.compression", "LZW"),
            "predictor": self.config.get("processing.orthophoto.output.predictor", "auto"),
            "tiled": self.config.get("processing.orthophoto.output.tiled", True),
            "block_size": self.config.get("processing.orthophoto.output.block_size", 512),
            "bigtiff": self.config.get("processing.orthophoto.output.bigtiff", "IF_SAFER"),
            "target_dtype": self.config.get("processing.orthophoto.output.target_dtype", "uint8"),
            "build_overviews": self.config.get("processing.orthophoto.output.build_overviews", True),
            "overview_levels": self.config.get("processing.orthophoto.output.overview_levels", [2, 4, 8, 16]),
            "target_resolution": self.config.get("processing.orthophoto.output.target_resolution", "auto"),
        }
        
        # Load orthophoto blend configuration with defaults
        self.blend_config = {
            "enabled": self.config.get("processing.orthophoto.blend.enabled", True),
            "method": self.config.get("processing.orthophoto.blend.method", "feather"),
            "feather_distance_px": self.config.get("processing.orthophoto.blend.feather_distance_px", 0),
            "input_nodata": self.config.get("processing.orthophoto.blend.input_nodata", 0),
            "edge_erosion_px": self.config.get("processing.orthophoto.blend.edge_erosion_px", 2),
        }
        
        # Load stitching method configuration with default
        self.stitching_method = self.config.get("processing.orthophoto.stitching_method", "gdal")
        
        # Load OpenCV configuration with defaults
        self.opencv_config = {
            "detector": self.config.get("processing.orthophoto.opencv.detector", "auto"),
            "ratio_test": self.config.get("processing.orthophoto.opencv.ratio_test", 0.75),
            "ransac_reproj_threshold": self.config.get("processing.orthophoto.opencv.ransac_reproj_threshold", 5.0),
            "min_matches": self.config.get("processing.orthophoto.opencv.min_matches", 10),
            "try_use_gpu": self.config.get("processing.orthophoto.opencv.try_use_gpu", False),
            "max_feature_dim": self.config.get("processing.orthophoto.opencv.max_feature_dim", 4000),
        }

    def _is_docker_available(self) -> bool:
        """
        Check if Docker is available and accessible.
        
        Returns:
            True if Docker is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _is_odm_docker_image_available(self) -> bool:
        """
        Check if the OpenDroneMap Docker image is available.
        
        Returns:
            True if ODM Docker image is available, False otherwise
        """
        try:
            # First check if docker info works
            info_result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if info_result.returncode != 0:
                return False
            
            # Then check if the image is available locally
            result = subprocess.run(
                ["docker", "images", "opendronemap/odm", "--format", "{{.Repository}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and "opendronemap/odm" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_odm_available(self, tiff_paths: List[str]) -> None:
        """
        Check if OpenDroneMap is available for the chosen execution method.
        
        Args:
            tiff_paths: List of TIFF file paths to be processed
            
        Raises:
            RuntimeError: If ODM is not available or if there are insufficient images
        """
        # Check minimum image count for ODM (needs at least 3-5 overlapping images)
        if len(tiff_paths) < 3:
            raise RuntimeError(
                f"OpenDroneMap requires at least 3 overlapping images to create an orthophoto, "
                f"but only {len(tiff_paths)} image(s) were provided. "
                f"Consider using a different stitching method like 'gdal' or 'opencv' for small image counts."
            )
        
        use_docker = self._should_use_docker()
        
        if use_docker:
            # Check Docker availability
            if not self._is_docker_available():
                raise RuntimeError(
                    "Docker is not available or not running. Please ensure Docker Desktop is installed "
                    "and running on your system. See https://docs.docker.com/get-docker/ for installation instructions."
                )
            
            # Check ODM Docker image availability
            if not self._is_odm_docker_image_available():
                raise RuntimeError(
                    "OpenDroneMap Docker image (opendronemap/odm) is not available. "
                    "Please pull the image using: docker pull opendronemap/odm"
                )
        else:
            # Check native ODM availability
            if not self.odm_path:
                raise RuntimeError(
                    "OpenDroneMap is not installed or not found in standard locations. "
                    "Please install ODM or ensure Docker is available with the opendronemap/odm image pulled. "
                    "See https://github.com/OpenDroneMap/ODM for installation instructions."
                )
            
            # Check if required Python is available for native ODM
            run_script = os.path.join(self.odm_path, "run.py")
            if not os.path.exists(run_script):
                raise RuntimeError(
                    f"OpenDroneMap installation appears to be incomplete. "
                    f"Expected run.py at {run_script} but file was not found."
                )

    def _should_use_docker(self) -> bool:
        """
        Determine if Docker should be used for ODM execution.
        
        Returns:
            True if Docker should be used, False otherwise
        """
        # Check if Docker is available and ODM image is present
        return self._is_docker_available() and self._is_odm_docker_image_available()

    def _find_odm_path(self) -> Optional[str]:
        """
        Find the path to OpenDroneMap installation.

        Returns:
            Path to OpenDroneMap or None if not found
        """
        # Check common OpenDroneMap installation paths
        possible_paths = [
            "/opt/opendronemap",
            "/usr/local/opendronemap",
            os.path.expanduser("~/OpenDroneMap"),
            "./OpenDroneMap",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                run_script = os.path.join(path, "run.sh")
                if os.path.exists(run_script):
                    self.logger.info(f"OpenDroneMap found: {path}")
                    return path

        self.logger.warning(
            "OpenDroneMap not found. Using alternative method."
        )
        return None

    def create_orthophoto(self, processed_data: Dict[str, Any], output_dir: str) -> str:
        """
        Create an orthophoto from processed data.

        Args:
            processed_data: Results from preprocessing containing TIFF paths and metadata
            output_dir: Directory to save the orthophoto

        Returns:
            Path to the created orthophoto

        Raises:
            ValidationError: If input validation fails
            RuntimeError: If orthophoto creation fails
        """
        try:
            # Validate input parameters
            if not processed_data or not isinstance(processed_data, dict):
                raise ValidationError(
                    "processed_data must be a non-empty dictionary",
                    details={"processed_data_type": type(processed_data)},
                )

            validate_file_path(output_dir, must_exist=False, must_be_writable=True)

            # Validate required keys in processed_data
            required_keys = ["tiff_paths", "metadata"]
            for key in required_keys:
                if key not in processed_data:
                    raise ValidationError(
                        f"Missing required key in processed_data: {key}",
                        details={
                            "required_key": key,
                            "available_keys": list(processed_data.keys()),
                        },
                    )

            # Validate TIFF paths
            tiff_paths = processed_data.get("tiff_paths", [])
            if not isinstance(tiff_paths, list) or len(tiff_paths) == 0:
                raise ValidationError(
                    "tiff_paths must be a non-empty list",
                    details={
                        "tiff_paths_type": type(tiff_paths),
                        "tiff_paths_length": (
                            len(tiff_paths) if isinstance(tiff_paths, list) else None
                        ),
                    },
                )

            for tiff_path in tiff_paths:
                validate_file_path(
                    tiff_path,
                    must_exist=True,
                    must_be_readable=True,
                    allowed_extensions=[".tif", ".tiff"],
                )

            self.logger.info(f"[{len(tiff_paths)} files] Starting orthophoto creation")

            # Create orthophoto using the selected stitching method
            orthophoto_path = self._dispatch_stitching(tiff_paths, output_dir, processed_data)

            self.logger.info(f"[{len(tiff_paths)} files] Orthophoto created: {orthophoto_path}")
            return orthophoto_path

        except Exception as e:
            self.logger.exception(f"Error creating orthophoto: {e}")
            raise

    def _dispatch_stitching(self, tiff_paths: List[str], output_dir: str, processed_data: Dict[str, Any]) -> str:
        """
        Dispatch to the appropriate stitching method based on configuration.
        
        Args:
            tiff_paths: List of paths to TIFF files
            output_dir: Directory to save results
            processed_data: Processed data containing metadata
            
        Returns:
            Path to the created orthophoto
            
        Raises:
            ValueError: If an unknown stitching method is specified
        """
        self.logger.info(f"Using stitching method: {self.stitching_method}")
        
        # Dictionary mapping method names to their corresponding functions
        stitching_methods = {
            "gdal": self._create_with_gdal,
            "odm": lambda paths, out_dir: self._create_with_odm(paths, out_dir, processed_data),
            "opencv": self._create_with_opencv,
        }
        
        # Get the appropriate method
        if self.stitching_method in stitching_methods:
            method = stitching_methods[self.stitching_method]
            # All methods have the same signature now
            return method(tiff_paths, output_dir)
        else:
            valid_methods = ", ".join(stitching_methods.keys())
            raise ValueError(f"Unknown stitching method: {self.stitching_method}. Valid options are: {valid_methods}")

    def _create_with_odm(self, tiff_paths: List[str], output_dir: str, processed_data: Dict[str, Any] = None) -> str:
        """
        Create orthophoto using OpenDroneMap.

        Args:
            tiff_paths: List of paths to TIFF files
            output_dir: Directory to save results
            processed_data: Optional processed data containing metadata

        Returns:
            Path to the created orthophoto

        Raises:
            RuntimeError: If OpenDroneMap execution fails
            FileNotFoundError: If results are not found
        """
        # Pre-flight checks - now using the dedicated availability check
        self._check_odm_available(tiff_paths)
        use_docker = self._should_use_docker()

        self.logger.info(f"[{len(tiff_paths)} files] Creating orthophoto using OpenDroneMap "
                         f"({'Docker' if use_docker else 'native'})...")

        # Create stable output path
        output_path = os.path.join(output_dir, "orthophoto.tif")
        
        try:
            # Create temporary directory for ODM
            temp_dir = tempfile.mkdtemp()
            try:
                project_dir = os.path.join(temp_dir, "project")
                os.makedirs(project_dir, exist_ok=True)

                # Copy files to ODM structure
                images_dir = os.path.join(project_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

                for tiff_path in tiff_paths:
                    dest_path = os.path.join(images_dir, os.path.basename(tiff_path))
                    self._copy_file(tiff_path, dest_path)

                # Create GPS file if necessary
                gps_file = self._create_gps_file(processed_data, project_dir)

                if use_docker:
                    # Run ODM via Docker
                    # Handle Windows paths for Docker volume mounts
                    import platform
                    if platform.system() == "Windows":
                        # On Windows, ensure paths are in the correct format for Docker
                        # Docker Desktop on Windows accepts both forward slashes and backslashes
                        # but we'll use forward slashes for consistency
                        docker_project_dir = project_dir.replace("\\", "/")
                    else:
                        docker_project_dir = project_dir
                    
                    cmd = [
                        "docker", "run", "--rm",  # Remove -it flags for better compatibility
                        "-v", f"{docker_project_dir}:/project",
                        "opendronemap/odm",
                        "--orthophoto-resolution",
                        str(self.config.get("processing.orthophoto_resolution", 0.05)),
                        "--dem-resolution",
                        str(self.config.get("processing.dem_resolution", 0.1)),
                        "--feature-quality",
                        self.config.get("processing.feature_quality", "high"),
                        "--matcher-neighbors",
                        str(self.config.get("processing.matcher_neighbors", 8)),
                        "--use-exif",
                        "false",
                    ]

                    if gps_file:
                        # Map the GPS file into the container
                        gps_filename = os.path.basename(gps_file)
                        cmd.extend(["--gps-file", f"/project/{gps_filename}"])

                    self.logger.info(f"[{len(tiff_paths)} files] Running OpenDroneMap via Docker...")
                    self.logger.info(f"Mounting host path: {docker_project_dir} to container path: /project")
                    self.logger.info(f"Executing command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.config.get("processing.odm_timeout", 7200),  # 2 hours default
                    )
                else:
                    # Run ODM natively
                    cmd = [
                        os.path.join(self.odm_path, "run.sh"),
                        "--project-path",
                        project_dir,
                        "--orthophoto-resolution",
                        str(self.config.get("processing.orthophoto_resolution", 0.05)),
                        "--dem-resolution",
                        str(self.config.get("processing.dem_resolution", 0.1)),
                        "--feature-quality",
                        self.config.get("processing.feature_quality", "high"),
                        "--matcher-neighbors",
                        str(self.config.get("processing.matcher_neighbors", 8)),
                        "--use-exif",
                        "false",
                    ]

                    if gps_file:
                        cmd.extend(["--gps-file", gps_file])

                    self.logger.info(f"[{len(tiff_paths)} files] Running OpenDroneMap natively...")
                    self.logger.info(f"Executing command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        cwd=self.odm_path,
                        capture_output=True,
                        text=True,
                        timeout=self.config.get("processing.odm_timeout", 7200),  # 2 hours default
                    )

                if result.returncode != 0:
                    # Include both stdout and stderr in the error message
                    error_msg = f"OpenDroneMap failed with return code {result.returncode}"
                    if result.stdout:
                        # Include last 50 lines of stdout for context
                        stdout_lines = result.stdout.strip().split('\n')
                        last_lines = stdout_lines[-50:] if len(stdout_lines) > 50 else stdout_lines
                        error_msg += f"\nLast {len(last_lines)} line(s) of stdout:\n" + '\n'.join(last_lines)
                    if result.stderr:
                        error_msg += f"\nstderr:\n{result.stderr}"
                    
                    self.logger.error(f"ODM failed: {error_msg}")
                    raise RuntimeError(error_msg)

                # Copy results
                odm_results_path = os.path.join(project_dir, "odm_orthophoto", "odm_orthophoto.tif")
                if os.path.exists(odm_results_path):
                    # Copy ODM output to temporary location for optimization
                    temp_output_path = os.path.join(temp_dir, "odm_output.tif")
                    self._copy_file(odm_results_path, temp_output_path)
                    
                    # Post-process ODM output via optimize_orthophoto
                    self.logger.info("Optimizing ODM output...")
                    optimized_path = self.optimize_orthophoto(temp_output_path, output_path)
                    
                    # Ensure the final output path is what the caller expects
                    if optimized_path != output_path:
                        self._copy_file(optimized_path, output_path)
                        os.remove(optimized_path)
                    
                    self.logger.info(f"Orthophoto created successfully: {output_path}")
                    return output_path
                else:
                    raise FileNotFoundError("ODM results not found")

            finally:
                # Clean up temporary directory
                try:
                    self.logger.debug(f"About to clean up temporary directory: {temp_dir}")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up temporary directory {temp_dir}: {cleanup_error}")

        except subprocess.TimeoutExpired:
            self.logger.error("OpenDroneMap execution timeout exceeded")
            raise RuntimeError("OpenDroneMap timeout")
        except Exception as e:
            self.logger.error(f"Error working with OpenDroneMap: {e}")
            raise

    def _convert_to_uint8(self, file_path: str) -> None:
        """
        Convert orthophoto to uint8 data type with proper scaling.
        
        Args:
            file_path: Path to the orthophoto file
        """
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            
            # Open dataset
            with open_gdal_dataset(file_path, gdal.GA_Update) as src_ds:
                # Check if already uint8
                band = src_ds.GetRasterBand(1)
                data_type = band.DataType
                
                if data_type == gdal.GDT_Byte:
                    # Already uint8, no conversion needed
                    return
                
                # Read data from all bands
                band_count = src_ds.RasterCount
                bands_data = []
                
                for i in range(1, band_count + 1):
                    band = src_ds.GetRasterBand(i)
                    data = band.ReadAsArray()
                    bands_data.append(data)
                
                # Get georeferencing info
                geotransform = src_ds.GetGeoTransform()
                projection = src_ds.GetProjection()
                
                # Get nodata value while still in the context
                nodata_value = band.GetNoDataValue()
                
            # Calculate min/max for scaling (use percentiles for robustness)
            all_data = np.stack(bands_data)
            min_val = np.percentile(all_data[all_data != nodata_value], 1) if nodata_value is not None else np.min(all_data)
            max_val = np.percentile(all_data[all_data != nodata_value], 99) if nodata_value is not None else np.max(all_data)
            
            # Scale to 0-255 range
            if max_val > min_val:
                scaled_bands = []
                for data in bands_data:
                    # Handle nodata values
                    nodata_mask = (data == nodata_value) if nodata_value is not None else np.zeros_like(data, dtype=bool)
                    
                    # Scale data
                    scaled = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    
                    # Apply nodata mask
                    scaled[nodata_mask] = 0
                    
                    scaled_bands.append(scaled)
            else:
                # If all values are the same, set to 128 (middle gray)
                scaled_bands = [np.full_like(data, 128, dtype=np.uint8) for data in bands_data]
            
            # Write back to file
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(file_path + '_temp', src_ds.RasterXSize, src_ds.RasterYSize,
                                  band_count, gdal.GDT_Byte)
            
            # Copy georeferencing
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)
            
            # Write scaled data
            for i, scaled_data in enumerate(scaled_bands):
                dst_band = dst_ds.GetRasterBand(i + 1)
                dst_band.WriteArray(scaled_data)
                dst_band.SetNoDataValue(0)
            
            # Close datasets
            dst_ds = None
            src_ds = None
            
            # Replace original file
            os.replace(file_path + '_temp', file_path)
            
        except Exception as e:
            self.logger.warning(f"Could not convert to uint8: {e}")
            # Clean up temp file if it exists
            if os.path.exists(file_path + '_temp'):
                os.remove(file_path + '_temp')

    def _build_overviews(self, file_path: str) -> None:
        """
        Build internal overviews (pyramids) for the orthophoto.
        
        Args:
            file_path: Path to the orthophoto file
        """
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            
            # Open dataset in update mode
            with open_gdal_dataset(file_path, gdal.GA_Update) as ds:
                # Build overviews with LZW compression
                ds.BuildOverviews(
                    resampling="average",
                    overviewlist=self.orthophoto_config["overview_levels"],
                    options=["COMPRESS_OVERVIEW=LZW"]
                )
                
        except Exception as e:
            self.logger.warning(f"Could not build overviews: {e}")

    def _warp_to_common_grid(self, tiff_paths: List[str], temp_dir: str, common_bounds: tuple,
                           pixel_size_x: float, pixel_size_y: float, srs) -> List[str]:
        """
        Warp all input images to a common grid.
        
        Args:
            tiff_paths: List of paths to TIFF files
            temp_dir: Temporary directory for output files
            common_bounds: (min_x, min_y, max_x, max_y) bounds for output
            pixel_size_x: Pixel size in X direction (from first image, used as fallback)
            pixel_size_y: Pixel size in Y direction (from first image, used as fallback)
            srs: Spatial reference system
            
        Returns:
            List of paths to warped TIFF files
        """
        from osgeo import gdal
        gdal.UseExceptions()
        
        # Determine target resolution
        target_resolution = self.orthophoto_config["target_resolution"]
        
        if target_resolution == "auto":
            # Compute finest resolution across all inputs
            xRes, yRes = self._compute_target_resolution(tiff_paths)
            resolution_source = "auto"
        elif isinstance(target_resolution, (int, float)):
            # Use single float value for both x and y
            xRes = float(target_resolution)
            yRes = float(target_resolution)
            resolution_source = "config"
        elif isinstance(target_resolution, list) and len(target_resolution) == 2:
            # Use explicit [x, y] values
            xRes = float(target_resolution[0])
            yRes = float(target_resolution[1])
            resolution_source = "config"
        else:
            # Fallback to input pixel sizes
            xRes = pixel_size_x
            yRes = pixel_size_y
            resolution_source = "fallback"
        
        # Sanity check for valid resolution
        if not (np.isfinite(xRes) and np.isfinite(yRes) and xRes > 0 and yRes > 0):
            raise RuntimeError("Could not determine target pixel resolution from inputs; please set processing.orthophoto.output.target_resolution explicitly in config.yaml")
        
        self.logger.info(f"Target resolution: xRes={xRes}, yRes={yRes} (source='{resolution_source}')")
        
        # Calculate width and height based on target resolution
        width = int((common_bounds[2] - common_bounds[0]) / xRes)
        height = int((common_bounds[3] - common_bounds[1]) / yRes)
        
        # Get nodata configuration
        input_nodata = self.blend_config["input_nodata"]
        
        # Initialize the list to store warped file paths
        warped_paths = []
        
        for i, tiff_path in enumerate(tiff_paths):
            warped_path = os.path.join(temp_dir, f"warped_{i}.tif")
            self.logger.info(f"Warping image {i+1}/{len(tiff_paths)}: {os.path.basename(tiff_path)}")
            
            # Open source dataset to check for alpha band and nodata
            self.logger.debug(f"About to open GDAL dataset: {tiff_path}")
            with open_gdal_dataset(tiff_path) as src_ds:
                self.logger.debug(f"Opened GDAL dataset: {tiff_path}")
                band_count = src_ds.RasterCount
                has_alpha = band_count in [2, 4]  # Grayscale+alpha or RGB+alpha
                
                # Check if file has explicit nodata value set
                src_nodata = None
                if band_count > 0:
                    band = src_ds.GetRasterBand(1)
                    src_nodata = band.GetNoDataValue()
            
            # Determine nodata handling options
            warp_options_dict = {
                "format": "GTiff",
                "outputBounds": common_bounds,
                "xRes": xRes,
                "yRes": yRes,
                "resampleAlg": "bilinear",
                "srcSRS": srs,
                "dstSRS": srs,
                "targetAlignedPixels": True,
            }
            
            # Handle nodata and alpha bands
            if has_alpha and input_nodata == "alpha":
                # Use alpha band for transparency
                warp_options_dict["dstAlpha"] = True
            elif input_nodata is not None and input_nodata != "alpha":
                # Use explicit nodata values
                if src_nodata is not None:
                    # Respect file's existing nodata value
                    warp_options_dict["srcNodata"] = src_nodata
                else:
                    # Use configured nodata value
                    warp_options_dict["srcNodata"] = input_nodata
                warp_options_dict["dstNodata"] = input_nodata
            # If input_nodata is None, don't override nodata values
            
            # Create warp options
            warp_options = gdal.WarpOptions(**warp_options_dict)
            
            # Perform the warp operation
            # Capture and immediately release the returned dataset to prevent file locks on Windows
            self.logger.debug(f"About to call gdal.Warp with source: {tiff_path}, destination: {warped_path}")
            warped_ds = gdal.Warp(warped_path, tiff_path, options=warp_options)
            warped_ds = None  # Release dataset to prevent file lock on Windows
            self.logger.debug(f"gdal.Warp released dataset for: {warped_path}")
            warped_paths.append(warped_path)
            
        return warped_paths

    def _compute_distance_weights(self, warped_paths: List[str], temp_dir: str) -> List[str]:
        """
        Compute distance transform weights for each warped image.
        
        Args:
            warped_paths: List of paths to warped TIFF files
            temp_dir: Temporary directory for output files
            
        Returns:
            List of paths to weight files (.npy)
        """
        
    def _compute_valid_mask(self, data, nodata_value=None, has_alpha=False):
        """Compute a 2D boolean mask indicating valid pixels.
        
        A pixel is valid if:
        - It is not flagged as nodata by the warped raster's nodata value (if any), AND
        - It is not equal to the configured nodata color (default 0 across all bands), AND
        - If an alpha band exists in the warped output, alpha > 0.
        
        Args:
            data: numpy array with shape (height, width, bands)
            nodata_value: Nodata value to check against
            has_alpha: Whether the dataset has an alpha band
            
        Returns:
            2D boolean numpy array where True indicates valid pixels
        """
        # Assume data is already in (height, width, bands) format
        height, width, bands = data.shape
        
        # Initialize mask as all valid
        valid_mask = np.ones((height, width), dtype=bool)
        
        # Check first band for nodata values
        if nodata_value is not None:
            valid_mask &= (data[:, :, 0] != nodata_value)
        
        # Check for configured nodata color (default 0)
        input_nodata = self.blend_config["input_nodata"]
        if input_nodata is not None and input_nodata != "alpha":
            if bands >= 3:
                # For RGB images, check if all bands are nodata
                # Handle both black (0) and white (255) nodata values
                if isinstance(input_nodata, (int, float)):
                    # Check for the specific nodata value
                    nodata_condition = np.all(data == input_nodata, axis=2)
                    valid_mask &= ~nodata_condition
                    
                    # Also check for white (255) if input_nodata is 0 (black)
                    if input_nodata == 0:
                        white_condition = np.all(data == 255, axis=2)
                        valid_mask &= ~white_condition
                else:
                    # For other nodata values, check if all bands match
                    nodata_condition = np.all(data == input_nodata, axis=2)
                    valid_mask &= ~nodata_condition
            else:
                # For single band, check if band equals nodata
                valid_mask &= (data[:, :, 0] != input_nodata)
                
                # Also check for white (255) if input_nodata is 0 (black)
                if input_nodata == 0:
                    valid_mask &= (data[:, :, 0] != 255)
        
        # Check alpha channel if present
        if has_alpha and bands > 1:
            # Assume last band is alpha
            alpha_band = data[:, :, -1]
            valid_mask &= (alpha_band > 0)
            
        return valid_mask
        
    def _edt(self, mask: np.ndarray) -> np.ndarray:
        """Euclidean distance transform of a boolean mask, returning float32.
        
        Prefers cv2.distanceTransform (returns float32 directly, ~2-3x faster, no
        float64 intermediate). Falls back to scipy.ndimage.distance_transform_edt
        when OpenCV is unavailable.
        
        Args:
            mask: 2D bool array. Distance is computed FROM zero pixels (i.e. the
                  returned value at each True pixel is the distance to the nearest
                  False pixel).
        
        Returns:
            2D float32 array, same shape as mask.
        """
        if CV2_AVAILABLE:
            # cv2.distanceTransform expects uint8 source, with non-zero = foreground.
            src = mask.astype(np.uint8, copy=False)
            # DIST_L2 with DIST_MASK_PRECISE gives near-exact Euclidean; float32 output.
            return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Fallback: scipy returns float64 — cast immediately to keep RAM bounded.
        return distance_transform_edt(mask).astype(np.float32, copy=False)
        
    def _compute_target_resolution(self, tiff_paths: List[str]) -> tuple:
        """Compute target resolution as the finest (smallest) pixel size across all inputs.
        
        Args:
            tiff_paths: List of paths to TIFF files
            
        Returns:
            Tuple of (xRes, yRes) with the finest resolution
            
        Raises:
            RuntimeError: If resolution cannot be determined or is invalid
        """
        from osgeo import gdal
        gdal.UseExceptions()
        
        resolutions = []
        
        for tiff_path in tiff_paths:
            with open_gdal_dataset(tiff_path) as src_ds:
                geotransform = src_ds.GetGeoTransform()
                pixel_size_x = abs(geotransform[1])
                pixel_size_y = abs(geotransform[5])
                resolutions.append((pixel_size_x, pixel_size_y))
                
                self.logger.debug(f"Input {os.path.basename(tiff_path)} resolution: x={pixel_size_x}, y={pixel_size_y}")
        
        # Find finest (smallest) resolution
        if resolutions:
            xRes = min(res[0] for res in resolutions)
            yRes = min(res[1] for res in resolutions)
            
            # Log warning if resolutions vary significantly
            if len(set(resolutions)) > 1:
                self.logger.info("Input resolutions vary - using finest resolution:")
                for i, (path, res) in enumerate(zip(tiff_paths, resolutions)):
                    self.logger.info(f"  {os.path.basename(path)}: x={res[0]}, y={res[1]}")
                self.logger.info(f"  Target resolution: x={xRes}, y={yRes}")
            
            # Sanity check
            if not (np.isfinite(xRes) and np.isfinite(yRes) and xRes > 0 and yRes > 0):
                raise RuntimeError("Could not determine target pixel resolution from inputs; please set processing.orthophoto.output.target_resolution explicitly in config.yaml")
                
            return (xRes, yRes)
        else:
            raise RuntimeError("Could not determine target pixel resolution from inputs; please set processing.orthophoto.output.target_resolution explicitly in config.yaml")
        
    def _compute_distance_weights(self, warped_paths: List[str], temp_dir: str) -> List[str]:
        """
        Compute distance transform weights for each warped image.
        
        Args:
            warped_paths: List of paths to warped TIFF files
            temp_dir: Temporary directory for output files
            
        Returns:
            List of paths to weight files (.npy)
        """
        from osgeo import gdal
        gdal.UseExceptions()
        
        weight_paths = []
        mask_paths = []
        
        # First, compute a combined mask to identify overlap areas
        combined_mask = None
        
        # Pass 1: Compute masks, save to disk, and accumulate combined_mask
        for i, warped_path in enumerate(warped_paths):
            self.logger.debug(f"About to open GDAL dataset for distance weights: {warped_path}")
            with open_gdal_dataset(warped_path) as src_ds:
                self.logger.debug(f"Opened GDAL dataset for distance weights: {warped_path}")
                band_count = src_ds.RasterCount
                has_alpha = band_count in [2, 4]  # Grayscale+alpha or RGB+alpha
                band = src_ds.GetRasterBand(1)
                nodata = band.GetNoDataValue()
                
                # Read all bands data for nodata check
                if band_count >= 3:
                    # For multi-band images, read all bands to properly check nodata
                    bands_data = []
                    for b in range(1, band_count + 1):
                        band_data = src_ds.GetRasterBand(b).ReadAsArray()
                        bands_data.append(band_data)
                    # Stack bands to create (height, width, bands) array
                    data = np.stack(bands_data, axis=2)
                    # Free individual band arrays
                    for band_data in bands_data:
                        del band_data
                else:
                    # For single band, read as is
                    band1_data = band.ReadAsArray()
                    data = band1_data[..., np.newaxis] if band1_data.ndim == 2 else band1_data
                
                # Read alpha band if needed
                alpha_data = None
                if has_alpha and self.blend_config["input_nodata"] == "alpha":
                    alpha_band = src_ds.GetRasterBand(band_count)
                    alpha_data = alpha_band.ReadAsArray()
                
                # Build valid mask using the _compute_valid_mask function
                valid_mask = self._compute_valid_mask(data, nodata, has_alpha)
                
                # Apply erosion if configured
                edge_erosion_px = self.blend_config["edge_erosion_px"]
                if edge_erosion_px > 0:
                    structure = np.ones((3, 3), dtype=bool)
                    valid_mask = binary_erosion(valid_mask, structure=structure, iterations=edge_erosion_px)
                
                # Log valid pixel ratio
                total_pixels = valid_mask.size
                valid_pixels = np.sum(valid_mask)
                valid_ratio = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                filename = os.path.basename(warped_path)
                nodata_info = f"nodata={nodata}" if nodata is not None else "nodata=unset"
                erosion_info = f"erosion={edge_erosion_px}px"
                self.logger.info(f"{filename}: valid pixels = {valid_pixels} / {total_pixels} ({valid_ratio:.1f}%) after {nodata_info} and {erosion_info}")
                
                # Save mask to disk and update combined_mask
                mask_path = os.path.join(temp_dir, f"mask_{i}.npy")
                np.save(mask_path, valid_mask)
                mask_paths.append(mask_path)
                
                # Update combined mask
                if combined_mask is None:
                    combined_mask = valid_mask.astype(np.uint8)
                else:
                    combined_mask += valid_mask.astype(np.uint8)
                
                # Free memory
                del valid_mask, data
                if 'band1_data' in locals():
                    del band1_data
                if alpha_data is not None:
                    del alpha_data
        
        # Force garbage collection between passes
        import gc
        gc.collect()
        
        # Pass 2: Load masks via memory mapping and compute weights
        for i, warped_path in enumerate(warped_paths):
            self.logger.info(f"Computing weights for image {i+1}/{len(warped_paths)}")
            
            # Load mask via memory mapping
            mask_path = mask_paths[i]
            mask = np.load(mask_path, mmap_mode='r')
            
            # For pixels where exactly one image is valid, force weight = 1
            # For pixels where multiple images are valid, use distance transform
            single_image_area = (combined_mask == 1) & mask
            multi_image_area = (combined_mask > 1) & mask
            
            # Initialize weights
            weights = np.zeros_like(mask, dtype=np.float32)
            
            # Set weight = 1 for single image areas
            weights[single_image_area] = 1.0
            
            # For multi-image areas, compute distance transform
            if np.any(multi_image_area):
                # Compute distance to nearest invalid pixel within the multi-image area
                # Distance from each valid pixel to the nearest invalid pixel.
                # _edt() returns float32 directly when OpenCV is available, avoiding the
                # float64 intermediate that scipy.distance_transform_edt would allocate.
                dist_transform = self._edt(np.asarray(mask, dtype=bool))
                
                # Apply feather distance limit if configured
                if self.blend_config["feather_distance_px"] > 0:
                    dist_transform = np.minimum(dist_transform, self.blend_config["feather_distance_px"])
                
                # Normalize distances to weights (0-1 range)
                max_dist = np.max(dist_transform[multi_image_area])
                if max_dist > 0:
                    weights[multi_image_area] = dist_transform[multi_image_area] / max_dist
                else:
                    weights[multi_image_area] = 1.0
                
                # Free memory
                del dist_transform
            
            # Quantize float32 weights in [0, 1] to uint8 in [0, 255] to save 4x RAM/disk.
            # Blending precision is unaffected because _blend_tiles re-normalizes per tile.
            weights_u8 = np.clip(weights * 255.0 + 0.5, 0, 255).astype(np.uint8)
            weight_path = os.path.join(temp_dir, f"weights_{i}.npy")
            np.save(weight_path, weights_u8)
            weight_paths.append(weight_path)
            
            # Free memory
            del weights, weights_u8, mask, single_image_area, multi_image_area
        
        # Clean up temporary mask files (best-effort)
        for mask_path in mask_paths:
            try:
                os.remove(mask_path)
            except Exception as e:
                self.logger.debug(f"Could not remove temporary mask file {mask_path}: {e}")
        
        return weight_paths

    def _blend_tiles(self, warped_paths: List[str], weight_paths: List[str],
                   output_path: str, width: int, height: int,
                   pixel_size_x: float, pixel_size_y: float,
                   min_x: float, min_y: float, max_x: float, max_y: float, srs) -> None:
        """
        Blend warped images using precomputed weights in tiled fashion.
        
        Args:
            warped_paths: List of paths to warped TIFF files
            weight_paths: List of paths to weight files
            output_path: Path to output file
            width: Output width in pixels
            height: Output height in pixels
            pixel_size_x: Pixel size in X direction
            pixel_size_y: Pixel size in Y direction
            min_x, min_y, max_x, max_y: Output bounds
            srs: Spatial reference system
        """
        from osgeo import gdal
        from contextlib import ExitStack
        gdal.UseExceptions()
        
        # Tile size for processing (avoid loading entire images into memory)
        tile_size = 2048
        
        # Build creation options from config
        creation_options = []
        if self.orthophoto_config["compression"] != "NONE":
            creation_options.extend([
                f"COMPRESS={self.orthophoto_config['compression']}",
            ])
        
        if self.orthophoto_config["tiled"]:
            creation_options.extend([
                "TILED=YES",
                f"BLOCKXSIZE={self.orthophoto_config['block_size']}",
                f"BLOCKYSIZE={self.orthophoto_config['block_size']}",
            ])
        
        creation_options.append(f"BIGTIFF={self.orthophoto_config['bigtiff']}")
        
        # Determine predictor based on data type and config
        predictor = self.orthophoto_config["predictor"]
        if predictor == "auto":
            # For integer data, use PREDICTOR=2; for float, use PREDICTOR=3
            predictor = 2  # Default to 2 for integer data
        
        if predictor != 1:  # PREDICTOR=1 is no predictor
            creation_options.append(f"PREDICTOR={predictor}")
        
        # Create output dataset
        driver = gdal.GetDriverByName("GTiff")
        with open_gdal_dataset(warped_paths[0]) as ref_ds:
            band_count = ref_ds.RasterCount
        
        dst_ds = driver.Create(output_path, width, height, band_count, gdal.GDT_Byte, options=creation_options)
        
        # Set geotransform and projection
        geotransform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(srs.ExportToWkt())
        
        # Load all weight arrays once via memory-map before the tile loop
        weight_arrays = [np.load(p, mmap_mode='r') for p in weight_paths]
        
        # Open each warped GDAL dataset once before the tile loop
        with ExitStack() as stack:
            warped_datasets = [stack.enter_context(open_gdal_dataset(p)) for p in warped_paths]
            
            # Process tiles
            total_tiles = 0
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    tile_width = min(tile_size, width - x)
                    tile_height = min(tile_size, height - y)
                    total_tiles += 1
                    
                    if total_tiles % 100 == 0:
                        self.logger.info(f"Blending tile {total_tiles}: ({x}, {y}) size {tile_width}x{tile_height}")
                    
                    # Slice each weight array for this tile
                    # Weights are stored as uint8 (0..255) on disk to save RAM/disk space; convert
                    # the small per-tile slice back to float32 in [0, 1] for the blending math.
                    tile_weights = [
                        w[y:y+tile_height, x:x+tile_width].astype(np.float32) / 255.0
                        for w in weight_arrays
                    ]
                    
                    # Process each band
                    for band_num in range(1, band_count + 1):
                        # Initialize accumulators for this band
                        weighted_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
                        weight_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
                        
                        # Accumulate weighted values from each image using pre-opened datasets
                        for i, src_ds in enumerate(warped_datasets):
                            band = src_ds.GetRasterBand(band_num)
                            data = band.ReadAsArray(xoff=x, yoff=y, win_xsize=tile_width, win_ysize=tile_height)
                            
                            # Apply weights
                            weighted_sum += data.astype(np.float32) * tile_weights[i]
                            weight_sum += tile_weights[i]
                            
                            del data  # Free memory immediately
                        
                        # Normalize by weight sum
                        # Avoid division by zero
                        with np.errstate(divide='ignore', invalid='ignore'):
                            result = np.where(weight_sum > 0, weighted_sum / weight_sum, 0)
                        
                        # Convert to uint8
                        result = np.clip(result, 0, 255).astype(np.uint8)
                        
                        # Write to output
                        dst_band = dst_ds.GetRasterBand(band_num)
                        dst_band.WriteArray(result, xoff=x, yoff=y)
                        dst_band.SetNoDataValue(0)
                        
                        # Free memory
                        del weighted_sum, weight_sum, result
                    
                    # Free memory for tile_weights at the end of each tile iteration
                    del tile_weights
        
        # Close dataset to release file lock on Windows
        dst_ds = None

    def _create_with_gdal(self, tiff_paths: List[str], output_dir: str) -> str:
        """
        Create orthophoto using GDAL with seamless blending (alternative method).

        Args:
            tiff_paths: List of paths to TIFF files
            output_dir: Directory to save results

        Returns:
            Path to the created orthophoto

        Raises:
            RuntimeError: If GDAL merge fails
        """
        try:
            self.logger.info(f"[{len(tiff_paths)} files] Creating orthophoto using GDAL with seamless blending")
            
            # Handle single-input degenerate case
            if len(tiff_paths) == 1:
                self.logger.info("Only one input image, using direct copy")
                output_path = os.path.join(output_dir, "orthophoto.tif")
                shutil.copy2(tiff_paths[0], output_path)
                return output_path

            # Calculate input files total size
            input_size = sum(os.path.getsize(path) for path in tiff_paths if os.path.exists(path))

            # Use gdal.Warp to create mosaic with tight bounding box
            output_path = os.path.join(output_dir, "orthophoto.tif")

            # Import GDAL
            try:
                from osgeo import gdal, osr
                gdal.UseExceptions()
            except ImportError:
                raise RuntimeError("GDAL library is required but not available. Install with: pip install gdal")

            # Calculate tight bounding box from input extents
            min_x, min_y, max_x, max_y = None, None, None, None
            first_srs = None
            pixel_size_x, pixel_size_y = None, None
            
            for tiff_path in tiff_paths:
                self.logger.debug(f"About to open GDAL dataset for bounds calculation: {tiff_path}")
                with open_gdal_dataset(tiff_path) as src_ds:
                    self.logger.debug(f"Opened GDAL dataset for bounds calculation: {tiff_path}")
                    geotransform = src_ds.GetGeoTransform()
                    srs = src_ds.GetSpatialRef()
                    
                    if first_srs is None:
                        first_srs = srs
                        pixel_size_x = abs(geotransform[1])
                        pixel_size_y = abs(geotransform[5])
                    
                    # Calculate bounds
                    x_size = src_ds.RasterXSize
                    y_size = src_ds.RasterYSize
                    ul_x = geotransform[0]
                    ul_y = geotransform[3]
                    lr_x = ul_x + geotransform[1] * x_size + geotransform[2] * y_size
                    lr_y = ul_y + geotransform[4] * x_size + geotransform[5] * y_size
                    
                    # Get actual bounds (accounting for rotation)
                    x_coords = [ul_x, ul_x + geotransform[1] * x_size, ul_x + geotransform[2] * y_size, lr_x]
                    y_coords = [ul_y, ul_y + geotransform[4] * x_size, ul_y + geotransform[5] * y_size, lr_y]
                    file_min_x, file_max_x = min(x_coords), max(x_coords)
                    file_min_y, file_max_y = min(y_coords), max(y_coords)
                    
                    # Update overall bounds
                    if min_x is None or file_min_x < min_x:
                        min_x = file_min_x
                    if min_y is None or file_min_y < min_y:
                        min_y = file_min_y
                    if max_x is None or file_max_x > max_x:
                        max_x = file_max_x
                    if max_y is None or file_max_y > max_y:
                        max_y = file_max_y

            # Calculate output dimensions
            if min_x is not None and pixel_size_x is not None:
                width = int((max_x - min_x) / pixel_size_x)
                height = int((max_y - min_y) / pixel_size_y)
                
                # Check if blending is enabled
                if self.blend_config["enabled"] and len(tiff_paths) > 1:
                    # Use seamless blending approach
                    with tempfile.TemporaryDirectory() as temp_dir:
                        self.logger.info(f"Reprojecting {len(tiff_paths)} images to common grid")
                        
                        # Warp all images to common grid
                        warped_paths = self._warp_to_common_grid(
                            tiff_paths, temp_dir, 
                            (min_x, min_y, max_x, max_y),
                            pixel_size_x, pixel_size_y, first_srs
                        )
                        
                        self.logger.info("Computing distance weights for blending")
                        # Compute distance weights for blending
                        weight_paths = self._compute_distance_weights(warped_paths, temp_dir)
                        
                        self.logger.info("Blending images with distance-weighted approach")
                        # Blend images using precomputed weights
                        self._blend_tiles(
                            warped_paths, weight_paths, output_path,
                            width, height, pixel_size_x, pixel_size_y,
                            min_x, min_y, max_x, max_y, first_srs
                        )
                        
                        # Force garbage collection to release GDAL file handles on Windows
                        import gc
                        gc.collect()
                        
                        # Additional safety net for Windows file locks
                        import time
                        import shutil
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                # The temporary directory will be automatically cleaned up
                                # by the context manager, but we want to ensure it's clean
                                # No explicit cleanup needed as context manager handles it
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                                else:
                                    self.logger.debug(f"About to clean up temporary directory (attempt {attempt + 1}): {temp_dir}")
                                    self.logger.warning(f"Failed to clean up temporary directory after {max_retries} attempts: {e}")
                else:
                    # Fall back to original behavior (last image wins)
                    self.logger.info("Using original GDAL warp (last image wins)")
                    
                    # Build creation options from config
                    creation_options = []
                    if self.orthophoto_config["compression"] != "NONE":
                        creation_options.extend([
                            f"COMPRESS={self.orthophoto_config['compression']}",
                        ])
                    
                    if self.orthophoto_config["tiled"]:
                        creation_options.extend([
                            "TILED=YES",
                            f"BLOCKXSIZE={self.orthophoto_config['block_size']}",
                            f"BLOCKYSIZE={self.orthophoto_config['block_size']}",
                        ])
                    
                    creation_options.append(f"BIGTIFF={self.orthophoto_config['bigtiff']}")
                    
                    # Determine predictor based on data type and config
                    predictor = self.orthophoto_config["predictor"]
                    if predictor == "auto":
                        # For integer data, use PREDICTOR=2; for float, use PREDICTOR=3
                        predictor = 2  # Default to 2 for integer data
                    
                    if predictor != 1:  # PREDICTOR=1 is no predictor
                        creation_options.append(f"PREDICTOR={predictor}")
                    
                    # Warp options for tight bounding box
                    warp_options = gdal.WarpOptions(
                        format="GTiff",
                        outputBounds=[min_x, min_y, max_x, max_y],
                        width=width,
                        height=height,
                        resampleAlg="bilinear",
                        dstNodata=0,
                        creationOptions=creation_options,
                        srcSRS=first_srs,
                        dstSRS=first_srs,
                    )
                    
                    # Perform the warp operation
                    self.logger.debug(f"About to call gdal.Warp with {len(tiff_paths)} sources, destination: {output_path}")
                    warped_ds = gdal.Warp(output_path, tiff_paths, options=warp_options)
                    warped_ds = None  # Release dataset to prevent file lock on Windows
                    self.logger.debug(f"gdal.Warp released dataset for: {output_path}")
                
                # Convert to target dtype if needed
                if self.orthophoto_config["target_dtype"] == "uint8":
                    self._convert_to_uint8(output_path)
                
                # Build overviews if requested
                if self.orthophoto_config["build_overviews"]:
                    self._build_overviews(output_path)
                
                # Log size reduction
                if os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
                    reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
                    self.logger.info(f"Orthophoto size: {input_size:,} → {output_size:,} bytes ({reduction:.1f}% reduction)")
                
                return output_path
            else:
                raise RuntimeError("Could not determine output bounds from input files")

        except Exception as e:
            self.logger.exception(f"Error creating orthophoto with GDAL: {e}")
            raise

    def _create_gps_file(
        self, processed_data: Dict[str, Any], project_dir: str
    ) -> Optional[str]:
        """
        Create GPS file for OpenDroneMap.

        Args:
            processed_data: Processing data containing metadata
            project_dir: Project directory

        Returns:
            Path to GPS file or None if not created
        """
        try:
            # Check for GPS data in metadata
            metadata = processed_data.get("metadata", {})
            if not metadata:
                return None

            gps_file = os.path.join(project_dir, "gps.txt")

            # Try to extract GPS data from metadata using known key structures
            lat = None
            lon = None
            alt = None

            # Try nested dicts first: "gps", "geolocation", "coordinates"
            for key in ("gps", "geolocation", "coordinates"):
                gps_dict = metadata.get(key)
                if isinstance(gps_dict, dict):
                    lat = gps_dict.get("latitude")
                    lon = gps_dict.get("longitude")
                    alt = gps_dict.get("altitude", 0.0)
                    if lat is not None and lon is not None:
                        break

            # Fall back to flat top-level keys
            if lat is None or lon is None:
                lat = metadata.get("lat")
                lon = metadata.get("lon")
                alt = metadata.get("alt", 0.0)

            # No GPS data found — do not create the file
            if lat is None or lon is None:
                return None

            # Default altitude to 0.0 if still missing
            if alt is None:
                alt = 0.0

            # Collect image filenames from tiff_paths
            tiff_paths = processed_data.get("tiff_paths", [])
            if tiff_paths:
                filenames = [os.path.basename(p) for p in tiff_paths]
            else:
                filenames = ["image.tif"]

            # Write ODM-compatible GPS file: filename lat lon alt per line
            with open(gps_file, "w") as f:
                for filename in filenames:
                    f.write(f"{filename} {float(lat):.6f} {float(lon):.6f} {float(alt):.6f}\n")

            return gps_file

        except Exception as e:
            self.logger.warning(f"Error creating GPS file: {e}")
            return None

    def _copy_file(self, src: str, dst: str) -> None:
        """
        Copy a file from source to destination.

        Args:
            src: Source file path
            dst: Destination file path
        """

        shutil.copy2(src, dst)

    def validate_orthophoto(self, orthophoto_path: str) -> Dict[str, Any]:
        """
        Validate the created orthophoto.

        Args:
            orthophoto_path: Path to the orthophoto file

        Returns:
            Dictionary with validation results
        """
        try:
            from ..utils.gdal_utils import get_raster_metadata

            metadata = get_raster_metadata(orthophoto_path)

            validation_results = {
                "valid": True,
                "width": metadata["width"],
                "height": metadata["height"],
                "bands": metadata["band_count"],
                "has_georeference": metadata["has_georeference"],
                "has_projection": metadata["has_projection"],
            }

            # Check for empty areas
            if metadata["band_stats"] and metadata["band_stats"].get(1):
                stats = metadata["band_stats"][1]
                if stats["max"] <= stats["min"]:  # max <= min
                    validation_results["valid"] = False
                    validation_results["error"] = (
                        "Image contains only empty values"
                    )

            return validation_results

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def optimize_orthophoto(self, orthophoto_path: str, output_path: str = None) -> str:
        """
        Optimize orthophoto (compression, pyramids).

        Args:
            orthophoto_path: Path to the source orthophoto
            output_path: Path to save the optimized file

        Returns:
            Path to the optimized orthophoto

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            if output_path is None:
                base_path = os.path.splitext(orthophoto_path)[0]
                output_path = f"{base_path}_optimized.tif"

            # Calculate input file size
            input_size = os.path.getsize(orthophoto_path) if os.path.exists(orthophoto_path) else 0

            # Import GDAL
            try:
                from osgeo import gdal
                gdal.UseExceptions()
            except ImportError:
                raise RuntimeError("GDAL library is required but not available. Install with: pip install gdal")

            # Build creation options from config
            creation_options = []
            if self.orthophoto_config["compression"] != "NONE":
                creation_options.extend([
                    f"COMPRESS={self.orthophoto_config['compression']}",
                ])
            
            if self.orthophoto_config["tiled"]:
                creation_options.extend([
                    "TILED=YES",
                    f"BLOCKXSIZE={self.orthophoto_config['block_size']}",
                    f"BLOCKYSIZE={self.orthophoto_config['block_size']}",
                ])
            
            creation_options.append(f"BIGTIFF={self.orthophoto_config['bigtiff']}")
            
            # Determine predictor based on data type and config
            predictor = self.orthophoto_config["predictor"]
            if predictor == "auto":
                # For integer data, use PREDICTOR=2; for float, use PREDICTOR=3
                predictor = 2  # Default to 2 for integer data
            
            if predictor != 1:  # PREDICTOR=1 is no predictor
                creation_options.append(f"PREDICTOR={predictor}")

            # Use gdal.Translate for optimization
            translate_options = gdal.TranslateOptions(
                format="GTiff",
                creationOptions=creation_options,
            )
            
            gdal.Translate(output_path, orthophoto_path, options=translate_options)
            
            # Convert to target dtype if needed
            if self.orthophoto_config["target_dtype"] == "uint8":
                self._convert_to_uint8(output_path)
            
            # Build overviews if requested
            if self.orthophoto_config["build_overviews"]:
                self._build_overviews(output_path)
            
            # Log size reduction
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0
                self.logger.info(f"Orthophoto optimized: {input_size:,} → {output_size:,} bytes ({reduction:.1f}% reduction)")
            
            return output_path

        except Exception as e:
            self.logger.error(f"Error optimizing orthophoto: {e}")
            raise

    def _create_with_opencv(self, tiff_paths: List[str], output_dir: str) -> str:
        """
        Create orthophoto using OpenCV feature-based stitching.
        
        This method implements feature-based stitching using OpenCV's stitching module
        with a fallback to manual SIFT/ORB + homography approach.
        
        Args:
            tiff_paths: List of paths to TIFF files
            output_dir: Directory to save results
            
        Returns:
            Path to the created orthophoto
            
        Raises:
            RuntimeError: If stitching fails due to insufficient matches or other issues
        """
        # Check if cv2 is available
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV stitching requires opencv-contrib-python. Install with: pip install opencv-contrib-python")
        
        self.logger.info(f"[{len(tiff_paths)} files] Starting OpenCV feature-based stitching")
        
        # Load and normalize images
        images = []
        for i, path in enumerate(tiff_paths):
            self.logger.info(f"Loading image {i+1}/{len(tiff_paths)}: {os.path.basename(path)}")
            img = self._load_and_normalize(path)
            images.append(img)
        
        # Try primary path using cv2.Stitcher
        result_path = self._try_cv2_stitcher(images, tiff_paths, output_dir)
        
        # If primary path failed, try manual fallback
        if result_path is None:
            self.logger.info("Falling back to manual stitching pipeline")
            result_path = self._manual_stitching_pipeline(images, tiff_paths, output_dir)
        
        self.logger.info(f"OpenCV stitching completed: {result_path}")
        return result_path
    
    def _load_and_normalize(self, image_path: str) -> np.ndarray:
        """
        Load image and normalize to uint8 if needed.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image as numpy array (BGR format for OpenCV)
        """
        # Try to load with OpenCV first (handles many formats including TIFF)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            # Fallback to GDAL for geospatial formats
            try:
                from osgeo import gdal
                gdal.UseExceptions()
                with open_gdal_dataset(image_path) as ds:
                    # Read first 3 bands or first band if grayscale
                    band_count = min(ds.RasterCount, 3)
                    bands = []
                    for i in range(1, band_count + 1):
                        band = ds.GetRasterBand(i)
                        band_data = band.ReadAsArray()
                        bands.append(band_data)
                    
                    # Stack bands
                    if len(bands) == 1:
                        img = bands[0]
                    elif len(bands) == 3:
                        # Stack as BGR (OpenCV format)
                        img = np.stack([bands[2], bands[1], bands[0]], axis=2)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            self.logger.info(f"Converting image from {img.dtype} to uint8")
            # For 16-bit images, use percentile-based normalization
            if img.dtype in [np.uint16, np.int16]:
                # Use 1st and 99th percentiles to avoid outliers
                p1, p99 = np.percentile(img, (1, 99))
                img = np.clip(img, p1, p99)
                img = ((img - p1) / (p99 - p1) * 255).astype(np.uint8)
            elif img.dtype in [np.float32, np.float64]:
                # For float images, assume 0-1 range or normalize
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255 range
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.full_like(img, 128, dtype=np.uint8)
            else:
                # For other dtypes, simple conversion
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Ensure 3 channels for color images
        if len(img.shape) == 2:
            # Grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            # Single channel to BGR
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Already BGR, no change needed
            pass
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def _try_cv2_stitcher(self, images: List[np.ndarray], tiff_paths: List[str], output_dir: str) -> Optional[str]:
        """
        Try to stitch images using cv2.Stitcher.
        
        Args:
            images: List of loaded images
            tiff_paths: List of original file paths (for naming)
            output_dir: Directory to save results
            
        Returns:
            Path to result file if successful, None otherwise
        """
        try:
            # Create stitcher
            if hasattr(cv2, 'Stitcher_create'):
                stitcher = cv2.Stitcher_create()  # OpenCV 3.x+
            else:
                stitcher = cv2.Stitcher.create()  # OpenCV 4.x+
            
            # Set GPU usage if requested
            if self.opencv_config["try_use_gpu"] and hasattr(stitcher, 'setTryUseGPU'):
                stitcher.setTryUseGPU(True)
            
            self.logger.info(f"Attempting cv2.Stitcher with {len(images)} images")
            
            # Perform stitching
            status, stitched = stitcher.stitch(images)
            
            # Check status
            status_messages = {
                cv2.Stitcher_OK: "OK",
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed",
            }
            
            if status == cv2.Stitcher_OK:
                self.logger.info("cv2.Stitcher succeeded")
                # Save result
                output_path = os.path.join(output_dir, "orthophoto_opencv.tif")
                self._save_pixel_space_tiff(stitched, output_path)
                
                # Also save PNG preview if possible
                try:
                    png_path = os.path.join(output_dir, "orthophoto_opencv.png")
                    # Convert BGR to RGB for PNG
                    if len(stitched.shape) == 3:
                        rgb_img = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(png_path, rgb_img)
                    else:
                        cv2.imwrite(png_path, stitched)
                    self.logger.info(f"Saved PNG preview: {png_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save PNG preview: {e}")
                
                return output_path
            else:
                status_msg = status_messages.get(status, f"Unknown error ({status})")
                if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                    self.logger.info("cv2.Stitcher requires ≥3 images, using geo-referenced fallback")
                else:
                    self.logger.warning(f"cv2.Stitcher failed with status {status}: {status_msg}")
                return None
                
        except Exception as e:
            self.logger.warning(f"cv2.Stitcher failed: {e}")
            return None
    
    def _manual_stitching_pipeline(self, images: List[np.ndarray], tiff_paths: List[str], output_dir: str) -> str:
        """
        Manual stitching pipeline using SIFT/ORB + homography + feathered blend.
        
        Args:
            images: List of loaded images
            tiff_paths: List of original file paths (for naming)
            output_dir: Directory to save results
            
        Returns:
            Path to result file
        """
        # For geo-referenced images, use geo-referenced placement
        if len(images) == 2 and len(tiff_paths) == 2:
            try:
                # Compute geo-referenced bounds
                min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, srs = self._compute_geo_bounds(tiff_paths)
                
                # Calculate output dimensions using geo-referenced bounds
                width = int(np.ceil((max_x - min_x) / pixel_size_x))
                height = int(np.ceil((max_y - min_y) / pixel_size_y))
                
                self.logger.info(f"Using geo-referenced canvas: {width}x{height}")
                
                # Use geo-referenced stitching
                result_img = self._geo_referenced_stitch(images, tiff_paths, min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, srs)
            except Exception as e:
                self.logger.warning(f"No geotransform on input(s); using feature-based (SIFT) stitching. Quality depends on overlap and texture; with only 2 images results may be unreliable.")
                self.logger.warning(f"Geo-referenced stitching failed, falling back to homography: {e}")
                result_img, _ = self._stitch_pair(images[0], images[1])
        else:
            # Handle multi-image case with geo-referenced stitching if all images have geotransforms
            try:
                # Check if all images have valid geotransforms
                from ..utils.gdal_utils import get_raster_metadata
                all_georeferenced = True
                for tiff_path in tiff_paths:
                    metadata = get_raster_metadata(tiff_path)
                    geotransform = metadata["geotransform"]
                    if geotransform is None or all(v == 0 for v in geotransform):
                        all_georeferenced = False
                        break
                
                if all_georeferenced and len(images) > 2:
                    # Compute geo-referenced bounds
                    min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, srs = self._compute_geo_bounds(tiff_paths)
                    
                    # Calculate output dimensions using geo-referenced bounds
                    width = int(np.ceil((max_x - min_x) / pixel_size_x))
                    height = int(np.ceil((max_y - min_y) / pixel_size_y))
                    
                    self.logger.info(f"Using geo-referenced canvas for {len(images)} images: {width}x{height}")
                    
                    # Use geo-referenced stitching for multi-image case
                    result_img = self._geo_referenced_stitch(images, tiff_paths, min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, srs)
                else:
                    # Fall back to homography-based stitching
                    if len(images) > 2:
                        # Count images without geotransforms
                        missing_geotransform_count = 0
                        for tiff_path in tiff_paths:
                            metadata = get_raster_metadata(tiff_path)
                            geotransform = metadata["geotransform"]
                            if geotransform is None or all(v == 0 for v in geotransform):
                                missing_geotransform_count += 1
                        self.logger.warning(f"No geotransform on {missing_geotransform_count}/{len(tiff_paths)} input(s); using feature-based (SIFT) stitching. Quality depends on overlap and texture; with only 2 images results may be unreliable.")
                        self.logger.info("Manual fallback supports pairwise stitching only; for N>2 inputs, prefer cv2.Stitcher")
                        # Stitch pairs sequentially left-to-right
                        result_img = images[0]
                        for i in range(1, len(images)):
                            self.logger.info(f"Stitching image pair {i}/{len(images)-1}")
                            result_img, _ = self._stitch_pair(result_img, images[i])
                    else:
                        # Single image, just copy it
                        result_img = images[0]
            except Exception as e:
                self.logger.warning(f"Geo-referenced stitching failed, falling back to homography: {e}")
                # Fall back to homography-based stitching
                if len(images) > 2:
                    self.logger.info("Manual fallback supports pairwise stitching only; for N>2 inputs, prefer cv2.Stitcher")
                    # Stitch pairs sequentially left-to-right
                    result_img = images[0]
                    for i in range(1, len(images)):
                        self.logger.info(f"Stitching image pair {i}/{len(images)-1}")
                        result_img, _ = self._stitch_pair(result_img, images[i])
                else:
                    # Single image, just copy it
                    result_img = images[0]
        
        # Save result
        output_path = os.path.join(output_dir, "orthophoto_opencv.tif")
        self._save_pixel_space_tiff(result_img, output_path)
        
        # Also save PNG preview if possible
        try:
            png_path = os.path.join(output_dir, "orthophoto_opencv.png")
            # Convert BGR to RGB for PNG
            if len(result_img.shape) == 3:
                rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(png_path, rgb_img)
            else:
                cv2.imwrite(png_path, result_img)
            self.logger.info(f"Saved PNG preview: {png_path}")
        except Exception as e:
            self.logger.warning(f"Could not save PNG preview: {e}")
        
        return output_path
    
    def _stitch_pair(self, img1: np.ndarray, img2: np.ndarray, geotransform1: Optional[tuple] = None, geotransform2: Optional[tuple] = None) -> tuple:
        """
        Stitch a pair of images using feature detection and homography.
        
        Args:
            img1: First image (reference)
            img2: Second image (to be aligned)
            geotransform1: Geo-transform for first image (optional)
            geotransform2: Geo-transform for second image (optional)
            
        Returns:
            Tuple of (stitched_image, success_flag)
        """
        # Detect and match features
        kp1, des1, kp2, des2, detector_name = self._detect_and_match(img1, img2)
        
        # Apply Lowe's ratio test
        matches = self._apply_ratio_test(des1, des2)
        
        self.logger.info(f"Found {len(matches)} good matches using {detector_name}")
        
        if len(matches) < self.opencv_config["min_matches"]:
            raise RuntimeError(f"OpenCV stitching failed: not enough feature matches ({len(matches)} < {self.opencv_config['min_matches']}) between images. Inputs likely don't overlap or lack texture.")
        
        # Compute homography
        H, inliers = self._compute_homography(kp1, kp2, matches)
        
        inlier_ratio = len(inliers) / len(matches) if len(matches) > 0 else 0
        self.logger.info(f"Found homography with {len(inliers)} inliers ({inlier_ratio:.2%} inlier ratio)")
        
        # Warp and blend images
        result_img = self._warp_and_blend(img1, img2, H)
        
        return result_img, True
    
    def _detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """
        Detect features and match them between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of (kp1, des1, kp2, des2, detector_name)
        """
        # Prepare images for feature detection (downscale if needed)
        small_img1, scale1 = self._prepare_for_features(img1)
        small_img2, scale2 = self._prepare_for_features(img2)
        
        # Convert to grayscale if needed
        if len(small_img1.shape) == 3:
            gray1 = cv2.cvtColor(small_img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = small_img1
            
        if len(small_img2.shape) == 3:
            gray2 = cv2.cvtColor(small_img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = small_img2
        
        # Choose detector
        detector_name = self.opencv_config["detector"]
        if detector_name == "auto":
            # Try SIFT first, fall back to ORB
            try:
                detector = cv2.SIFT_create()
                detector_name = "SIFT"
                self.logger.info("Using SIFT detector")
            except AttributeError:
                detector = cv2.ORB_create(nfeatures=5000)
                detector_name = "ORB"
                self.logger.info("Using ORB detector (SIFT not available)")
        elif detector_name == "sift":
            try:
                detector = cv2.SIFT_create()
                detector_name = "SIFT"
                self.logger.info("Using SIFT detector")
            except AttributeError:
                # Fallback to ORB if SIFT is not available
                detector = cv2.ORB_create(nfeatures=5000)
                detector_name = "ORB"
                self.logger.warning("SIFT requested but not available, falling back to ORB")
        else:  # orb
            detector = cv2.ORB_create(nfeatures=5000)
            detector_name = "ORB"
            self.logger.info("Using ORB detector")
        
        # Detect features
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        self.logger.info(f"Detected {len(kp1)} features in image 1, {len(kp2)} in image 2")
        
        if des1 is None or des2 is None:
            raise RuntimeError("OpenCV stitching failed: could not detect features in one or both images")
        
        # Rescale keypoints back to original image coordinates if downscaling was applied
        if scale1 != 1.0:
            # Rescale keypoints for img1
            scale1_inv = 1.0 / scale1
            for kp in kp1:
                kp.pt = (kp.pt[0] * scale1_inv, kp.pt[1] * scale1_inv)
        
        if scale2 != 1.0:
            # Rescale keypoints for img2
            scale2_inv = 1.0 / scale2
            for kp in kp2:
                kp.pt = (kp.pt[0] * scale2_inv, kp.pt[1] * scale2_inv)
        
        return kp1, des1, kp2, des2, detector_name
    
    def _apply_ratio_test(self, des1: np.ndarray, des2: np.ndarray) -> list:
        """
        Apply Lowe's ratio test to filter good matches.
        
        Args:
            des1: Descriptors for first image
            des2: Descriptors for second image
            
        Returns:
            List of good matches
        """
        # Create matcher based on descriptor type
        if des1.dtype == np.uint8:
            # ORB descriptors - use Hamming distance
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # SIFT descriptors - use L2 distance
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Find matches
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        ratio = self.opencv_config["ratio_test"]
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _compute_homography(self, kp1: list, kp2: list, matches: list) -> tuple:
        """
        Compute homography matrix using RANSAC.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches between images
            
        Returns:
            Tuple of (homography_matrix, inliers)
        """
        if len(matches) < 4:
            raise RuntimeError(f"OpenCV stitching failed: not enough matches ({len(matches)}) to compute homography (minimum 4 required)")
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Compute homography with RANSAC
        threshold = self.opencv_config["ransac_reproj_threshold"]
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        
        if H is None:
            raise RuntimeError("OpenCV stitching failed: homography estimation failed")
        
        # Extract inliers
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        
        return H, inliers
    
    def _compute_geo_bounds(self, tiff_paths: List[str]) -> tuple:
        """
        Compute tight bounding box from geo-referenced input extents.
        
        Args:
            tiff_paths: List of paths to geo-referenced TIFF files
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, srs)
        """
        from ..utils.gdal_utils import get_raster_metadata
        from osgeo import osr
        
        min_x, min_y, max_x, max_y = None, None, None, None
        first_srs = None
        pixel_size_x, pixel_size_y = None, None
        
        for tiff_path in tiff_paths:
            metadata = get_raster_metadata(tiff_path)
            geotransform = metadata["geotransform"]
            srs_wkt = metadata["projection"]
            
            if first_srs is None:
                first_srs = osr.SpatialReference()
                first_srs.ImportFromWkt(srs_wkt)
                pixel_size_x = abs(geotransform[1])
                pixel_size_y = abs(geotransform[5])
            
            # Calculate bounds
            x_size = metadata["width"]
            y_size = metadata["height"]
            ul_x = geotransform[0]
            ul_y = geotransform[3]
            lr_x = ul_x + geotransform[1] * x_size + geotransform[2] * y_size
            lr_y = ul_y + geotransform[4] * x_size + geotransform[5] * y_size
            
            # Get actual bounds (accounting for rotation)
            x_coords = [ul_x, ul_x + geotransform[1] * x_size, ul_x + geotransform[2] * y_size, lr_x]
            y_coords = [ul_y, ul_y + geotransform[4] * x_size, ul_y + geotransform[5] * y_size, lr_y]
            file_min_x, file_max_x = min(x_coords), max(x_coords)
            file_min_y, file_max_y = min(y_coords), max(y_coords)
            
            # Update overall bounds
            if min_x is None or file_min_x < min_x:
                min_x = file_min_x
            if min_y is None or file_min_y < min_y:
                min_y = file_min_y
            if max_x is None or file_max_x > max_x:
                max_x = file_max_x
            if max_y is None or file_max_y > max_y:
                max_y = file_max_y
        
        return min_x, min_y, max_x, max_y, pixel_size_x, pixel_size_y, first_srs
    
    def _warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp second image and blend with first using distance transform.
        
        Args:
            img1: First image (reference)
            img2: Second image (to be warped)
            H: Homography matrix
            
        Returns:
            Blended image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Compute corners of warped image2
        corners = np.array([
            [0, 0, 1],
            [w2, 0, 1],
            [w2, h2, 1],
            [0, h2, 1]
        ], dtype=np.float32)
        
        # Apply homography
        warped_corners = np.dot(H, corners.T).T
        # Normalize homogeneous coordinates
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        
        # Find bounding box of warped image
        min_x = min(0, np.min(warped_corners[:, 0]))
        max_x = max(w1, np.max(warped_corners[:, 0]))
        min_y = min(0, np.min(warped_corners[:, 1]))
        max_y = max(h1, np.max(warped_corners[:, 1]))
        
        # Create translation matrix to ensure positive coordinates
        tx = -min_x
        ty = -min_y
        translation = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Adjust homography with translation
        H_translated = np.dot(translation, H)
        
        # Calculate canvas size
        canvas_width = int(np.ceil(max_x - min_x))
        canvas_height = int(np.ceil(max_y - min_y))
        
        self.logger.info(f"Output canvas size: {canvas_width}x{canvas_height}")
        
        # Warp img2 to canvas
        warped_img2 = cv2.warpPerspective(img2, H_translated, (canvas_width, canvas_height))
        
        # Translate img1 to canvas
        warped_img1 = np.zeros((canvas_height, canvas_width, img1.shape[2]), dtype=img1.dtype)
        y_offset = int(np.floor(ty))
        x_offset = int(np.floor(tx))
        warped_img1[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
        
        # Create masks for valid data areas
        mask1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        mask1[y_offset:y_offset+h1, x_offset:x_offset+w1] = 255
        
        mask2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        # Create mask from warped image (non-black pixels)
        if len(warped_img2.shape) == 3:
            mask2 = np.any(warped_img2 > 0, axis=2).astype(np.uint8) * 255
        else:
            mask2 = (warped_img2 > 0).astype(np.uint8) * 255
        
        # Find overlap area
        overlap = cv2.bitwise_and(mask1, mask2)
        
        # For areas where only one image exists, use that image directly
        only_img1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
        only_img2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
        
        # For overlap area, use distance transform for feathered blending
        if np.any(overlap):
            # Use tile-based processing to avoid memory issues with large canvases
            tile_size = 2048  # Process in 2048x2048 tiles
            result = np.zeros_like(warped_img1, dtype=np.uint8)
            
            # Process tiles
            for y in range(0, canvas_height, tile_size):
                for x in range(0, canvas_width, tile_size):
                    # Define tile boundaries
                    y_end = min(y + tile_size, canvas_height)
                    x_end = min(x + tile_size, canvas_width)
                    tile_h = y_end - y
                    tile_w = x_end - x
                    
                    # Extract tile regions
                    tile_overlap = overlap[y:y_end, x:x_end]
                    tile_only_img1 = only_img1[y:y_end, x:x_end]
                    tile_only_img2 = only_img2[y:y_end, x:x_end]
                    tile_warped_img1 = warped_img1[y:y_end, x:x_end]
                    tile_warped_img2 = warped_img2[y:y_end, x:x_end]
                    
                    # Process tile
                    if np.any(tile_overlap):
                        # Compute distance transforms for this tile (ensure float32)
                        tile_dist1 = cv2.distanceTransform(tile_only_img1 | tile_overlap, cv2.DIST_L2, 5).astype(np.float32)
                        tile_dist2 = cv2.distanceTransform(tile_only_img2 | tile_overlap, cv2.DIST_L2, 5).astype(np.float32)
                        
                        # Normalize each distance map to [0,1] before combining (in-place where possible)
                        max_dist1 = np.max(tile_dist1)
                        max_dist2 = np.max(tile_dist2)
                        if max_dist1 > 0:
                            tile_dist1 /= max_dist1  # In-place division
                        if max_dist2 > 0:
                            tile_dist2 /= max_dist2  # In-place division
                        
                        # Compute weight sum with guard against division by zero
                        tile_weight_sum = tile_dist1 + tile_dist2
                        tile_weight_sum = np.where(tile_weight_sum > 0, tile_weight_sum, 1)
                        
                        # Compute normalized weights
                        tile_w1 = np.where(tile_weight_sum > 0, tile_dist1 / tile_weight_sum, 0)
                        tile_w2 = np.where(tile_weight_sum > 0, tile_dist2 / tile_weight_sum, 0)
                        
                        # Free intermediates
                        del tile_dist1, tile_dist2, tile_weight_sum
                        
                        # Apply blending in overlap area for this tile
                        tile_result = np.zeros((tile_h, tile_w, tile_warped_img1.shape[2]), dtype=np.float32)
                        for c in range(tile_warped_img1.shape[2]):
                            # Use in-place operations where possible
                            weighted_img1 = tile_w1 * tile_warped_img1[:, :, c].astype(np.float32)
                            weighted_img2 = tile_w2 * tile_warped_img2[:, :, c].astype(np.float32)
                            tile_result[:, :, c] = weighted_img1 + weighted_img2
                            # Free intermediates immediately
                            del weighted_img1, weighted_img2
                        
                        tile_result = np.clip(tile_result, 0, 255, out=tile_result).astype(np.uint8)
                        
                        # Copy blended result to output
                        result[y:y_end, x:x_end] = tile_result
                        
                        # Free intermediates
                        del tile_w1, tile_w2, tile_result
                    else:
                        # No overlap in this tile, copy directly
                        result[y:y_end, x:x_end] = tile_warped_img1
            
            # Apply single-image areas (outside overlap)
            # In areas where only img1 exists, keep img1
            # In areas where only img2 exists, use img2
            if len(result.shape) == 3:
                for c in range(result.shape[2]):
                    result[only_img1 > 0, c] = warped_img1[only_img1 > 0, c]
                    result[only_img2 > 0, c] = warped_img2[only_img2 > 0, c]
            else:
                result[only_img1 > 0] = warped_img1[only_img1 > 0]
                result[only_img2 > 0] = warped_img2[only_img2 > 0]
        else:
            # No overlap, just combine images
            result = warped_img1.copy()
        
        # Apply single-image areas
        # In areas where only img1 exists, keep img1
        # In areas where only img2 exists, use img2
        if len(result.shape) == 3:
            for c in range(result.shape[2]):
                result[only_img1 > 0, c] = warped_img1[only_img1 > 0, c]
                result[only_img2 > 0, c] = warped_img2[only_img2 > 0, c]
        else:
            result[only_img1 > 0] = warped_img1[only_img1 > 0]
            result[only_img2 > 0] = warped_img2[only_img2 > 0]
        
        # Force garbage collection to free any remaining intermediates
        gc.collect()
        
        return result
    
    def _save_pixel_space_tiff(self, image: np.ndarray, output_path: str) -> None:
        """
        Save image as non-georeferenced TIFF with compression.
        
        Args:
            image: Image to save
            output_path: Path to save the image
        """
        # Try to save with cv2 first (simpler)
        try:
            # Save with LZW compression if possible
            cv2.imwrite(output_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 5])  # LZW
            self.logger.info(f"Saved orthophoto with LZW compression: {output_path}")
            return
        except Exception as e:
            self.logger.warning(f"Could not save with LZW compression: {e}")
        
        # Fallback to GDAL for more control
        try:
            from osgeo import gdal, gdalconst
            gdal.UseExceptions()
            
            # Create driver
            driver = gdal.GetDriverByName('GTiff')
            
            # Determine image dimensions and bands
            if len(image.shape) == 2:
                height, width = image.shape
                bands = 1
            else:
                height, width, bands = image.shape
            
            # Create dataset
            options = [
                'COMPRESS=LZW',
                'TILED=YES',
                'BIGTIFF=IF_SAFER'
            ]
            dst_ds = driver.Create(output_path, width, height, bands, gdal.GDT_Byte, options=options)
            
            # Write data
            if bands == 1:
                dst_ds.GetRasterBand(1).WriteArray(image)
            else:
                # OpenCV uses BGR, GDAL expects RGB - but since this is pixel-space,
                # we'll keep the BGR order to maintain consistency
                for i in range(bands):
                    dst_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])
            
            # Close dataset
            dst_ds = None
            self.logger.info(f"Saved orthophoto with GDAL: {output_path}")
        except Exception as e:
            # Final fallback - save with cv2 without compression
            self.logger.warning(f"Could not save with GDAL: {e}")
            cv2.imwrite(output_path, image)
            self.logger.info(f"Saved orthophoto without compression: {output_path}")

    def _prepare_for_features(self, img: np.ndarray) -> tuple:
        """
        Prepare image for feature detection by downscaling if needed.
        
        Args:
            img: Input image array
            
        Returns:
            Tuple of (processed_image, scale_factor) where scale_factor is the
            ratio of original size to processed size (1.0 if no scaling applied)
        """
        max_dim = self.opencv_config["max_feature_dim"]
        height, width = img.shape[:2]
        
        # Check if downscaling is needed
        max_current_dim = max(height, width)
        if max_current_dim <= max_dim:
            # No downscaling needed
            self.logger.info(f"Image size {width}x{height} within limit ({max_dim}px), using original size for features")
            return img, 1.0
        
        # Calculate scale factor
        scale = max_dim / max_current_dim
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Downscale image
        small_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        self.logger.info(f"Downscaled image from {width}x{height} to {new_width}x{new_height} (scale={scale:.3f}) for feature detection")
        
        # Force garbage collection to free any intermediates
        gc.collect()
        
        return small_img, scale
    
    def _geo_referenced_stitch(self, images: List[np.ndarray], tiff_paths: List[str],
                                 min_x: float, min_y: float, max_x: float, max_y: float,
                                 pixel_size_x: float, pixel_size_y: float, srs) -> np.ndarray:
        """
        Stitch images using geo-referenced placement and distance-weighted blending.
        
        Args:
            images: List of loaded images
            tiff_paths: List of original file paths
            min_x, min_y, max_x, max_y: Canvas bounds in geospatial coordinates
            pixel_size_x, pixel_size_y: Pixel size in geospatial units
            srs: Spatial reference system
            
        Returns:
            Stitched image as numpy array
        """
        from ..utils.gdal_utils import get_raster_metadata
        
        # Calculate canvas dimensions
        canvas_width = int(np.ceil((max_x - min_x) / pixel_size_x))
        canvas_height = int(np.ceil((max_y - min_y) / pixel_size_y))
        
        self.logger.info(f"Creating geo-referenced canvas: {canvas_width}x{canvas_height}")
        
        # Initialize accumulators for blending using disk-backed memmap arrays
        # canvas_sum holds Σ(color×w) as float32, weight_sum holds Σ(w) as float32
        # Create temporary files for memmap arrays
        import tempfile
        canvas_sum_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        weight_sum_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        canvas_sum_file.close()
        weight_sum_file.close()
        
        # Create memmap arrays backed by temporary files
        canvas_sum = np.memmap(canvas_sum_file.name, dtype=np.float32, mode='w+',
                              shape=(canvas_height, canvas_width, 3))
        weight_sum = np.memmap(weight_sum_file.name, dtype=np.float32, mode='w+',
                              shape=(canvas_height, canvas_width))
        
        # Initialize to zero
        canvas_sum[:] = 0
        weight_sum[:] = 0
        
        # Ensure data is written to disk
        canvas_sum.flush()
        weight_sum.flush()
        
        try:
            # Process each image
            for i, (image, tiff_path) in enumerate(zip(images, tiff_paths)):
                self.logger.info(f"Processing image {i+1}/{len(images)}: {os.path.basename(tiff_path)}")
                
                try:
                    # Get image geotransform
                    metadata = get_raster_metadata(tiff_path)
                    geotransform = metadata["geotransform"]
                    
                    if geotransform is None:
                        raise RuntimeError(f"Image {tiff_path} has no geotransform")
                    
                    # Get image dimensions
                    image_height, image_width = image.shape[:2]
                    
                    # Compute canvas_max_y
                    canvas_max_y = max_y
                    
                    # Compute placement rectangle on canvas
                    tile_x, tile_y, tile_width, tile_height = self._tile_to_canvas_rect(
                        geotransform, min_x, min_y, pixel_size_x, pixel_size_y, image_width, image_height, canvas_max_y
                    )
                    
                    self.logger.info(f"  Tile rect: ({tile_x}, {tile_y}) size {tile_width}x{tile_height}")
                    
                    # Resample image to canvas resolution if needed
                    resampled_image = self._resample_to_canvas_resolution(
                        image, geotransform, pixel_size_x, pixel_size_y
                    )
                    
                    # Ensure resampled image matches expected tile size
                    if resampled_image.shape[0] != tile_height or resampled_image.shape[1] != tile_width:
                        # Resize to match expected tile size
                        resampled_image = cv2.resize(resampled_image, (tile_width, tile_height),
                                                   interpolation=cv2.INTER_LINEAR)
                    
                    # Create valid pixel mask
                    if len(resampled_image.shape) == 3:
                        valid_mask = np.any(resampled_image > 0, axis=2)
                    else:
                        valid_mask = resampled_image > 0
                    
                    # Compute distance transform weights
                    weights = self._compute_tile_weights(valid_mask)
                    
                    # Blend tile into canvas
                    self._blend_tile_into_canvas(
                        resampled_image, weights, tile_x, tile_y,
                        canvas_sum, weight_sum
                    )
                    
                    self.logger.info(f"  Successfully processed image {i+1}")
                    
                except Exception as e:
                    self.logger.warning(f"  Failed to process image {i+1} ({tiff_path}): {e}")
                    continue
            
            # Flush data to disk before final normalization
            canvas_sum.flush()
            weight_sum.flush()
        
        finally:
            # Clean up temporary files
            try:
                del canvas_sum
                del weight_sum
                os.unlink(canvas_sum_file.name)
                os.unlink(weight_sum_file.name)
            except:
                pass
        
        # Normalize canvas to create final image
        # Allocate the final output as uint8 (1.95 GB — unavoidable; this is the actual orthophoto)
        final = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Iterate over the same tile grid used during blending (2048×2048 tiles):
        tile_size = 2048
        for y0 in range(0, canvas_height, tile_size):
            y1 = min(y0 + tile_size, canvas_height)
            for x0 in range(0, canvas_width, tile_size):
                x1 = min(x0 + tile_size, canvas_width)
                
                # Extract ROI from accumulators
                w = weight_sum[y0:y1, x0:x1]
                c = canvas_sum[y0:y1, x0:x1]
                
                # Perform normalization: final = canvas_sum / weight_sum
                # But we need to avoid division by zero
                out = np.where(w[..., None] > 0, c / w[..., None], 0.0)
                final[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
                
                # Free intermediates immediately
                del w, c, out
        
        result = final
        
        # Force garbage collection to free any remaining intermediates
        gc.collect()
        
        self.logger.info("Geo-referenced stitching completed successfully")
        return result
    
    def _tile_to_canvas_rect(self, geotransform: tuple, canvas_min_x: float, canvas_min_y: float,
                               pixel_size_x: float, pixel_size_y: float, image_width: int, image_height: int,
                               canvas_max_y: float) -> tuple:
        """
        Compute tile placement rectangle on canvas from geotransform.
        
        Args:
            geotransform: GDAL geotransform tuple (6 elements)
            canvas_min_x, canvas_min_y: Canvas origin in geospatial coordinates
            pixel_size_x, pixel_size_y: Canvas pixel size in geospatial units
            image_width, image_height: Image dimensions in pixels
            canvas_max_y: Canvas maximum Y coordinate in geospatial coordinates
            
        Returns:
            Tuple of (offset_x, offset_y, width, height) in canvas pixels
        """
        # Image bounds in geospatial coordinates
        img_min_x = geotransform[0]
        img_max_y = geotransform[3]  # Top edge (remember Y is inverted in geotransform)
        
        # Calculate image bounds in geospatial coordinates
        img_max_x = img_min_x + geotransform[1] * image_width + geotransform[2] * image_height
        img_min_y = img_max_y + geotransform[4] * image_width + geotransform[5] * image_height
        
        # Calculate offset in canvas pixels (use floor for offset)
        offset_x = int(np.floor((img_min_x - canvas_min_x) / pixel_size_x))
        offset_y = int(np.floor((canvas_max_y - img_max_y) / pixel_size_y))
        
        # Calculate width and height in canvas pixels (use ceil for size)
        width = int(np.ceil((img_max_x - img_min_x) / pixel_size_x))
        height = int(np.ceil((img_max_y - img_min_y) / pixel_size_y))
        
        # Force garbage collection to free any intermediates
        gc.collect()
        
        return offset_x, offset_y, width, height
    
    def _resample_to_canvas_resolution(self, image: np.ndarray, geotransform: tuple,
                                         target_pixel_size_x: float, target_pixel_size_y: float) -> np.ndarray:
        """
        Resample image to match canvas resolution.
        
        Args:
            image: Input image
            geotransform: Image's geotransform
            target_pixel_size_x, target_pixel_size_y: Target pixel size
            
        Returns:
            Resampled image
        """
        # Get image's pixel size from geotransform
        img_pixel_size_x = abs(geotransform[1])
        img_pixel_size_y = abs(geotransform[5])
        
        # If resolution matches, return as-is
        if (abs(img_pixel_size_x - target_pixel_size_x) < 1e-10 and
            abs(img_pixel_size_y - target_pixel_size_y) < 1e-10):
            return image
        
        # Calculate scale factors
        scale_x = img_pixel_size_x / target_pixel_size_x
        scale_y = img_pixel_size_y / target_pixel_size_y
        
        # Calculate new dimensions
        new_width = int(image.shape[1] * scale_x)
        new_height = int(image.shape[0] * scale_y)
        
        # Choose interpolation method based on scale
        if scale_x < 1.0 or scale_y < 1.0:
            # Downscaling - use area interpolation
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling - use linear interpolation
            interpolation = cv2.INTER_LINEAR
        
        # Resample image
        resampled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Force garbage collection to free any intermediates
        gc.collect()
        
        return resampled
    
    def _compute_tile_weights(self, valid_mask: np.ndarray) -> np.ndarray:
        """
        Compute distance transform weights for a tile.
        
        Args:
            valid_mask: Boolean mask indicating valid pixels
            
        Returns:
            Weight array (float32) normalized to [0, 1]
        """
        # Convert to uint8 for cv2.distanceTransform
        mask_uint8 = valid_mask.astype(np.uint8) * 255
        
        # Compute distance transform
        if CV2_AVAILABLE:
            # Use OpenCV distance transform (returns float32 directly)
            dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        else:
            # Fallback to scipy (returns float64, cast to float32)
            from scipy.ndimage import distance_transform_edt
            dist_transform = distance_transform_edt(mask_uint8).astype(np.float32)
        
        # Normalize to [0, 1]
        max_dist = np.max(dist_transform)
        if max_dist > 0:
            weights = dist_transform / max_dist
        else:
            weights = np.ones_like(dist_transform, dtype=np.float32)
        
        # Force garbage collection to free any intermediates
        gc.collect()
        
        return weights
    
    # Hand-trace verification for 2-image overlap:
    # Image 1: pixel value = 200, weight = 0.5
    # Image 2: pixel value = 200, weight = 0.5
    #
    # With float32 accumulators:
    # Tile contribution to canvas_sum = color * weight = 200 * 0.5 = 100.0
    # Tile contribution to weight_sum = weight = 0.5
    # After both tiles: canvas_sum = 100.0 + 100.0 = 200.0, weight_sum = 0.5 + 0.5 = 1.0
    # Final pixel = canvas_sum / weight_sum = 200.0 / 1.0 = 200.0
    #
    # This confirms the math is correct and the final pixel value will be 200.0,
    # which when cast to uint8 will be exactly 200.
    def _blend_tile_into_canvas(self, tile: np.ndarray, weights: np.ndarray,
                                  offset_x: int, offset_y: int,
                                  canvas_sum: np.ndarray, weight_sum: np.ndarray) -> None:
        """
        Blend a tile into the canvas accumulators.
        
        Args:
            tile: Tile image data
            weights: Weight array for the tile
            offset_x, offset_y: Tile offset on canvas
            canvas_sum: Accumulator for weighted sum
            weight_sum: Accumulator for weight sum
        """
        # Get canvas dimensions
        canvas_height, canvas_width = canvas_sum.shape[:2]
        
        # Calculate tile bounds on canvas
        x_start = max(0, offset_x)
        y_start = max(0, offset_y)
        x_end = min(canvas_width, offset_x + tile.shape[1])
        y_end = min(canvas_height, offset_y + tile.shape[0])
        
        # Calculate corresponding tile bounds
        tile_x_start = max(0, -offset_x)
        tile_y_start = max(0, -offset_y)
        tile_x_end = tile_x_start + (x_end - x_start)
        tile_y_end = tile_y_start + (y_end - y_start)
        
        # Skip if tile is completely outside canvas
        if x_start >= canvas_width or y_start >= canvas_height or x_end <= 0 or y_end <= 0:
            return
        
        # Extract valid regions
        tile_region = tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        weight_region = weights[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        
        # Update canvas accumulators
        if len(tile_region.shape) == 3:
            # Multi-band image
            for c in range(tile_region.shape[2]):
                # Compute per-tile contribution: tile_color_f32 * tile_weight_f32
                weighted_tile = tile_region[:, :, c].astype(np.float32)
                weighted_tile *= weight_region  # In-place multiplication
                
                # Add to canvas_sum directly (no scaling needed)
                canvas_sum[y_start:y_end, x_start:x_end, c] += weighted_tile
                
                # Flush to disk
                canvas_sum.flush()
                
                # Free intermediate
                del weighted_tile
        else:
            # Single-band image
            weighted_tile = tile_region.astype(np.float32)
            weighted_tile *= weight_region  # In-place multiplication
            
            # Add to canvas_sum directly (no scaling needed)
            canvas_sum[y_start:y_end, x_start:x_end, 0] += weighted_tile
            
            # Flush to disk
            canvas_sum.flush()
            
            # Free intermediate
            del weighted_tile
        
        # Add weight_region to weight_sum directly (no scaling needed)
        weight_sum[y_start:y_end, x_start:x_end] += weight_region
        
        # Flush to disk
        weight_sum.flush()
        
        # Free intermediate
        del weight_region
