"""
Orthophoto processing module for creating orthophotos using OpenDroneMap or GDAL.

This module provides functionality to create orthophotos from processed TIFF files
using either OpenDroneMap (preferred) or GDAL as a fallback.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any, List, Optional

from ..core.config import Config, get_config
from ..utils.logger import setup_logger
from ..utils.validators import validate_file_path
from ..utils.exceptions import ValidationError


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
        self.logger = setup_logger("OrthophotoProcessor")
        self.odm_path = self._find_odm_path()

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

            # Create orthophoto
            if self.odm_path:
                orthophoto_path = self._create_with_odm(tiff_paths, output_dir, processed_data)
            else:
                orthophoto_path = self._create_with_gdal(tiff_paths, output_dir)

            self.logger.info(f"[{len(tiff_paths)} files] Orthophoto created: {orthophoto_path}")
            return orthophoto_path

        except Exception as e:
            self.logger.error(f"Error creating orthophoto: {e}")
            raise

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
        try:
            # Create temporary directory for ODM
            with tempfile.TemporaryDirectory() as temp_dir:
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

                # Build ODM command
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

                # Run ODM
                self.logger.info(f"[{len(tiff_paths)} files] Running OpenDroneMap...")
                result = subprocess.run(
                    cmd,
                    cwd=self.odm_path,
                    capture_output=True,
                    text=True,
                    timeout=self.config.get(
                        "processing.odm_timeout", 3600
                    ),  # 1 hour default
                )

                if result.returncode != 0:
                    self.logger.error(f"ODM failed with error: {result.stderr}")
                    raise RuntimeError(f"OpenDroneMap error: {result.stderr}")

                # Copy results
                odm_results_dir = os.path.join(
                    project_dir, "odm_orthophoto", "odm_orthophoto.tif"
                )
                if os.path.exists(odm_results_dir):
                    output_path = os.path.join(output_dir, "orthophoto.tif")
                    self._copy_file(odm_results_dir, output_path)
                    return output_path
                else:
                    raise FileNotFoundError("ODM results not found")

        except subprocess.TimeoutExpired:
            self.logger.error("OpenDroneMap execution timeout exceeded")
            raise RuntimeError("OpenDroneMap timeout")
        except Exception as e:
            self.logger.error(f"Error working with OpenDroneMap: {e}")
            raise

    def _create_with_gdal(self, tiff_paths: List[str], output_dir: str) -> str:
        """
        Create orthophoto using GDAL (alternative method).

        Args:
            tiff_paths: List of paths to TIFF files
            output_dir: Directory to save results

        Returns:
            Path to the created orthophoto

        Raises:
            RuntimeError: If GDAL merge fails
        """
        try:
            self.logger.info(f"[{len(tiff_paths)} files] Creating orthophoto using GDAL")

            # Use gdal_merge.py to create mosaic
            output_path = os.path.join(output_dir, "orthophoto.tif")

            cmd = [
                "gdal_merge.py",
                "-o",
                output_path,
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LZW",
                "-co",
                "TILED=YES",
            ]
            cmd.extend(tiff_paths)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"GDAL merge error: {result.stderr}")
                raise RuntimeError(f"GDAL merge error: {result.stderr}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error creating orthophoto with GDAL: {e}")
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

            # GDAL command for optimization
            cmd = [
                "gdal_translate",
                orthophoto_path,
                output_path,
                "-co",
                "COMPRESS=LZW",
                "-co",
                "TILED=YES",
                "-co",
                "BIGTIFF=IF_NEEDED",
                "-co",
                "PREDICTOR=2",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Optimization error: {result.stderr}")
                raise RuntimeError(f"Optimization error: {result.stderr}")

            # Create pyramids
            pyramid_cmd = [
                "gdaladdo",
                "-r",
                "average",
                output_path,
                "2",
                "4",
                "8",
                "16",
            ]

            subprocess.run(pyramid_cmd, capture_output=True, text=True)

            self.logger.info(f"Orthophoto optimized: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error optimizing orthophoto: {e}")
            raise
