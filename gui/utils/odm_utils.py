"""
ODM utility functions for checking ODM availability and status.
"""

import subprocess
import shutil
from typing import Tuple


def check_odm_status(image_count: int) -> Tuple[bool, str]:
    """
    Check if ODM is available and can be used for stitching.
    
    Args:
        image_count: Number of images currently selected/loaded
        
    Returns:
        Tuple of (is_available, human_readable_reason)
    """
    # Check minimum image count first (takes precedence)
    if image_count < 3:
        return False, f"ODM disabled: requires at least 3 overlapping images (you have {image_count})."
    
    # Check Docker availability
    try:
        docker_result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=3
        )
        docker_available = docker_result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        docker_available = False
    
    if docker_available:
        # Check if ODM Docker image is available
        try:
            # First check if docker info works
            info_result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if info_result.returncode == 0:
                # Check if the image is available locally
                image_result = subprocess.run(
                    ["docker", "images", "opendronemap/odm", "--format", "{{.Repository}}"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if image_result.returncode == 0 and "opendronemap/odm" in image_result.stdout:
                    return True, "ODM ready (Docker image detected)."
                else:
                    # Docker is available but image is missing
                    # Mark as available with note that image will be pulled on first run
                    return True, "ODM ready (Docker detected, image will be pulled on first run)."
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    # Check for native ODM installation
    if shutil.which("odm"):
        return True, "ODM ready (native installation detected)."
    
    # If we get here, ODM is not available
    if docker_available:
        return False, "ODM disabled: Docker is available but ODM image is missing. Run `docker pull opendronemap/odm`."
    else:
        return False, "ODM disabled: Docker is not available. Install Docker Desktop or pull the `opendronemap/odm` image."