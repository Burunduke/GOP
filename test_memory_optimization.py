#!/usr/bin/env python3
"""
Test script to verify memory optimization for large file uploads.
"""

import os
import tempfile
import psutil
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_usage():
    """Test memory usage with simulated large file upload."""
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test file upload utilities
    from gui.utils.file_upload_utils import FileUploadManager
    from gui.utils.memory_monitor import MemoryMonitor
    
    # Test with multiple file sizes
    test_sizes = [
        (50, "50MB file"),  # Within limit
        (150, "150MB file"),  # Above memory limit but within file limit
        (250, "250MB file")   # The problematic size
    ]
    
    results = []
    
    for size_mb, description in test_sizes:
        print(f"\nTesting {description}...")
        
        # Create test content
        test_content = "A" * size_mb * 1024 * 1024  # size_mb MB of data
        
        # Encode to base64 (simulating browser upload)
        import base64
        encoded_content = base64.b64encode(test_content.encode()).decode()
        
        # Create upload content string
        upload_content = f"data:application/octet-stream;base64,{encoded_content}"
        
        print(f"Testing {size_mb}MB file upload with memory monitoring...")
        
        # Monitor memory during upload
        monitor = MemoryMonitor()
        monitor.log_memory_usage(f"before {size_mb}MB upload test")
        
        try:
            upload_manager = FileUploadManager()
            
            # This should use streaming and not load the entire file into memory
            temp_path, file_size, checksum = upload_manager.save_uploaded_content_to_temp_file(
                upload_content, f"test_{size_mb}mb_file.dat"
            )
            
            print(f"File processed successfully: {file_size / (1024*1024):.1f} MB")
            print(f"Checksum: {checksum}")
            
            # Check memory usage after upload
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")
            
            # Clean up
            upload_manager.cleanup_temp_file(temp_path)
            
            # Memory increase should be reasonable (less than 2x the file size)
            max_expected_increase = min(size_mb * 2, 200)  # Cap at 200MB
            if memory_increase < max_expected_increase:
                print(f"✅ Memory optimization test PASSED for {size_mb}MB file - memory usage: {memory_increase:.1f} MB")
                results.append(True)
            else:
                print(f"⚠️ Memory optimization test WARNING for {size_mb}MB file - high memory usage: {memory_increase:.1f} MB (expected < {max_expected_increase} MB)")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Test failed for {size_mb}MB file with error: {e}")
            results.append(False)
    
    return all(results)

def test_configuration():
    """Test configuration settings."""
    from gui.config import config
    
    app_config = config['default']
    
    print("\nConfiguration settings:")
    print(f"MAX_FILE_SIZE: {app_config.MAX_FILE_SIZE / (1024*1024*1024):.1f} GB")
    print(f"MAX_MEMORY_FILE_SIZE: {app_config.MAX_MEMORY_FILE_SIZE / (1024*1024):.1f} MB")
    print(f"STREAMING_CHUNK_SIZE: {app_config.STREAMING_CHUNK_SIZE} bytes")
    
    return True

if __name__ == "__main__":
    print("Testing memory optimization for large file uploads...\n")
    
    # Test configuration
    test_configuration()
    
    # Test memory usage
    success = test_memory_usage()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("The memory optimization for large file uploads is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)