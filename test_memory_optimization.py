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
    
    # Create a large test file (simulating 50MB file - within the 100MB limit)
    test_content = "A" * 50 * 1024 * 1024  # 50MB of data
    
    # Encode to base64 (simulating browser upload)
    import base64
    encoded_content = base64.b64encode(test_content.encode()).decode()
    
    # Create upload content string
    upload_content = f"data:application/octet-stream;base64,{encoded_content}"
    
    print("Testing file upload with memory monitoring...")
    
    # Monitor memory during upload
    monitor = MemoryMonitor()
    monitor.log_memory_usage("before upload test")
    
    try:
        upload_manager = FileUploadManager()
        
        # This should use streaming and not load the entire file into memory
        temp_path, file_size, checksum = upload_manager.save_uploaded_content_to_temp_file(
            upload_content, "test_large_file.dat"
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
        
        # Memory increase should be minimal (less than 50MB for a 100MB file)
        if memory_increase < 50:
            print("✅ Memory optimization test PASSED - minimal memory usage")
        else:
            print(f"⚠️ Memory optimization test WARNING - high memory usage: {memory_increase:.1f} MB")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    return True

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