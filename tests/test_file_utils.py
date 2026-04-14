"""
Tests for file utilities
"""

import unittest
import tempfile
import os
import shutil
from typing import List
from src.utils.file_utils import (
    ensure_dir,
    get_file_extension,
    validate_file_path,
    copy_file,
    move_file,
    delete_file,
    get_file_size,
    find_files,
    create_backup,
)


class TestFileUtils(unittest.TestCase):
    """Tests for file utilities"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        self.test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        self.test_file_path2 = os.path.join(self.temp_dir, "test_file2.jpg")

        with open(self.test_file_path, "w") as f:
            f.write("Test content")

        with open(self.test_file_path2, "w") as f:
            f.write("Test image content")

    def tearDown(self) -> None:
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_dir_new_directory(self) -> None:
        """Test creating new directory"""
        new_dir = os.path.join(self.temp_dir, "new_directory")

        ensure_dir(new_dir)

        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))

    def test_ensure_dir_existing_directory(self) -> None:
        """Test handling existing directory"""
        # Directory already exists
        ensure_dir(self.temp_dir)

        # Should not raise errors
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_ensure_dir_nested_directories(self) -> None:
        """Test creating nested directories"""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")

        ensure_dir(nested_dir)

        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.isdir(nested_dir))

    def test_get_file_extension(self) -> None:
        """Test getting file extension"""
        self.assertEqual(get_file_extension(self.test_file_path), "txt")
        self.assertEqual(get_file_extension(self.test_file_path2), "jpg")
        self.assertEqual(get_file_extension("file_without_extension"), "")
        self.assertEqual(get_file_extension("multiple.dots.in.name.txt"), "txt")

    def test_validate_file_path_exists(self) -> None:
        """Test validating existing file"""
        result = validate_file_path(self.test_file_path)
        self.assertTrue(result)

        result = validate_file_path(self.test_file_path, extensions=[".txt"])
        self.assertTrue(result)

    def test_validate_file_path_not_exists(self) -> None:
        """Test validating non-existent file"""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.txt")

        result = validate_file_path(non_existent_path)
        self.assertFalse(result)

    def test_validate_file_path_wrong_extension(self) -> None:
        """Test validating file with wrong extension"""
        result = validate_file_path(self.test_file_path, extensions=[".jpg", ".png"])
        self.assertFalse(result)

    def test_copy_file(self) -> None:
        """Test copying file"""
        dst_path = os.path.join(self.temp_dir, "copied_file.txt")

        copy_file(self.test_file_path, dst_path)

        self.assertTrue(os.path.exists(dst_path))

        # Check content
        with open(self.test_file_path, "r") as src, open(dst_path, "r") as dst:
            self.assertEqual(src.read(), dst.read())

    def test_copy_file_nonexistent_source(self) -> None:
        """Test copying non-existent file"""
        src_path = os.path.join(self.temp_dir, "non_existent.txt")
        dst_path = os.path.join(self.temp_dir, "copied_file.txt")

        with self.assertRaises(FileNotFoundError):
            copy_file(src_path, dst_path)

    def test_move_file(self) -> None:
        """Test moving file"""
        dst_path = os.path.join(self.temp_dir, "moved_file.txt")

        move_file(self.test_file_path, dst_path)

        self.assertFalse(os.path.exists(self.test_file_path))
        self.assertTrue(os.path.exists(dst_path))

        # Check content
        with open(dst_path, "r") as f:
            self.assertEqual(f.read(), "Test content")

    def test_move_file_nonexistent_source(self) -> None:
        """Test moving non-existent file"""
        src_path = os.path.join(self.temp_dir, "non_existent.txt")
        dst_path = os.path.join(self.temp_dir, "moved_file.txt")

        with self.assertRaises(FileNotFoundError):
            move_file(src_path, dst_path)

    def test_delete_file(self) -> None:
        """Test deleting file"""
        delete_file(self.test_file_path)

        self.assertFalse(os.path.exists(self.test_file_path))

    def test_delete_file_nonexistent(self) -> None:
        """Test deleting non-existent file"""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.txt")

        # Should not raise errors
        delete_file(non_existent_path)

    def test_get_file_size(self) -> None:
        """Test getting file size"""
        size = get_file_size(self.test_file_path)

        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)
        self.assertEqual(size, len("Test content"))

    def test_get_file_size_nonexistent(self) -> None:
        """Test getting size of non-existent file"""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.txt")

        with self.assertRaises(FileNotFoundError):
            get_file_size(non_existent_path)

    def test_find_files_default_pattern(self) -> None:
        """Test finding files with default pattern"""
        files = find_files(self.temp_dir)

        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)
        self.assertIn(self.test_file_path, files)
        self.assertIn(self.test_file_path2, files)

    def test_find_files_custom_pattern(self) -> None:
        """Test finding files with custom pattern"""
        txt_files = find_files(self.temp_dir, pattern="*.txt")

        self.assertIsInstance(txt_files, list)
        self.assertEqual(len(txt_files), 1)
        self.assertIn(self.test_file_path, txt_files)
        self.assertNotIn(self.test_file_path2, txt_files)

    def test_find_files_no_matches(self) -> None:
        """Test finding files with no matches"""
        pdf_files = find_files(self.temp_dir, pattern="*.pdf")

        self.assertIsInstance(pdf_files, list)
        self.assertEqual(len(pdf_files), 0)

    def test_find_files_nonexistent_directory(self) -> None:
        """Test finding files in non-existent directory"""
        non_existent_dir = os.path.join(self.temp_dir, "non_existent")

        with self.assertRaises(FileNotFoundError):
            find_files(non_existent_dir)

    def test_create_backup(self) -> None:
        """Test creating backup"""
        backup_path = create_backup(self.test_file_path)

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith(".bak"))

        # Check content
        with open(self.test_file_path, "r") as src, open(backup_path, "r") as backup:
            self.assertEqual(src.read(), backup.read())

    def test_create_backup_custom_suffix(self) -> None:
        """Test creating backup with custom suffix"""
        backup_path = create_backup(self.test_file_path, backup_suffix=".backup")

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith(".backup"))

    def test_create_backup_nonexistent_file(self) -> None:
        """Test creating backup of non-existent file"""
        non_existent_path = os.path.join(self.temp_dir, "non_existent.txt")

        with self.assertRaises(FileNotFoundError):
            create_backup(non_existent_path)

    def test_create_backup_in_nonexistent_directory(self) -> None:
        """Test creating backup in non-existent directory"""
        # Create file in subdirectory
        subdir = os.path.join(self.temp_dir, "subdir")
        ensure_dir(subdir)

        file_in_subdir = os.path.join(subdir, "file.txt")
        with open(file_in_subdir, "w") as f:
            f.write("Content")

        # Attempt to create backup in non-existent directory
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")

        # Test that backup is created in the same directory
        backup_path = create_backup(file_in_subdir)
        self.assertTrue(os.path.exists(backup_path))
        self.assertEqual(backup_path, file_in_subdir + ".bak")


if __name__ == "__main__":
    unittest.main()
