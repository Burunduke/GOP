"""
Тесты для файловых утилит
"""

import unittest
import tempfile
import os
import shutil
from src.utils.file_utils import (
    ensure_dir, get_file_extension, validate_file_path,
    copy_file, move_file, delete_file, get_file_size,
    find_files, create_backup
)


class TestFileUtils(unittest.TestCase):
    """Тесты файловых утилит"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Создание тестовых файлов
        self.test_file_path = os.path.join(self.temp_dir, 'test_file.txt')
        self.test_file_path2 = os.path.join(self.temp_dir, 'test_file2.jpg')
        
        with open(self.test_file_path, 'w') as f:
            f.write('Test content')
        
        with open(self.test_file_path2, 'w') as f:
            f.write('Test image content')
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_dir_new_directory(self):
        """Тест создания новой директории"""
        new_dir = os.path.join(self.temp_dir, 'new_directory')
        
        ensure_dir(new_dir)
        
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))
    
    def test_ensure_dir_existing_directory(self):
        """Тест обработки существующей директории"""
        # Директория уже существует
        ensure_dir(self.temp_dir)
        
        # Не должно вызывать ошибок
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_ensure_dir_nested_directories(self):
        """Тест создания вложенных директорий"""
        nested_dir = os.path.join(self.temp_dir, 'level1', 'level2', 'level3')
        
        ensure_dir(nested_dir)
        
        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.isdir(nested_dir))
    
    def test_get_file_extension(self):
        """Тест получения расширения файла"""
        self.assertEqual(get_file_extension(self.test_file_path), '.txt')
        self.assertEqual(get_file_extension(self.test_file_path2), '.jpg')
        self.assertEqual(get_file_extension('file_without_extension'), '')
        self.assertEqual(get_file_extension('multiple.dots.in.name.txt'), '.txt')
    
    def test_validate_file_path_exists(self):
        """Тест валидации существующего файла"""
        result = validate_file_path(self.test_file_path)
        self.assertTrue(result)
        
        result = validate_file_path(self.test_file_path, extensions=['.txt'])
        self.assertTrue(result)
    
    def test_validate_file_path_not_exists(self):
        """Тест валидации несуществующего файла"""
        non_existent_path = os.path.join(self.temp_dir, 'non_existent.txt')
        
        result = validate_file_path(non_existent_path)
        self.assertFalse(result)
    
    def test_validate_file_path_wrong_extension(self):
        """Тест валидации файла с неверным расширением"""
        result = validate_file_path(self.test_file_path, extensions=['.jpg', '.png'])
        self.assertFalse(result)
    
    def test_copy_file(self):
        """Тест копирования файла"""
        dst_path = os.path.join(self.temp_dir, 'copied_file.txt')
        
        copy_file(self.test_file_path, dst_path)
        
        self.assertTrue(os.path.exists(dst_path))
        
        # Проверка содержимого
        with open(self.test_file_path, 'r') as src, open(dst_path, 'r') as dst:
            self.assertEqual(src.read(), dst.read())
    
    def test_copy_file_nonexistent_source(self):
        """Тест копирования несуществующего файла"""
        src_path = os.path.join(self.temp_dir, 'non_existent.txt')
        dst_path = os.path.join(self.temp_dir, 'copied_file.txt')
        
        with self.assertRaises(FileNotFoundError):
            copy_file(src_path, dst_path)
    
    def test_move_file(self):
        """Тест перемещения файла"""
        dst_path = os.path.join(self.temp_dir, 'moved_file.txt')
        
        move_file(self.test_file_path, dst_path)
        
        self.assertFalse(os.path.exists(self.test_file_path))
        self.assertTrue(os.path.exists(dst_path))
        
        # Проверка содержимого
        with open(dst_path, 'r') as f:
            self.assertEqual(f.read(), 'Test content')
    
    def test_move_file_nonexistent_source(self):
        """Тест перемещения несуществующего файла"""
        src_path = os.path.join(self.temp_dir, 'non_existent.txt')
        dst_path = os.path.join(self.temp_dir, 'moved_file.txt')
        
        with self.assertRaises(FileNotFoundError):
            move_file(src_path, dst_path)
    
    def test_delete_file(self):
        """Тест удаления файла"""
        delete_file(self.test_file_path)
        
        self.assertFalse(os.path.exists(self.test_file_path))
    
    def test_delete_file_nonexistent(self):
        """Тест удаления несуществующего файла"""
        non_existent_path = os.path.join(self.temp_dir, 'non_existent.txt')
        
        # Не должно вызывать ошибок
        delete_file(non_existent_path)
    
    def test_get_file_size(self):
        """Тест получения размера файла"""
        size = get_file_size(self.test_file_path)
        
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)
        self.assertEqual(size, len('Test content'))
    
    def test_get_file_size_nonexistent(self):
        """Тест получения размера несуществующего файла"""
        non_existent_path = os.path.join(self.temp_dir, 'non_existent.txt')
        
        with self.assertRaises(FileNotFoundError):
            get_file_size(non_existent_path)
    
    def test_find_files_default_pattern(self):
        """Тест поиска файлов с шаблоном по умолчанию"""
        files = find_files(self.temp_dir)
        
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)
        self.assertIn(self.test_file_path, files)
        self.assertIn(self.test_file_path2, files)
    
    def test_find_files_custom_pattern(self):
        """Тест поиска файлов с пользовательским шаблоном"""
        txt_files = find_files(self.temp_dir, pattern='*.txt')
        
        self.assertIsInstance(txt_files, list)
        self.assertEqual(len(txt_files), 1)
        self.assertIn(self.test_file_path, txt_files)
        self.assertNotIn(self.test_file_path2, txt_files)
    
    def test_find_files_no_matches(self):
        """Тест поиска файлов без совпадений"""
        pdf_files = find_files(self.temp_dir, pattern='*.pdf')
        
        self.assertIsInstance(pdf_files, list)
        self.assertEqual(len(pdf_files), 0)
    
    def test_find_files_nonexistent_directory(self):
        """Тест поиска файлов в несуществующей директории"""
        non_existent_dir = os.path.join(self.temp_dir, 'non_existent')
        
        with self.assertRaises(FileNotFoundError):
            find_files(non_existent_dir)
    
    def test_create_backup(self):
        """Тест создания резервной копии"""
        backup_path = create_backup(self.test_file_path)
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith('.bak'))
        
        # Проверка содержимого
        with open(self.test_file_path, 'r') as src, open(backup_path, 'r') as backup:
            self.assertEqual(src.read(), backup.read())
    
    def test_create_backup_custom_suffix(self):
        """Тест создания резервной копии с пользовательским суффиксом"""
        backup_path = create_backup(self.test_file_path, backup_suffix='.backup')
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith('.backup'))
    
    def test_create_backup_nonexistent_file(self):
        """Тест создания резервной копии несуществующего файла"""
        non_existent_path = os.path.join(self.temp_dir, 'non_existent.txt')
        
        with self.assertRaises(FileNotFoundError):
            create_backup(non_existent_path)
    
    def test_create_backup_in_nonexistent_directory(self):
        """Тест создания резервной копии в несуществующей директории"""
        # Создание файла в поддиректории
        subdir = os.path.join(self.temp_dir, 'subdir')
        ensure_dir(subdir)
        
        file_in_subdir = os.path.join(subdir, 'file.txt')
        with open(file_in_subdir, 'w') as f:
            f.write('Content')
        
        # Попытка создать резервную копию в несуществующей директории
        nonexistent_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        with self.assertRaises(FileNotFoundError):
            create_backup(file_in_subdir, backup_dir=nonexistent_dir)


if __name__ == '__main__':
    unittest.main()