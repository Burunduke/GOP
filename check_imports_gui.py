#!/usr/bin/env python3
"""
Script to check for unused imports in Python files
"""
import ast
import os
import sys
from pathlib import Path

def get_imports(file_path):
    """Extract all imports from a Python file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set(), set()
    
    imports = set()
    from_imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                from_imports.add(node.module.split('.')[0])
    
    return imports, from_imports

def get_used_names(file_path):
    """Extract all names used in a Python file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set()
    
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                names.add(node.value.id)
    
    return names

def check_file_for_unused_imports(file_path):
    """Check a single file for unused imports"""
    imports, from_imports = get_imports(file_path)
    used_names = get_used_names(file_path)
    
    all_imports = imports.union(from_imports)
    unused = all_imports - used_names
    
    return unused

def main():
    """Main function"""
    src_dir = Path("gui")
    if not src_dir.exists():
        print("gui directory not found")
        return 1
    
    unused_imports_found = False
    
    for py_file in src_dir.rglob("*.py"):
        unused = check_file_for_unused_imports(py_file)
        if unused:
            print(f"{py_file}: Unused imports: {', '.join(unused)}")
            unused_imports_found = True
    
    if not unused_imports_found:
        print("No unused imports found in gui/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())