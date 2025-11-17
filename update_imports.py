import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update imports to match the new structure
    updated_content = content
    
    # Update chaosnet imports
    updated_content = re.sub(
        r'from chaosnet\.(?!core|io|sim|training|utils)(\w+)', 
        r'from chaosnet.\1', 
        updated_content
    )
    
    # Update relative imports if any
    updated_content = re.sub(
        r'from \.(\w+)', 
        r'from chaosnet.\1', 
        updated_content
    )
    
    # Update imports from root directory files
    root_imports = [
        'train_mnist_sleep', 'train_language', 'train_addition',
        'multimodel_trainin', 'plot_training', 'generate_visualizations'
    ]
    for module in root_imports:
        updated_content = updated_content.replace(
            f'import {module}',
            f'from examples import {module}'
        )
    
    # Only write back if changes were made
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    return False

def main():
    repo_root = Path(__file__).parent
    updated_files = 0
    
    # Process all Python files in the repository
    for py_file in repo_root.rglob('*.py'):
        if str(py_file).endswith('update_imports.py'):
            continue  # Skip this script
            
        try:
            if update_imports_in_file(py_file):
                print(f"Updated imports in: {py_file.relative_to(repo_root)}")
                updated_files += 1
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    print(f"\nUpdated imports in {updated_files} files.")

if __name__ == "__main__":
    main()
