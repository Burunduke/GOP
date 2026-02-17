"""
Setup script for GOP - Гиперспектральная обработка и анализ растений
Версия 2.0.0 - Чистая научная архитектура без GUI
"""

from setuptools import setup, find_packages
import os

# Чтение версии из __init__.py
def get_version():
    version_file = os.path.join('src', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return '2.0.0'

# Чтение зависимостей из requirements.txt
def get_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f
                   if line.strip() and not line.startswith('#')
                   and not ('extra ==' in line or ';' in line)]
    return []

# Чтение длинного описания из README.md
def get_long_description():
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return """
GOP - Гиперспектральная обработка и анализ растений
===============================================

Научная библиотека для обработки гиперспектральных данных и анализа состояния растений
с использованием вегетационных индексов.

Основные возможности:
- Обработка гиперспектральных данных (BIL/HDR, TIFF)
- Создание ортофотопланов
- Расчет вегетационных индексов
- Сегментация изображений сверхвысокого разрешения
- Научный анализ и статистика
- CLI интерфейс для автоматизации

"""

setup(
    name='gop-hyperspectral',
    version=get_version(),
    author='Индыков Дмитрий Андреевич',
    author_email='indykovdm@example.com',
    description='Научная библиотека для гиперспектральной обработки и анализа растений',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/indykovdm/GOP',
    project_urls={
        'Bug Tracker': 'https://github.com/indykovdm/GOP/issues',
        'Documentation': 'https://github.com/indykovdm/GOP/wiki',
        'Source Code': 'https://github.com/indykovdm/GOP',
        'Scientific Papers': 'https://github.com/indykovdm/GOP/papers',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Agriculture',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Natural Language :: Russian',
    ],
    python_requires='>=3.9',
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.0,<8.0.0',
            'pytest-cov>=4.1.0,<5.0.0',
            'pytest-mock>=3.10.0,<4.0.0',
            'black>=23.0.0,<24.0.0',
            'flake8>=6.0.0,<7.0.0',
            'mypy>=1.0.0,<2.0.0',
            'sphinx>=7.0.0,<8.0.0',
            'sphinx-rtd-theme>=1.2.0,<2.0.0',
            'sphinx-autodoc-typehints>=1.24.0,<2.0.0',
            'pre-commit>=3.0.0,<4.0.0',
            'isort>=5.12.0,<6.0.0',
            'pylint>=2.17.0,<3.0.0',
        ],
        'gui': [
            'PyQt6>=6.4.0,<7.0.0',
        ],
        'gpu': [
            'torch>=2.0.0,<3.0.0',
            'torchvision>=0.15.0,<1.0.0',
            'cupy-cuda11x>=12.0.0,<13.0.0; sys_platform != "darwin"',
        ],
        'advanced': [
            'pywavelets>=1.4.0,<2.0.0',
            'xarray>=2023.1.0,<2024.0.0',
            'dask>=2023.1.0,<2024.0.0',
        ],
        'cloud': [
            'boto3>=1.26.0,<2.0.0',
            'google-cloud-storage>=2.7.0,<3.0.0',
            'azure-storage-blob>=12.14.0,<13.0.0',
        ],
        'all': [
            'PyQt6>=6.4.0,<7.0.0',
            'cupy-cuda11x>=12.0.0,<13.0.0; sys_platform != "darwin"',
            'pywavelets>=1.4.0,<2.0.0',
            'xarray>=2023.1.0,<2024.0.0',
            'dask>=2023.1.0,<2024.0.0',
            'boto3>=1.26.0,<2.0.0',
            'google-cloud-storage>=2.7.0,<3.0.0',
            'azure-storage-blob>=12.14.0,<13.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gop=cli:cli',
            'gop-process=cli:process',
            'gop-batch=cli:batch',
            'gop-info=cli:info',
            'gop-create-rgb=cli:create_rgb',
            'gop-list-indices=cli:list_indices',
            'gop-show-config=cli:show_config',
        ],
    },
    include_package_data=True,
    package_data={
        'gop': [
            'config/*.yaml',
            'data/*.json',
            'models/*.pth',
        ],
    },
    data_files=[
        ('config', ['config/config.yaml']),
        ('docs', [
            'docs/научная_работа_ортофотоплан_гиперспектральная_съемка.md',
            'docs/ВКР.md',
            'docs/Мага.md'
        ]),
        ('examples', [
            'examples/basic_processing.py',
            'examples/batch_processing.py',
            'examples/scientific_analysis.py',
        ]),
    ],
    zip_safe=False,
    keywords=[
        'hyperspectral',
        'remote sensing',
        'vegetation indices',
        'orthophoto',
        'agriculture',
        'precision farming',
        'image segmentation',
        'plant analysis',
        'scientific computing',
        'гиперспектральная съемка',
        'вегетационные индексы',
        'ортофотоплан',
        'сельское хозяйство',
        'точное земледелие',
        'научные вычисления',
    ],
    license='MIT',
    platforms=['any'],
    
    # Дополнительная метаданные для научных библиотек
    metadata_version='2.1',
    provides=['gop'],
    requires=['numpy', 'scipy', 'gdal', 'scikit-learn'],
    obsoletes=[],
    
    # Информация о совместимости
    setup_requires=['wheel>=0.37.0'],
    python_requires='>=3.9',
    
    # Тестирование
    test_suite='tests',
    tests_require=[
        'pytest>=7.4.0,<8.0.0',
        'pytest-cov>=4.1.0,<5.0.0',
        'pytest-mock>=3.10.0,<4.0.0',
        'pytest-xdist>=3.0.0,<4.0.0',
    ],
    
    # Документация
    command_options={
        'build_sphinx': {
            'project': ('setup.py', 'GOP - Гиперспектральная обработка'),
            'version': ('setup.py', get_version()),
            'release': ('setup.py', get_version()),
        }
    },
)