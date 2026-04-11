# Установка и настройка GOP

Это руководство описывает установку и настройку проекта GOP (Гиперспектральная обработка и анализ растений).

## 🚀 Быстрая установка

### Предварительные требования

- **Python 3.9+** (рекомендуется 3.10+)
- **8GB RAM** (рекомендуется 16GB)
- **2GB свободного места** на диске

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/indykovdm/GOP.git
cd GOP
```

### Шаг 2: Установка зависимостей

```bash
# Базовая установка (рекомендуется)
pip install -r requirements.txt

# Для разработки
pip install -r requirements-dev.txt

# С поддержкой GPU (только Linux/Windows)
pip install -r requirements-gpu.txt
```

### Шаг 3: Проверка установки

```bash
python -c "import src.core.pipeline; print('GOP успешно установлен')"
```

## 📦 Варианты установки

### Через pip (рекомендуется)

```bash
# Базовая установка
pip install .

# С дополнительными компонентами
pip install .[dev]    # Для разработки
pip install .[gui]    # С поддержкой GUI
pip install .[gpu]    # С поддержкой GPU
pip install .[all]    # Все компоненты
```

### Через requirements файлы

| Файл | Назначение | Команда |
|------|------------|---------|
| `requirements.txt` | Основные зависимости | `pip install -r requirements.txt` |
| `requirements-dev.txt` | Разработка и тестирование | `pip install -r requirements-dev.txt` |
| `requirements-gpu.txt` | GPU ускорение | `pip install -r requirements-gpu.txt` |
| `requirements-all.txt` | Все зависимости | `pip install -r requirements-all.txt` |

## 🔧 Системные требования

### Минимальные
- **Python**: 3.9+
- **Память**: 8GB RAM
- **Диск**: 2GB свободного места
- **ОС**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Рекомендуемые
- **Python**: 3.10+
- **Память**: 16GB RAM
- **Диск**: 5GB свободного места
- **GPU**: NVIDIA с поддержкой CUDA (для ускорения)

## 📊 Основные зависимости

### Научные вычисления
- `numpy` - Основные операции с массивами
- `scipy` - Научные алгоритмы
- `matplotlib` - Визуализация данных

### Обработка изображений
- `opencv-python` - Компьютерное зрение
- `scikit-image` - Обработка изображений
- `spectral` - Гиперспектральные данные

### Геопространственные данные
- `gdal` - Геоданные
- `rasterio` - Растровые данные
- `geopandas` - Векторные данные

### Визуализация
- `plotly` - Интерактивная визуализация
- `seaborn` - Статистическая визуализация

## 🛠️ Устранение проблем

### Проблемы с GDAL

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Windows
# Скачайте wheel файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
pip install GDAL-3.7.0-cp310-cp310-win_amd64.whl
```

### Проблемы с PyTorch

```bash
# CPU версия (рекомендуется для начала)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU версия
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Проблемы с виртуальным окружением

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Linux/macOS)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate

# Установка в виртуальном окружении
pip install -r requirements.txt
```

## 🔍 Проверка установки

### Тест основных компонентов

```python
# Проверка импорта основных модулей
try:
    import numpy as np
    import cv2
    import rasterio
    from src.core.pipeline import Pipeline
    print("✅ Все основные модули загружены успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
```

### Запуск тестового примера

```bash
python examples/basic_processing.py
```

## 🔄 Обновление

### Обновление зависимостей

```bash
pip install --upgrade -r requirements.txt
```

### Проверка устаревших пакетов

```bash
pip list --outdated
```

## 📋 Следующие шаги

После успешной установки:

1. **Ознакомьтесь с [руководством пользователя](USER_GUIDE.md)** - как использовать GOP
2. **Попробуйте [примеры](examples/)** - готовые сценарии использования
3. **Изучите [архитектуру](ARCHITECTURE.md)** - понимание системы
4. **Начните с [быстрого старта](README.md#быстрый-старт)** - первое использование

## ❓ Часто задаваемые вопросы

**Q: Какая версия Python рекомендуется?**
A: Python 3.10+ для лучшей производительности и совместимости.

**Q: Нужен ли GPU для работы?**
A: Нет, GOP работает на CPU, но GPU ускоряет обработку больших данных.

**Q: Как установить на сервер без GUI?**
A: Используйте `requirements.txt` - он не включает GUI зависимости.

**Q: Поддерживается ли Windows?**
A: Да, но для некоторых зависимостей могут потребоваться дополнительные шаги.

---

*Для получения дополнительной помощи обратитесь к [руководству пользователя](USER_GUIDE.md) или создайте issue на GitHub.*