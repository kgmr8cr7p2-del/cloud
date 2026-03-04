# Screen Aim Assistant — Инструкция по запуску

## Системные требования

- **ОС**: Windows 10 / 11 (64-bit)
- **Python**: 3.10+
- **GPU** (опционально): NVIDIA с поддержкой CUDA 11.8+
- **RAM**: минимум 4 ГБ, рекомендуется 8+ ГБ

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd tgbot
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Установить зависимости

**Базовая установка (CPU):**
```bash
pip install -r requirements.txt
```

**С поддержкой GPU (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install onnxruntime-gpu
```

**С TensorRT (опционально):**
```bash
pip install tensorrt pycuda
```

### 4. Скачать модель

По умолчанию используется `yolov8n.pt`. При первом запуске ultralytics
загрузит её автоматически. Или скачайте заранее:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Варианты моделей (по скорости/точности):
- `yolov8n.pt` — nano (самая быстрая)
- `yolov8s.pt` — small
- `yolov8m.pt` — medium
- `yolov8l.pt` — large

## Запуск

```bash
python main.py
```

## Запуск тестов

```bash
pip install pytest
pytest tests/ -v
```

## Экспорт модели

### .pt → ONNX
Из GUI: вкладка "Экспорт" → "Экспортировать в ONNX"

Или вручную:
```python
from inference.onnx_export import export_to_onnx
export_to_onnx("yolov8n.pt", opset=17, input_size=640)
```

### ONNX → TensorRT Engine
Из GUI: вкладка "Экспорт" → "Собрать TensorRT Engine"

Или вручную:
```python
from inference.tensorrt_builder import build_engine
build_engine("yolov8n.onnx", fp16=True, workspace_mb=4096)
```

## Сборка через PyInstaller

```bash
pip install pyinstaller
pyinstaller screen_aim.spec
```

Готовый exe будет в `dist/ScreenAimAssistant/`.

## Структура проекта

```
├── main.py              # Точка входа, оркестрация pipeline
├── config/              # JSON-конфигурация, версионирование, thread-safe
├── app_logging/         # Файловый лог + ring buffer для GUI
├── capture/             # Захват ROI (dxcam / mss)
├── inference/           # Torch, ONNX, TensorRT бэкенды
├── detection/           # Постпроцессинг: NMS, фильтры
├── tracking/            # Удержание цели (lock + association)
├── aim/                 # Наведение мыши (SendInput, speed profiles)
├── filters/             # One Euro Filter, EMA
├── overlay/             # Win32 прозрачный оверлей
├── gui/                 # DearPyGui интерфейс
├── telemetry/           # FPS / latency (current, avg, p95)
├── tests/               # Юнит-тесты
├── requirements.txt
├── pyproject.toml
└── screen_aim.spec      # PyInstaller спецификация
```

## Горячие клавиши (по умолчанию)

| Клавиша | Действие |
|---------|----------|
| F6 | Вкл/выкл управления мышью |
| F7 | Вкл/выкл оверлея |
| F8 | Пауза/продолжение захвата |
| X2 (мышь) | Удержание для активации наведения |

Все горячие клавиши настраиваются во вкладке "Горячие клавиши".

## Решение проблем

**"dxcam failed, falling back to mss"**
— dxcam не работает в вашей конфигурации. Переключите метод захвата на `mss` в GUI.

**"CUDA not available"**
— Установите версию PyTorch с CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**"TensorRT not available"**
— Установите `pip install tensorrt pycuda`. Требуется NVIDIA GPU.

**Низкий FPS инференса**
— Используйте модель `yolov8n.pt`, уменьшите входное разрешение до 320,
  включите FP16, или соберите TensorRT engine.
