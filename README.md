# Проект Sample DeepSeek R1 LLM & WhisperAI

Этот репозиторий содержит **демонстрационный проект**, показывающий использование:
- **Whisper LoRA Fine-Tuned** для **автоматического распознавания речи (ASR) на русском языке**.
- **DeepSeek Meeting Summary** для **суммаризации расшифровок встреч**.

В проект включены **небольшой образец датасета**, **видео для тестирования** и **эксперименты по дообучению и инференсу моделей**.

---

## 📂 Структура директории
```
meeting_key_points_ai/
│── sample_datasets/
│   ├── sample_golos/                  # 10 train + 2 test примеров из датасета Golos
│   ├── sample_meeting/                # 10 примеров из кастомного датасета Meeting
│
│── examples/
│   ├── meeting_video.mp4              # Видеофайл для тестирования моделей
│
│── sample_implementation.ipynb        # Запускает модели на meeting_video.mp4
│── sample_experiments.ipynb           # Дообучает модели на sample_datasets
│
│── real_results/
│   ├── notebooks/
│       ├── implementation.ipynb       # Результаты sample_implementation.ipynb
│       ├── experiments.ipynb          # Результаты sample_experiments.ipynb
│   ├── outputs/                       # CSV-отчёты по экспериментам
│
│── README.md                          # Документация проекта
│── requirements.txt                   # Зависимости
│── .gitignore                         # Исключает ненужные файлы
```

---

## 📊 Датасеты
### **1. Датасет Golos (подмножество)**
- **Источник:** [Golos GitHub Repository](https://github.com/salute-developers/golos/tree/master/golos#golos-dataset)
- **Содержание:** 10 обучающих примеров + 2 тестовых примера

### **2. Датасет Meeting (кастомный подмножество)**
- **Основан на:** **AMI Corpus** & **ICSI Corpus**
- **Содержание:** 10 примеров стенограмм встреч

---

## 🎥 Видеофайл для тестирования
Файл **`examples/meeting_video.mp4`** предоставлен для **тестирования возможностей моделей по распознаванию речи и суммаризации**.

---

## 🚀 Использование моделей
### **Whisper LoRA Fine-Tuned**
- **Модель:** [UDZH/whisper-small-lora-finetuned-ru](https://huggingface.co/UDZH/whisper-small-lora-finetuned-ru)
- **Пример использования:**
```python
from huggingface_hub import hf_hub_download
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Загрузка модели Whisper LoRA
repo_id = "UDZH/whisper-small-lora-finetuned-ru"
lora_weights_path = hf_hub_download(repo_id=repo_id, filename="whisper_lora_weights.pth")

# Загрузка базовой модели и применение LoRA весов
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")

lora_weights = torch.load(lora_weights_path, map_location="cuda")
model.load_state_dict(lora_weights, strict=False)
```

### **DeepSeek Meeting Summary**
- **Модель:** [UDZH/deepseek-meeting-summary](https://huggingface.co/UDZH/deepseek-meeting-summary)
- **Пример использования:**
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Загрузка модели и токенизатора
model = AutoModelForSeq2SeqLM.from_pretrained("UDZH/deepseek-meeting-summary")
tokenizer = AutoTokenizer.from_pretrained("UDZH/deepseek-meeting-summary")

# Пример суммаризации
text = "Пример стенограммы встречи."
inputs = tokenizer(text, return_tensors="pt")
summary = model.generate(**inputs)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
```

---

## 📓 Ноутбуки
### **1. `sample_implementation.ipynb`**
- **Цель:** Запуск моделей на `meeting_video.mp4`.
- **Этапы:**
  1. Извлечение аудио из видео
  2. Запуск **Whisper LoRA** для распознавания речи
  3. Запуск **DeepSeek Summary** для суммаризации
  4. Отображение результатов

### **2. `sample_experiments.ipynb`**
- **Цель:** Дообучение моделей на **sample_datasets/**.
- **Этапы:**
  1. Загрузка подмножества **Golos** и **Meeting**
  2. Дообучение **Whisper LoRA** на Golos
  3. Дообучение **DeepSeek Summary** на стенограммах встреч
  4. Оценка результатов

---

## 📈 Результаты экспериментов
Все реальные эксперименты и выводы хранятся в `real_results/`.
- **`implementation.ipynb`** → Логи и расшифровки работы на полноразмерных примерах
- **`experiments.ipynb`** → Логи обучения моделей
- **`outputs/`** → CSV-отчёты по производительности моделей

---

## 🔧 Установка
### **1. Установите зависимости**
```bash
pip install -r requirements.txt
```
Или установите их напрямую в ноутбуке.

---

## 📜 Лицензия
Этот репозиторий распространяется под лицензией **MIT License**.

---

## 👥 Авторы
- **@UDZH** – Дообучение моделей и подготовка датасетов
- **Uliana Dzhumok** – Настройка репозитория и документации (Опционально)

---

## ⭐ Благодарности
Отдельное спасибо:
- **Hugging Face** за хостинг моделей
- **OpenAI** за Whisper
- **DeepSeek** за их модели
- **Создателям датасета Golos**
- **AMI & ICSI Corpus поставщикам**
- **Unsloth** за инструменты дообучения

