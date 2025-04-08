# Дообучение Llama-4-Maverick с использованием MoE Adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Этот репозиторий содержит код для эффективного дообучения модели Llama-4-Maverick с использованием метода MoE Adapter (Mixture of Experts Adapter). Метод позволяет значительно снизить требования к вычислительным ресурсам при сохранении высокого качества дообучения.

> **Важно**: Этот код позволяет дообучать Llama-4-Maverick на домашнем компьютере с одной GPU (NVIDIA RTX 3090/4090), используя всего 0.1% параметров от общего количества параметров модели.

## Содержание

- [Обзор](#обзор)
- [Как работает MoE Adapter](#как-работает-moe-adapter)
- [Преимущества метода](#преимущества-метода)
- [Требования](#требования)
- [Структура проекта](#структура-проекта)
- [Быстрый старт](#быстрый-старт)
- [Установка](#установка)
- [Использование](#использование)
  - [Подготовка данных](#подготовка-данных)
  - [Настройка конфигурации](#настройка-конфигурации)
  - [Запуск дообучения](#запуск-дообучения)
- [Параметры конфигурации](#параметры-конфигурации)
- [Тестирование модели](#тестирование-модели)
- [Примеры](#примеры)
- [Частые вопросы (FAQ)](#частые-вопросы-faq)
- [Устранение проблем](#устранение-проблем)
- [Лицензия](#лицензия)

## Обзор

Метод MoE Adapter объединяет преимущества параметрически-эффективных методов дообучения (таких как LoRA) с архитектурой Mixture of Experts. Это позволяет:

- Обучать менее 0.1% параметров от общего количества параметров модели
- Эффективно использовать GPU с ограниченным объемом VRAM (например, RTX 3090, RTX 4090)
- Сохранять высокое качество генерации текста
- Предотвращать переобучение и коллапс специализации экспертов

Особенно эффективен этот метод для моделей с архитектурой MoE, таких как Llama-4-Maverick, которые уже имеют встроенные механизмы маршрутизации и специализации экспертов.

## Как работает MoE Adapter

MoE Adapter состоит из трех основных компонентов:

### 1. Адаптеры экспертов

Для каждого эксперта в модели создается небольшой адаптер, использующий метод LoRA (низкоранговая адаптация). Это позволяет каждому эксперту специализироваться на определенных типах данных или задачах.

```python
# Пример реализации адаптера эксперта
class ExpertAdapter(nn.Module):
    def __init__(self, hidden_size, expert_dim=32):
        super().__init__()
        self.down = nn.Linear(hidden_size, expert_dim)  # Понижение размерности
        self.activation = nn.GELU()                    # Нелинейная активация
        self.up = nn.Linear(expert_dim, hidden_size)    # Повышение размерности

    def forward(self, x):
        return self.up(self.activation(self.down(x)))
```

### 2. Маршрутизатор (Router)

Маршрутизатор определяет, какие эксперты должны обрабатывать каждый токен входной последовательности. Обычно используется подход Top-K, когда для каждого токена выбираются K экспертов с наивысшими вероятностями.

```python
# Пример реализации маршрутизатора
class MoERouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)  # Слой маршрутизации
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states):
        # Вычисляем вероятности маршрутизации
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Выбираем top-k экспертов
        weights, indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
        return weights, indices
```

### 3. Механизмы балансировки нагрузки

Для предотвращения ситуации, когда несколько экспертов используются слишком часто, а другие - редко, применяются специальные функции потерь и ограничения емкости.

```python
# Пример функции потери для балансировки нагрузки
def load_balancing_loss(router_probs, num_experts):
    # Вычисляем долю токенов, направляемых к каждому эксперту
    router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
    # Вычисляем потерю балансировки нагрузки
    return torch.mean(router_prob_per_expert * router_prob_per_expert) * num_experts
```

Все эти компоненты работают вместе, позволяя модели эффективно адаптироваться к новым данным и задачам при минимальных вычислительных затратах.

## Преимущества метода

По сравнению с другими методами параметрически-эффективного дообучения, MoE Adapter имеет ряд значительных преимуществ:

### 1. Эффективность использования памяти

- **Меньше параметров**: Требует обучения всего 0.1% параметров от общего количества параметров модели
- **Низкие требования к VRAM**: Модель на 8 млрд параметров можно дообучать на одной GPU с 24 ГБ VRAM
- **Совместимость с квантизацией**: Отлично работает с 4-битной и 8-битной квантизацией

### 2. Адаптивная специализация

- **Специализация экспертов**: Каждый эксперт фокусируется на определенных типах данных или задачах
- **Динамический выбор экспертов**: Маршрутизатор выбирает наиболее подходящих экспертов для каждого токена
- **Лучшая генерализация**: Модель лучше работает с разнородными данными

### 3. Масштабируемость и гибкость

- **Модульность**: Можно добавлять новых экспертов без переобучения существующих
- **Простота интеграции**: Легко интегрируется с существующими моделями и инфраструктурой
- **Производительность**: Минимальное влияние на скорость инференса

### 4. Особенно эффективен для MoE-моделей

- **Синергия с MoE-архитектурой**: Естественно интегрируется с моделями, уже имеющими архитектуру MoE
- **Настройка маршрутизатора**: Позволяет тонко настраивать маршрутизатор для новых задач
- **Улучшение специализации**: Усиливает специализацию существующих экспертов для конкретных задач

## Требования

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4.0+
- Accelerate 0.20.0+
- CUDA-совместимая GPU с минимум 24 ГБ VRAM (NVIDIA RTX 3090, RTX 4090 или лучше)

## Структура проекта

```
llama4-maverick-finetuning/
├── fine_tune_llama4_maverick.py  # Основной скрипт для дообучения
├── run_fine_tuning.py            # Скрипт для запуска дообучения с конфигурацией
├── test_model.py                 # Скрипт для тестирования дообученной модели
├── config.json                   # Файл конфигурации
├── data/                         # Директория с данными для дообучения
│   ├── train.json                # Тренировочные данные
│   └── validation.json           # Валидационные данные
└── output/                       # Директория для сохранения результатов (создается автоматически)
```

## Быстрый старт

Для быстрого запуска дообучения выполните следующие шаги:

### 1. Подготовьте данные

Создайте файл `data/train.json` с вашими данными для дообучения в формате:

```json
[
  {
    "prompt": "Ваш вопрос или инструкция",
    "completion": "Желаемый ответ модели"
  },
  ...
]
```

### 2. Настройте конфигурацию

Отредактируйте файл `config.json` в соответствии с вашими потребностями. Основные параметры, которые стоит настроить:

```json
{
  "model_name_or_path": "meta-llama/Llama-4-Maverick-8B",  // Путь к модели
  "load_in_8bit": true,                                  // Использовать 8-битную квантизацию
  "num_experts": 8,                                      // Количество экспертов
  "expert_capacity": 2,                                  // Количество токенов на эксперта (top-k)
  "per_device_train_batch_size": 2,                      // Размер батча
  "gradient_accumulation_steps": 16                      // Шаги накопления градиентов
}
```

### 3. Запустите дообучение

Запустите скрипт быстрого старта:

```bash
./quick_start.sh
```

Или запустите процесс дообучения вручную:

```bash
# Установите зависимости
pip install -r requirements.txt

# Запустите дообучение
python run_fine_tuning.py --config config.json
```

### 4. Протестируйте модель

После завершения дообучения протестируйте модель:

```bash
python test_model.py --model_path ./output --interactive
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/llama4-maverick-finetuning-code.git
cd llama4-maverick-finetuning-code
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
# Установите PyTorch с поддержкой CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установите все необходимые библиотеки
pip install -r requirements.txt
```

4. Проверьте доступность GPU:
```bash
python -c "import torch; print('GPU доступна:', torch.cuda.is_available()); print('Количество GPU:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0))"
```

## Частые вопросы (FAQ)

### Какие модели поддерживаются?

Код оптимизирован для моделей с архитектурой MoE, таких как:
- Llama-4-Maverick (8B, 70B)
- Mixtral (8x7B, 8x22B)
- Grok-1.5 (314B)
- Switch Transformer

Однако он также может быть адаптирован для обычных моделей трансформеров, таких как Llama 3, Gemma и др.

### Какие минимальные требования к GPU?

Для дообучения модели Llama-4-Maverick-8B с использованием 8-битной квантизации:
- Минимум: NVIDIA RTX 3090 (24 ГБ VRAM)
- Рекомендуется: NVIDIA RTX 4090 (24 ГБ VRAM) или лучше

Для более крупных моделей рекомендуется использовать несколько GPU или профессиональные карты (A100, H100).

### Сколько времени занимает дообучение?

Время дообучения зависит от многих факторов, включая:
- Размер модели
- Объем данных для дообучения
- Мощность GPU
- Настройки квантизации и оптимизации

Примерные временные рамки для Llama-4-Maverick-8B на RTX 4090:
- 1000 примеров: 1-2 часа
- 5000 примеров: 4-6 часов
- 10000 примеров: 8-12 часов

### Как уменьшить требования к памяти?

Для уменьшения требований к памяти можно:
- Использовать 4-битную квантизацию (`load_in_4bit: true`)
- Уменьшить размер батча (`per_device_train_batch_size: 1`)
- Увеличить шаги накопления градиентов (`gradient_accumulation_steps: 32`)
- Уменьшить количество экспертов (`num_experts: 4`)
- Ограничить длину последовательности (`max_seq_length: 512`)

### Как использовать дообученную модель в своем приложении?

После дообучения модель сохраняется в директории `output/`. Вы можете загрузить ее в своем приложении с помощью Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_path = "./output"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Генерация текста
prompt = "Ваш запрос"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Использование

### Подготовка данных

Подготовьте данные для дообучения в формате JSON. Каждый пример должен содержать поля `prompt` и `completion`:

```json
[
  {
    "prompt": "Ваш вопрос или инструкция",
    "completion": "Желаемый ответ модели"
  },
  ...
]
```

Поместите файлы с данными в директорию `data/`:
- `data/train.json` - тренировочные данные
- `data/validation.json` - валидационные данные (опционально)

### Настройка конфигурации

Отредактируйте файл `config.json` в соответствии с вашими потребностями. Основные параметры:

```json
{
  "model_name_or_path": "meta-llama/Llama-4-Maverick-8B",
  "use_lora": true,
  "lora_rank": 8,
  "use_moe": true,
  "num_experts": 8,
  "expert_capacity": 2,
  "load_in_8bit": true,
  "train_file": "data/train.json",
  "validation_file": "data/validation.json",
  "max_seq_length": 512,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 16,
  "learning_rate": 2e-4,
  "num_train_epochs": 3
}
```

### Запуск дообучения

```bash
python run_fine_tuning.py --config config.json
```

Процесс дообучения будет выводить информацию о прогрессе, значениях потерь и других метриках. Результаты будут сохранены в директории `output/`.

## Параметры конфигурации

### Основные параметры модели

- `model_name_or_path`: Путь к базовой модели или идентификатор на Hugging Face
- `use_lora`: Использовать ли LoRA для параметрически-эффективного дообучения
- `lora_rank`: Ранг матриц LoRA (обычно от 4 до 64)
- `lora_alpha`: Параметр масштабирования для LoRA
- `lora_dropout`: Вероятность dropout для LoRA

### Параметры MoE

- `use_moe`: Использовать ли MoE Adapter
- `num_experts`: Количество экспертов в MoE
- `expert_capacity`: Количество токенов, обрабатываемых каждым экспертом (top-k)

### Параметры оптимизации памяти

- `load_in_8bit`: Использовать ли 8-битную квантизацию
- `load_in_4bit`: Использовать ли 4-битную квантизацию

### Параметры данных

- `train_file`: Путь к файлу с тренировочными данными
- `validation_file`: Путь к файлу с валидационными данными
- `max_seq_length`: Максимальная длина последовательности

### Параметры обучения

- `output_dir`: Директория для сохранения результатов
- `per_device_train_batch_size`: Размер батча на устройство
- `gradient_accumulation_steps`: Количество шагов для накопления градиентов
- `learning_rate`: Скорость обучения
- `weight_decay`: Коэффициент регуляризации весов
- `num_train_epochs`: Количество эпох обучения
- `warmup_ratio`: Доля шагов для разогрева оптимизатора

## Тестирование модели

После дообучения вы можете протестировать модель с помощью скрипта `test_model.py`:

```bash
# Интерактивный режим
python test_model.py --model_path ./output --interactive

# Тестирование с конкретным запросом
python test_model.py --model_path ./output --prompt "Ваш запрос"

# Тестирование с примерами
python test_model.py --model_path ./output
```

## Примеры

### Пример запуска дообучения с минимальными требованиями к памяти

```bash
# Редактируем config.json
{
  "model_name_or_path": "meta-llama/Llama-4-Maverick-8B",
  "use_lora": true,
  "lora_rank": 8,
  "use_moe": true,
  "num_experts": 4,
  "expert_capacity": 1,
  "load_in_4bit": true,
  "max_seq_length": 512,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 32,
  "learning_rate": 2e-4,
  "num_train_epochs": 3
}

# Запускаем дообучение
python run_fine_tuning.py --config config.json
```

### Пример запуска с несколькими GPU

```bash
# Редактируем config.json
{
  "model_name_or_path": "meta-llama/Llama-4-Maverick-8B",
  "use_lora": true,
  "lora_rank": 16,
  "use_moe": true,
  "num_experts": 8,
  "expert_capacity": 2,
  "load_in_8bit": true,
  "max_seq_length": 1024,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-4,
  "num_train_epochs": 3
}

# Запускаем дообучение с использованием нескольких GPU
CUDA_VISIBLE_DEVICES=0,1 python run_fine_tuning.py --config config.json
```

## Устранение проблем

### Ошибка CUDA out of memory

**Проблема**: При запуске дообучения возникает ошибка `CUDA out of memory`.

**Решение**:
1. Уменьшите размер батча (`per_device_train_batch_size: 1`)
2. Увеличьте шаги накопления градиентов (`gradient_accumulation_steps: 32`)
3. Используйте 4-битную квантизацию (`load_in_4bit: true`)
4. Уменьшите длину последовательности (`max_seq_length: 512` или меньше)
5. Уменьшите количество экспертов (`num_experts: 4`)

### Ошибка при загрузке модели

**Проблема**: При загрузке модели возникает ошибка `Unable to load model`.

**Решение**:
1. Убедитесь, что у вас есть доступ к модели (Llama-4-Maverick требует авторизации на Hugging Face)
2. Проверьте наличие токена Hugging Face: `huggingface-cli login`
3. Убедитесь, что указано правильное имя модели в `model_name_or_path`

### Проблемы с квантизацией

**Проблема**: Ошибки при использовании квантизации (`load_in_8bit` или `load_in_4bit`).

**Решение**:
1. Убедитесь, что установлены последние версии библиотек: `pip install -U bitsandbytes transformers accelerate`
2. Проверьте совместимость вашей GPU с библиотекой bitsandbytes
3. Для Linux может потребоваться установка дополнительных библиотек: `apt-get install -y libcudnn8`

### Низкая производительность дообучения

**Проблема**: Дообучение происходит слишком медленно.

**Решение**:
1. Используйте более быструю GPU или несколько GPU
2. Увеличьте размер батча и уменьшите шаги накопления градиентов (если позволяет память)
3. Включите режим смешанной точности (`fp16: true`)
4. Используйте Flash Attention, если ваша GPU поддерживает его

## Лицензия

MIT
