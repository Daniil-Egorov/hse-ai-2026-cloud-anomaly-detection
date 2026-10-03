## Вводные

### Пререквизиты:
1. Желательны основы ML/DL (классификация, кластеризация, метрики качества).
2. Базовые знания о распределенных облачных системах и мониторинге.

### Описание:
Проект востребован, над темой активно работают многие исследователи. Он подойдёт как начинающим, так и продвинутым ресерчерам, а также может стать хорошей основой для дальнейшей работы уже в рамках ВКР. По сложным техническим вопросам (работа облаков, специфика данных, примеры успешных решений) можно будет получить консультации у специалистов из ИСП РАН.

### Цель проекта
Построить систему автоматического анализа телеметрии (метрик и логов) распределенной инфраструктуры и выявления подозрительных состояний для обеспечения безопасности инфраструктуры.

### Возможные задачи проекта
Сбор данных: метрики Prometheus/OpenTelemetry, открытые датасеты (loghub). Логи реальныхоблачных систем (есть большие наборы данных собранных из реальных проектов).

#### ML-подходы
- Unsupervised: DBSCAN, Isolation Forest, One-Class SVM.
- Ensemble из supervised и unsupervised.

#### DL-подходы:
- VAE для поиска по reconstruction error.
- Semi-supervised: DQNLog, Deeplog
- LSTM/Logbert/Mamba для работы с последовательностями (аномалии по ошибке прогноза).
- экспериментальные архитектуры Mamba + LLM, включая модели семейства Nemotron.

#### Сервисная часть:
- Работа с брокерами (kafka или rabbitmq)
- Онлайн обработка данных
- Продвинутый инференс (triton) либо fastapi .
- В зависимости от интересов проект можно сместить в сторону ML/DL research, исследования LLM/Mamba для детекции аномалий, либо разработки полноценной онлайн системы обнаружения аномалий в облачной инфраструктуре.

## Roadmap по проекту

```mermaid
gantt
    title System Design Roadmap
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Foundations
    Framework and estimation : done, n01, 2026-10-05, 7d
    LXD lab                  : active, n02, after n01, 3d

    section Core
    Networking and APIs      : n03, after n02, 10d
    Storage and databases    : n04, after n03, 14d
    Replication and sharding : crit, n05, after n04, 14d
    Distributed theory       : crit, n06, after n05, 14d

    section Scale and ops
    Caching and scaling      : n07, after n06, 10d
    Messaging                : n08, after n07, 14d
    Reliability              : n09, after n08, 10d

    section Finish
    Casebook                 : cb, after n05, 90d
    Capstone                 : milestone, cap, after cb, 0d
```

