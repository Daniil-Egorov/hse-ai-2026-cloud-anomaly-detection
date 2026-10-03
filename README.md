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
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d
    section Foundations
    01 Framework & estimation   :done,    n01, 2026-10-05, 7d
    02 LXD/Incus lab            :active,  n02, after n01, 3d
    section Core
    03 Networking & APIs        :         n03, after n02, 10d
    04 Storage & databases      :         n04, after n03, 14d
    05 Replication & sharding   :crit,    n05, after n04, 14d
    06 Distributed theory       :crit,    n06, after n05, 14d
    section Scale & ops
    07 Caching & scaling        :
    08 Messaging                :         n08, after n07, 14d
    09 Reliability              :         n09, after n08, 10d
    section Finish
    Casebook (40 problems)      :
    Capstone                    :milestone, after cb, 0d
```

