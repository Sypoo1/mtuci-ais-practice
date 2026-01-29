# MTUCI Shop Detector - Docker Setup

## Быстрый старт

### 1. Создайте .env файл

```bash
cp .env.example .env
```

Отредактируйте .env с вашими настройками (по умолчанию можно оставить как есть):

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mtuci_shop_detector
DB_USER=postgres
DB_PASSWORD=postgres
```

### 2. Установите зависимости (локально, если нужно)

```bash
uv add sqlalchemy psycopg2-binary python-dotenv
```

### 3. Запустите приложение в Docker

```bash
docker-compose up -d
```

Это запустит:
- PostgreSQL базу данных на порту 5432
- Streamlit приложение на порту 8501

### 4. Откройте приложение

Перейдите в браузере: http://localhost:8501

## Локальная разработка (приложение локально + БД в Docker)

Если вы хотите запустить приложение локально для разработки, но использовать базу данных в Docker:

### 1. Запустите только базу данных

```bash
docker-compose -f docker-compose.db-only.yml up -d
```

### 2. Создайте .env файл (если еще не создан)

```bash
cp .env.example .env
```

Убедитесь, что в .env указан `DB_HOST=localhost`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mtuci_shop_detector
DB_USER=postgres
DB_PASSWORD=postgres
```

### 3. Установите зависимости

```bash
uv sync
```

### 4. Запустите приложение локально

```bash
uv run streamlit run main.py
```

### 5. Откройте приложение

http://localhost:8501

### Остановка базы данных

```bash
docker-compose -f docker-compose.db-only.yml down
```

---

## Управление (полный Docker)

### Остановить приложение
```bash
docker-compose down
```

### Остановить и удалить данные
```bash
docker-compose down -v
rm -rf postgres_data
```

### Просмотр логов
```bash
# Все сервисы
docker-compose logs -f

# Только приложение
docker-compose logs -f app

# Только база данных
docker-compose logs -f db_postgres
```

### Перезапуск
```bash
docker-compose restart
```

### Пересборка после изменений
```bash
docker-compose up -d --build
```

## Подключение к базе данных

### Из хоста (вашего компьютера)
```bash
psql -h localhost -p 5432 -U postgres -d mtuci_shop_detector
# Пароль: postgres
```

### Из контейнера
```bash
docker exec -it db_postgres psql -U postgres -d mtuci_shop_detector
```

## Просмотр данных аналитики

```sql
-- Все записи
SELECT * FROM detection_analytics;

-- Последние 10 записей
SELECT * FROM detection_analytics ORDER BY id DESC LIMIT 10;

-- Статистика по типам файлов
SELECT
    file_type,
    COUNT(*) as total_requests,
    AVG(CASE WHEN file_type = 'image' THEN person_count ELSE person_count_avg END) as avg_people
FROM detection_analytics
GROUP BY file_type;

-- Статистика по моделям
SELECT
    model_name,
    COUNT(*) as usage_count,
    AVG(confidence_threshold) as avg_confidence
FROM detection_analytics
GROUP BY model_name;
```

## Структура проекта

```
.
├── docker-compose.yml      # Конфигурация Docker Compose
├── Dockerfile             # Образ приложения
├── .dockerignore          # Исключения для Docker
├── main.py                # Основное приложение
├── database.py            # Работа с БД
├── database_schema.sql    # SQL схема
├── .env.example           # Пример переменных окружения
├── pyproject.toml         # Зависимости Python
└── postgres_data/         # Данные PostgreSQL (создается автоматически)
```

## Переменные окружения

Настройки базы данных задаются в `docker-compose.yml`:

```yaml
environment:
  - DB_HOST=db_postgres
  - DB_PORT=5432
  - DB_NAME=mtuci_shop_detector
  - DB_USER=postgres
  - DB_PASSWORD=postgres
```

## Настройка веб-камеры в Docker

Для работы веб-камеры в Docker контейнере необходимо:

### 1. Дать доступ к X11 серверу (для Linux/macOS)

```bash
xhost +local:docker
```

### 2. Перезапустить контейнер

```bash
docker-compose down
docker-compose up -d --build
```

### 3. Проверить доступные видео-устройства

```bash
ls -la /dev/video*
```

Если у вас другие видео-устройства (например, `/dev/video2`), отредактируйте `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
  - /dev/video2:/dev/video2  # Добавьте ваше устройство
```

### Для macOS

На macOS доступ к веб-камере из Docker ограничен. Рекомендуется запускать приложение локально:

```bash
# Запустите только БД в Docker
docker-compose -f docker-compose.db-only.yml up -d

# Запустите приложение локально
uv run streamlit run main.py
```

### Для Windows

На Windows с WSL2:

1. Убедитесь, что WSL2 имеет доступ к камере
2. Используйте WSLg для графических приложений
3. Или запускайте приложение локально (см. выше)

### Альтернатива: Использование видео-файлов

Если веб-камера не работает, вы всегда можете использовать:
- Загрузку видео-файлов через интерфейс
- Загрузку изображений для анализа

## Troubleshooting

### Веб-камера не работает в Docker

**Ошибка:** "Could not open webcam or video source"

**Решение 1:** Проверьте права доступа
```bash
# Дайте права на видео-устройства
sudo chmod 666 /dev/video0
sudo chmod 666 /dev/video1

# Добавьте пользователя в группу video
sudo usermod -aG video $USER
```

**Решение 2:** Запустите с правами root
```bash
docker-compose down
docker-compose up -d
```

**Решение 3:** Используйте локальный запуск
```bash
docker-compose -f docker-compose.db-only.yml up -d
uv run streamlit run main.py
```

### Порт 5432 уже занят
Измените порт в `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # Используйте 5433 вместо 5432
```

### Порт 8501 уже занят
Измените порт в `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Используйте 8502 вместо 8501
```

### База данных не инициализируется
```bash
docker-compose down -v
rm -rf postgres_data
docker-compose up -d
```

### Приложение не подключается к БД
Проверьте логи:
```bash
docker-compose logs app
docker-compose logs db_postgres
```

## Производство

Для production использования:
1. Измените пароли в `docker-compose.yml`
2. Используйте volume для постоянного хранения данных
3. Настройте backup базы данных
4. Добавьте reverse proxy (nginx)