# Backend

FastAPI + SQLAlchemy + Celery, PostgreSQL (pgvector) и Redis поднимаются через
`docker compose` из корня репозитория.

## Базы данных: где лежат данные

Данные Postgres лежат **в каталоге проекта**:

```
backend/postgres-data/          # кластер Postgres 16, примонтирован в контейнер
backend/redis-data/             # данные Redis (там только брокер Celery)
```

Так было с самого начала и так остаётся: данные рядом с кодом, их видно в
файловом менеджере, бэкап — обычное копирование каталога.

### Поднимать compose нужно из этого каталога WSL

```bash
cd /home/musalser/projects/life-logs     # именно отсюда
docker compose up -d postgres
```

Docker Desktop привязывает bind-mount к тому каталогу, **из которого запущен**.
Если запустить копию проекта на стороне Windows, контейнер получит тот же путь
`/var/lib/postgresql/data`, но уже поверх tmpfs: база откроется пустой, таблиц
не будет. Данные при этом не теряются — они остаются в этом каталоге целыми,
но контейнер их не увидит.

Проверить, что смонтирован каталог проекта:

```bash
docker compose exec postgres mount | grep postgresql/data
```

Ожидаемый ответ — строка с `type ext4` (или `9p`). Если там `type tmpfs`,
контейнер поднят не из этого каталога: остановите его, перейдите в каталог
проекта и поднимите заново.

> Раньше в проекте существовала вторая копия на стороне Windows
> (`C:\Users\Sasha\github.com\musalser\life-logs`) — именно её контейнеры
> держали данные в сентябре. Сейчас канонический каталог один: этот.

### Проверка, что данные переживут пересоздание контейнера

```bash
docker compose exec postgres psql -U postgres -d life_logs \
  -c "create table if not exists _check(x int); insert into _check values (1);"
docker compose up -d --force-recreate postgres
docker compose exec postgres psql -U postgres -d life_logs -c "select * from _check;"
# строка должна остаться; после проверки: drop table _check;
```

### Перенос данных из другого места

Каталог на хосте принадлежит пользователю `postgres` (uid 999), поэтому `cp`
из обычной оболочки не сработает — копируем контейнером:

```bash
docker compose stop postgres
docker run --rm \
  -v /путь/к/старому/кластеру:/src:ro \
  -v "$PWD/backend/postgres-data:/dst" \
  pgvector/pgvector:pg16 bash -c "
    rm -rf /dst/*; cp -a /src/. /dst/
    rm -f /dst/postmaster.pid          # след прошлого запуска
    chown -R 999:999 /dst"
docker compose up -d postgres
```

`postmaster.pid` удалять обязательно: это не блокировка, но Postgres видит в нём
признак незавершённой остановки кластера.

### Дампы и восстановление

```bash
# снять дамп
docker compose exec -T postgres pg_dump -U postgres -d life_logs \
  > backend/backups/life_logs_$(date +%F).sql

# восстановить (в пустую базу)
docker compose exec -T postgres psql -U postgres -c "drop database life_logs;" \
  -c "create database life_logs;"
docker compose exec -T postgres psql -U postgres -d life_logs -q -f - \
  < backend/backups/life_logs_2026-10-15.sql
```

Каталог `backend/backups/` в `.gitignore`: дампы — рабочий инструмент, не код.

### Коллация после смены образа

Кластер, созданный другим образом Postgres, может ругаться на версию коллации
(`collation version mismatch` в логах). Лечится без переноса данных:

```bash
docker compose exec -T postgres psql -U postgres -d life_logs \
  -c "ALTER DATABASE life_logs REFRESH COLLATION VERSION;" -c "REINDEX DATABASE life_logs;"
```

## Миграции

Alembic, каталог `revisions/`:

```bash
.venv/bin/python -m alembic current      # где мы сейчас
.venv/bin/python -m alembic upgrade head # применить
.venv/bin/python -m alembic check        # нет ли расхождения моделей и схемы
```

## Сервисы знаний

* `app/services/knowledge_source_service.py` — источники текста (страница
  рукописи, запись дневника), статусы, запуск и переизвлечение;
* `app/services/extraction_pipeline.py` — сам конвейер (вызовы A–E,
  дедупликация, запись знаний);
* `app/services/analytics_service.py` — аналитика с проверкой ссылок;
* `app/services/rag_service.py` — поиск по архиву для чата.

Backfill источников для уже существующих страниц:

```bash
.venv/bin/python -m app.scripts.backfill_sources --dry-run
```
