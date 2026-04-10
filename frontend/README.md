# Life Logs Frontend

Vue 3 + Tailwind CSS + Vuex одностраничный интерфейс для общения с "живым дневником" и отправки сообщений на FastAPI-бэкенд.

## Требования

- Node.js 18+
- npm 9+
- Запущенный backend по умолчанию на `http://127.0.0.1:8000`

## Запуск

```bash
cd frontend
npm install
npm run dev
```

После запуска UI будет доступен по адресу, который покажет Vite (по умолчанию `http://localhost:5173`).

## Настройка API

Можно задать собственный адрес API через переменную окружения:

```
# файл frontend/.env
VITE_API_BASE_URL=http://192.168.0.10:8000
```

## Сборка

```bash
cd frontend
npm run build
npm run preview
```

`npm run preview` запустит статический сервер для проверки production-сборки.
