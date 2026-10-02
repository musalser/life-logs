# Браузер для агента: как проверять интерфейс своими глазами

Инструкция для ИИ-агента, который работает в этой же среде (WSL2) и хочет
проверять вёрстку не по коду, а по факту. Всё ниже проверено на этом проекте:
`frontend` (Nuxt, dev на `:3000`) + `backend` (FastAPI на `:8000`).

Зачем это нужно — что браузер поймал за один вечер работы над интерфейсом:

* `<mark>` из `v-html` не получил scoped-стили Vue → браузер рисовал свою жёлтую
  подсветку (в коде всё выглядело правильно);
* клик по слову в транскрипции выделял **другое** слово: жирное слово слоя шире,
  раскладка слоя уезжала, а каретку браузер считает по своей раскладке textarea;
* на Windows полоса прокрутки появлялась от переполнения в 1 px, отнимала 17 px
  ширины и оставалась навсегда — на Linux (оверлейные полосы) этого не видно;
* после «вписать по ширине» первый щелчок колеса прыгал с 25 % на 116 % (устаревшая
  внутренняя цель) — видно только в рантайме;
* подсветка слова закрывалась синим прямоугольником выделения ровно в момент клика.

Ни один из этих багов не виден при чтении кода и не ловится юнит-тестами.

---

## 0. Ограничения среды (прочитать до начала)

| ограничение | следствие |
| --- | --- |
| WSL2 без дисплея | нужен headless-браузер, обычный Chrome не запустится |
| `sudo` заблокирован (`no new privileges`); даже с расширенными правами просит пароль | ставим браузер **без root** (раздел 2) |
| `/tmp` очищается при перезагрузке | установку придётся повторить; скрипт — в приложении А |
| `~/.npm` может содержать root-файлы | свой кэш: `npm_config_cache=/tmp/npm-cache` |
| нет доступа к `/home/*/.cache` | всё складывать в `/tmp` (`PLAYWRIGHT_BROWSERS_PATH`) |

Проверка, что окружение «пустое»:

```bash
which chromium chromium-browser google-chrome 2>/dev/null || echo "браузера нет"
sudo -n true 2>&1 | head -1          # "no new privileges" — это ожидаемо
```

---

## 1. Сначала проверьте вариант A: подключиться к уже запущенному Chrome

Если на Windows-хосте запущен Chrome/Edge с `--remote-debugging-port=9222`, из WSL
он доступен **не по 127.0.0.1, а по IP шлюза** (в WSL2 это отдельная сеть):

```bash
HOST=$(ip route | awk '/^default/ {print $3}')     # например 192.168.80.1
curl -s --max-time 8 "http://$HOST:9222/json/version"
```

Если ответ есть — установка не нужна вообще, и вы получаете **настоящий Windows
Chrome** (а значит, виндовые скроллбары, шрифты, масштаб 125 % — то, что headless
Linux не показывает):

```js
import { chromium } from 'playwright'
const browser = await chromium.connectOverCDP(`http://${host}:9222`)
const context = browser.contexts()[0] || (await browser.newContext())
```

Минус: зависит от того, что кто-то этот Chrome запустил (в моей сессии он
периодически исчезал), и вы работаете в чужом профиле — не открывайте лишних
вкладок, не меняйте чужие настройки.

Если порт не отвечает — переходите к варианту B.

---

## 2. Вариант B: Chromium от Playwright без root (проверено)

Три шага. Первый — скачать браузер (root не нужен), второй — добыть системные
библиотеки **без установки в систему**, третий — запускать с `LD_LIBRARY_PATH`.

**Две разные вещи, которые легко перепутать:** *браузер* (бинарник Chromium) и
*npm-пакет* `playwright` (библиотека для скриптов). Команда `npx playwright
screenshot` работает без пакета, а `import { chromium } from 'playwright'` — нет
(`ERR_MODULE_NOT_FOUND`). Ставим и то, и другое.

```bash
# 1a. Пакет для скриптов (в рабочем каталоге, например /tmp/pw)
mkdir -p /tmp/pw && cd /tmp/pw
npm_config_cache=/tmp/npm-cache npm i playwright@latest --no-audit --no-fund

# 1b. Браузер (≈115 МБ). PLAYWRIGHT_BROWSERS_PATH — потому что ~/.cache недоступен,
#     npm_config_cache — потому что ~/.npm может быть с root-файлами.
#     Эту переменную нужно ставить и при запуске скриптов, иначе Playwright
#     пойдёт искать браузер в ~/.cache/ms-playwright и не найдёт его.
export PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers
export npm_config_cache=/tmp/npm-cache
npx -y playwright@latest install chromium

# 2. Библиотеки: скачать .deb как обычный пользователь и распаковать в /tmp
mkdir -p /tmp/pw/deb /tmp/pw/rootfs && cd /tmp/pw/deb
apt-get download libnspr4 libnss3 libasound2t64 libatk1.0-0t64 libatk-bridge2.0-0t64 \
  libatspi2.0-0t64 libcups2t64 libgbm1 libxkbcommon0 libxcomposite1 libxdamage1 \
  libxfixes3 libxrandr2 libpango-1.0-0 libcairo2
for f in *.deb; do dpkg -x "$f" /tmp/pw/rootfs; done

# 3. Проверка: скриншот простой страницы
cat > /tmp/pw/test.html <<'HTML'
<html><body style="background:#0f172a;color:#f8fafc;font:16px sans-serif">
проверка <b style="color:#e879f9">цветного</b> текста</body></html>
HTML
PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers npm_config_cache=/tmp/npm-cache \
LD_LIBRARY_PATH=/tmp/pw/rootfs/usr/lib/x86_64-linux-gnu:/tmp/pw/rootfs/lib/x86_64-linux-gnu \
npx -y playwright@latest screenshot --viewport-size=520,120 /tmp/pw/test.html /tmp/pw/test.png
ls -la /tmp/pw/test.png
```

Если браузер не стартует, он сам скажет, чего не хватает:

```
error while loading shared libraries: libnspr4.so: cannot open shared object file
```

Добавьте нужный пакет в `apt-get download` (или получите полный список:
`npx -y playwright@latest install-deps --dry-run chromium` — но `install-deps`
без `--dry-run` полезет в `sudo`, чего мы избегаем). Шрифты в этом образе уже есть
(`fc-list | wc -l` → 89), поэтому пакеты `fonts-*` не понадобились.

**Дальше во всех запусках нужны те же две переменные.** Удобно завернуть:

```bash
export PWENV="PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers LD_LIBRARY_PATH=/tmp/pw/rootfs/usr/lib/x86_64-linux-gnu:/tmp/pw/rootfs/lib/x86_64-linux-gnu"
env $PWENV node shot.mjs
```

---

## 3. Вход в приложение: токен вместо пароля

Приложение закрыто авторизацией, но пароль знать не нужно — токен выпускается тем же
кодом, что и на сервере:

```bash
cd backend && .venv/bin/python -c "
from app.deps import create_access_token
print(create_access_token('musalser'))" > /tmp/pw/token.txt
```

Фронтенд хранит его в `localStorage` под ключом `life_logs_auth_token`
(`frontend/utils/auth-fetch.js`), поэтому ставим его **до** загрузки страницы:

```js
await context.addInitScript(
  ([key, value]) => localStorage.setItem(key, value),
  ['life_logs_auth_token', token]
)
```

**Обязательный шаг — перехватить `POST /auth/refresh`.** Иначе приложение при старте
зовёт его, не получает httpOnly-cookie, получает 401 и **стирает токен** — страница
останется пустой, а в консоли будут только 401:

```js
await context.route('**/auth/refresh', (route) =>
  route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ access_token: token, token_type: 'bearer' }),
  })
)
```

Токен живёт 60 минут (`access_token_expire_minutes`) — **выпускайте его перед каждым
прогоном**, иначе получите пустую страницу без внятной причины. Признак: API-ответы
401, а в DOM нет ожидаемых элементов.

---

## 4. Каркас скрипта (сохранить как `/tmp/pw/shot.mjs`)

```js
import fs from 'node:fs'
import { chromium } from 'playwright'

const token = fs.readFileSync('/tmp/pw/token.txt', 'utf8').trim()
const url = process.argv[2] || 'http://127.0.0.1:3000/manuscripts?author=1&page=31'

const browser = await chromium.launch({ headless: true })
const context = await browser.newContext({
  viewport: { width: 1680, height: 1000 },
  deviceScaleFactor: 2,          // чёткие скриншоты для чтения текста
})
await context.addInitScript(
  ([k, v]) => localStorage.setItem(k, v),
  ['life_logs_auth_token', token]
)
await context.route('**/auth/refresh', (r) =>
  r.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ access_token: token, token_type: 'bearer' }),
  })
)

const page = await context.newPage()
page.on('pageerror', (e) => console.log('pageerror:', String(e).slice(0, 200)))
page.on('console', (m) => {
  if (m.type() === 'error') console.log('console:', m.text().slice(0, 200))
})

await page.goto(url, { waitUntil: 'networkidle' })
await page.waitForSelector('.line-editor', { timeout: 60000 })  // ждём именно контент
await page.waitForTimeout(1500)   // раскладка «доезжает»: авто-высоты, шрифты, HMR

// 1) сначала убедиться, что страница действительно отрисовалась
console.log('элементов:', await page.evaluate(() => document.querySelectorAll('.line-editor').length))

// 2) метрики вместо «на глаз»
const metrics = await page.evaluate(() => {
  const area = document.querySelector('.line-editor__input')
  const layer = document.querySelector('.line-editor__layer')
  const style = getComputedStyle(area)
  return {
    heights: [area.scrollHeight, layer.scrollHeight],
    noClipping: area.scrollHeight <= area.clientHeight + 1,
    font: style.font,
  }
})
console.log(JSON.stringify(metrics, null, 1))

// 3) скриншот (и обязательно посмотреть его картинкой, а не только сохранить)
await page.locator('section', { has: page.locator('.line-editor') }).first()
  .screenshot({ path: '/tmp/pw/column.png' })

await browser.close()
```

Запуск: `env $PWENV node /tmp/pw/shot.mjs`.

---

## 5. Приёмы проверки, которые себя оправдали

1. **Метрики вместо картинки.** `page.evaluate` + `getComputedStyle`,
   `scrollHeight/clientHeight`, `getBoundingClientRect`. Скриншот врёт мелкими
   деталями на уменьшенном превью (мне казалось, что цветная скобка белая —
   пиксельная проверка показала `rgb(232,121,249)`).
2. **Инвариант вместо симптома.** Вместо «полосы не видно» — «для каждой строки
   `scrollHeight <= clientHeight + 1`»: это ловит причину и работает там, где
   симптом не воспроизводится (оверлейные полосы Linux).
3. **Проверка, что шрифт действительно моноширинный** (без неё «цветной жирный
   текст» разъезжается): ширина `iiiiiiii` должна равняться `mmmmmmmm`, а ширина
   слова в `font-weight: 400` — ширине в `700`. Задавайте свойства явно:
   `getComputedStyle(el).cssText` в Chrome пустой/неполный.
4. **Покадровый семплинг анимации** — так измерена плавность зума:

   ```js
   const samples = await page.evaluate(async ({ x, y }) => {
     const collected = []
     const record = () => {
       const m = new DOMMatrix(getComputedStyle(content).transform)
       collected.push(m.a)
       if (collected.length < 60) requestAnimationFrame(record)
     }
     requestAnimationFrame(record)
     viewer.dispatchEvent(new WheelEvent('wheel', { deltaY: -240, clientX: x, clientY: y, bubbles: true }))
     await new Promise((r) => setTimeout(r, 900))
     return collected
   }, { x, y })
   ```

   Результат: 15 различных значений масштаба за 34 кадра, дрейф якоря 0.1 px.
5. **Настоящие события, а не синтетика:** `page.mouse.click/wheel/dblclick`,
   `page.getByRole('button', { name: '…' }).click()` — ближе к тому, что делает
   пользователь; `dispatchEvent` годится только там, где нужен точный контроль.
6. **Нативные тултипы на скриншот не попадают** — проверяйте атрибут:
   `mark.getAttribute('title')`.
7. **`selectOption({ label })`** для `<select>` и **`:has()`** для фильтрации
   дубликатов (см. грабли №3).
8. **Пиксельная проверка** (когда превью обманывает) — через PIL:
   `Counter(img.crop(box).getdata()).most_common(3)`.
9. **Клики и ввод**: перед кликом убедиться, что элемент в вьюпорте
   (`scrollIntoView` + `elementFromPoint`), см. грабли №1.
10. **Браузер ≠ пакет.** `npx playwright install chromium` ставит бинарник,
    `npm i playwright` — библиотеку для `import`. И `PLAYWRIGHT_BROWSERS_PATH`
    нужен в обоих случаях: без него библиотека ищет браузер в недоступном
    `~/.cache/ms-playwright` (`ERR_MODULE_NOT_FOUND` — про пакет, «Executable
    doesn't exist» — про браузер).

---

## 6. Грабли (все выловлены лично, каждая стоила прогона)

1. **Клик мимо элемента.** Координаты из `getBoundingClientRect()` — в координатах
   вьюпорта, а элемент может быть ниже («y=2006» при окне 1000 px): клик уходит в
   `<html>`. Лечение: `await el.scrollIntoView({ block: 'center' })` **внутри того
   же** `evaluate`, затем пересчитать прямоугольник и проверить
   `document.elementFromPoint(x, y)`.
2. **Раскладка доезжает после загрузки.** Авто-высоты, шрифты, HMR: измерения и
   клики до `waitForTimeout(1000–1500)` / `waitForFunction(...)` врут.
3. **`locator('header')` → strict mode violation**: в приложении два `<header>`
   (layout и страницы). Лечение: `.filter({ has: page.locator('h1') })` или
   `getByRole('banner')`.
4. **Headless Linux ≠ Windows.** Оверлейные скроллбары, другие шрифты, нет
   масштаба 125 %. Всё, что связано с полосами/переносами, проверяйте инвариантом,
   а лучше — на Windows Chrome (вариант A).
5. **`require` в `.mjs`** падает — используйте `import` (или назовите файл `.cjs`).
6. **Устаревший токен** — самая частая причина «страница пустая»: API отвечает 401,
   приложение стирает токен. Выпускайте заново.
7. **Nuxt dev пересобирается после правок** — скриншот может быть от старого кода.
   Дайте серверу 1–3 секунды и перезагрузите страницу.
8. **`nuxt build` при живом `nuxt dev` ломает dev** (общий каталог `.nuxt`;
   получал 503 из живого сервера). Сборку делайте в стороне:
   `NUXT_BUILD_DIR=.nuxt-check npx nuxt build` — проверено: сборка проходит,
   dev-сервер остаётся живым. Каталог потом удалить.
9. **Тестовые записи в БД.** Если проверка что-то записала (я проверял «исключить
   слово» — оно легло в таблицу), уберите за собой и **проверьте**, что убрали.
10. **Долгие шаги — в фон.** Установка браузера, сборка, распознавание: запускать
    через фоновую задачу (`setsid nohup … > log 2>&1 &`) и читать лог, а не
    блокировать шаг на минуты.
11. **Один сервер, а не два.** Не поднимайте вторую копию приложения «для теста»:
    проверяйте на том, что уже запущено. Если всё же подняли — скажите об этом
    и как это остановить (я так сломал и восстановил dev-сервер и сказал об этом
    явно).

---

## 7. Гигиена

* токен — только для локального приложения; не коммитить, не печатать целиком;
* не трогать пользовательские данные для проверок: одноразовые копии, а не
  страницы пользователя; не перезаписывать продакшен-артефакты (сборщик языковой
  модели я проверял, записывая результат в `/tmp`);
* после проверок: убрать временные файлы, удалить тестовые записи из БД, не
  оставлять чужие процессы сломанными.

---

## 8. Чек-лист перед выводами

- [ ] страница действительно отрисовалась (счётчик ожидаемых элементов > 0);
- [ ] у твёрдого утверждения есть **инвариант**, а не только скриншот;
- [ ] скриншот **посмотрен картинкой** (при сомнении — проверены пиксели);
- [ ] проверено на двух размерах окна / масштабах, если речь о вёрстке;
- [ ] тестовые данные убраны, чужие процессы живы;
- [ ] в выводе разделено «проверено браузером», «проверено тестом» и «не проверено».

---

## Приложение А. Скрипт установки целиком

```bash
#!/usr/bin/env bash
# /tmp/setup-browser.sh — ставит Chromium от Playwright без root.
# Идемпотентен: повторный запуск просто ничего не ломает.
# Проверено с нуля после перезагрузки (которая очищает /tmp): ~1 минута.
set -e
export PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers
export npm_config_cache=/tmp/npm-cache
mkdir -p /tmp/pw/deb /tmp/pw/rootfs

# библиотека для скриптов + сам браузер
cd /tmp/pw && npm i playwright@latest --no-audit --no-fund
npx -y playwright@latest install chromium

cd /tmp/pw/deb
apt-get download libnspr4 libnss3 libasound2t64 libatk1.0-0t64 libatk-bridge2.0-0t64 \
  libatspi2.0-0t64 libcups2t64 libgbm1 libxkbcommon0 libxcomposite1 libxdamage1 \
  libxfixes3 libxrandr2 libpango-1.0-0 libcairo2
for f in *.deb; do dpkg -x "$f" /tmp/pw/rootfs; done

export LD_LIBRARY_PATH=/tmp/pw/rootfs/usr/lib/x86_64-linux-gnu:/tmp/pw/rootfs/lib/x86_64-linux-gnu
printf '<html><body style="font:16px sans-serif">ок</body></html>' > /tmp/pw/test.html
npx -y playwright@latest screenshot --viewport-size=200,60 /tmp/pw/test.html /tmp/pw/test.png
echo "OK: $(ls -la /tmp/pw/test.png)"
```

## Приложение Б. Токен для локального приложения

```bash
cd /path/to/backend && .venv/bin/python -c "
from app.deps import create_access_token
print(create_access_token('musalser'))" > /tmp/pw/token.txt
```

Ключ `localStorage` (`life_logs_auth_token`) и необходимость перехвата
`/auth/refresh` описаны в разделе 3: без перехвата приложение стирает токен.
