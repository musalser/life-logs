#!/usr/bin/env bash
#
# wsl-host-env.sh — определяет адрес Windows-хоста из WSL2 (NAT) и публикует
# адреса сервисов в backend/.env.
#
# Зачем: на Windows 10 нет `networkingMode=mirrored`, поэтому WSL2 ходит на
# Windows через шлюз по умолчанию. Этот адрес меняется после каждого
# `wsl --shutdown`, поэтому он вычисляется в рантайме, а не хардкодится.
#
# Использование:
#   source scripts/wsl-host-env.sh   # экспортировать переменные в текущий shell
#   bash   scripts/wsl-host-env.sh   # только обновить backend/.env
#
# Новый сервис на Windows = одна строка в SERVICE_URLS ниже.
#
# ВНИМАНИЕ: файл рассчитан на `source`, поэтому здесь намеренно нет
# `set -e` / `set -u` — они утекли бы в вызывающий shell и сломали его.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/backend/.env"

# --- 1. Адрес Windows-хоста ---------------------------------------------------

resolve_windows_host() {
    local host=""

    # WSL2 NAT: Windows-хост — это шлюз по умолчанию виртуалки WSL.
    if command -v ip >/dev/null 2>&1; then
        host="$(ip route show default 2>/dev/null | awk '/^default/ {print $3; exit}')"
    fi

    if [[ -n "${host}" ]]; then
        printf '%s' "${host}"
        return 0
    fi

    # Нативный Linux, WSL1 или mirrored-сеть: всё на localhost.
    printf '%s' "127.0.0.1"
}

WIN_HOST="${WIN_HOST:-$(resolve_windows_host)}"
export WIN_HOST

# --- 2. Адреса сервисов -------------------------------------------------------
# Новый сервис на Windows → добавьте одну строку здесь.
declare -A SERVICE_URLS=(
    [OLLAMA_URL]="http://${WIN_HOST}:11434"
    # [REDIS_URL]="redis://${WIN_HOST}:6379"
    # [POSTGRES_URL]="postgresql+psycopg://postgres:postgres@${WIN_HOST}:5432/life_logs"
)

# --- 3. Upsert в backend/.env -------------------------------------------------
# Заменяет `KEY=...`, если ключ уже есть, иначе дописывает в конец.
# Остальные строки файла сохраняются.

upsert_env() {
    local file="$1" key="$2" value="$3" tmp

    [[ -f "${file}" ]] || : > "${file}"

    if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "${file}"; then
        tmp="$(mktemp)"
        if awk -v key="${key}" -v value="${value}" '
            $0 ~ "^[[:space:]]*" key "[[:space:]]*=" { print key "=" value; next }
            { print }
        ' "${file}" > "${tmp}"; then
            mv "${tmp}" "${file}"
        else
            rm -f "${tmp}"
            echo "wsl-host-env: не удалось обновить ${key} в ${file}" >&2
        fi
    else
        printf '%s=%s\n' "${key}" "${value}" >> "${file}"
    fi
}

for name in "${!SERVICE_URLS[@]}"; do
    value="${SERVICE_URLS[${name}]}"
    export "${name}=${value}"          # no-op при запуске, полезно при source
    upsert_env "${ENV_FILE}" "${name}" "${value}"
done

# --- 4. Отчёт -----------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "wsl-host-env: WIN_HOST=${WIN_HOST} (переменные экспортированы)"
else
    echo "wsl-host-env: WIN_HOST=${WIN_HOST}"
fi

for name in "${!SERVICE_URLS[@]}"; do
    printf '  %s=%s\n' "${name}" "${SERVICE_URLS[${name}]}"
done

echo "wsl-host-env: обновлён ${ENV_FILE#"${ROOT_DIR}"/}"
