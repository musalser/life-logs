<template>
  <div class="flex flex-col gap-4">
    <header class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 class="text-lg font-semibold">{{ pageId ? 'Запись дневника' : 'Новая запись' }}</h2>
      </div>

      <div class="flex flex-col items-start gap-2 md:items-end">
        <div class="flex items-center gap-2">
          <button
            type="button"
            class="inline-flex items-center justify-center rounded-full bg-app-primary px-5 py-2 text-sm font-semibold text-app-on-primary transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
            :disabled="!markdown.trim() || saving"
            @click="save"
          >
            {{ saving ? 'Сохранение...' : 'Сохранить' }}
          </button>
        </div>
        <div class="flex items-center gap-2">
          <span v-if="sourceStatus" class="rounded-full px-2.5 py-0.5 text-[11px] font-semibold" :class="sourceStatusTone(sourceStatus.status)">
            {{ sourceStatusLabel(sourceStatus.status) }}
          </span>
          <button
            v-if="sourceStatus && sourceStatus.status === 'failed'"
            type="button"
            class="text-[11px] text-app-accent transition hover:underline"
            :disabled="extracting"
            @click="reextract"
          >
            {{ extracting ? 'запуск…' : 'повторить' }}
          </button>
        </div>
        <p class="text-xs text-app-muted">{{ savedLabel }}</p>
      </div>
    </header>

    <section class="flex flex-col gap-4">
      <div class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-wide text-app-muted">Markdown</p>
        <textarea
          v-model="markdown"
          class="min-h-[50vh] w-full resize-none rounded-xl border border-app-border/40 bg-app-input/60 px-4 py-3 text-base text-app-text shadow-inner shadow-black/20 focus:border-app-primary focus:outline-none"
          placeholder="# Что произошло сегодня?
- Главное событие дня…
- Маленькая победа…

## Цели
- …"
        ></textarea>
      </div>

      <div class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-wide text-app-muted">Превью</p>

        <div v-if="sanitizedHtml" class="prose-preview" v-html="sanitizedHtml"></div>
        <p v-else class="text-sm text-app-muted">Начните печатать слева, чтобы увидеть превью.</p>
      </div>
    </section>

    <p v-if="saveError" class="rounded-xl border border-app-border/40 bg-app-input/40 px-4 py-3 text-sm text-app-error">
      {{ saveError }}
    </p>
  </div>
</template>

<script setup>
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { authFetch } from '~/utils/auth-fetch'
import { pollSourceStatus, sourceStatusLabel, sourceStatusTone } from '~/composables/useKnowledge'

marked.setOptions({ gfm: true, breaks: true })

const { public: { apiBaseUrl } } = useRuntimeConfig()

const props = defineProps({
  pageId: { type: Number, default: null }
})
const emit = defineEmits(['updated'])

const markdown = ref('')
const lastSavedAt = ref(null)
const saving = ref(false)
const saveError = ref('')
const selectedPageId = ref(null)

const sourceStatus = ref(null)
const extracting = ref(false)
const pollingStop = ref(null)

const sanitizedHtml = computed(() => {
  const source = markdown.value.trim()
  if (!source) return ''
  const raw = marked.parse(source)
  return DOMPurify.sanitize(raw)
})

const savedLabel = computed(() => {
  if (!lastSavedAt.value) return 'Не сохранено'
  try {
    return `Сохранено: ${new Date(lastSavedAt.value).toLocaleString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short'
    })}`
  } catch {
    return 'Сохранено'
  }
})

const api = (path, init) => authFetch(apiBaseUrl, `${apiBaseUrl}${path}`, init)

const loadSelectedPage = async (pageId) => {
  saveError.value = ''
  stopPolling()
  const pageIdNum = Number(pageId)
  if (!Number.isInteger(pageIdNum)) {
    selectedPageId.value = null
    markdown.value = ''
    lastSavedAt.value = null
    sourceStatus.value = null
    return
  }

  try {
    const response = await api(`/diary/pages/${pageIdNum}`)
    if (response.status === 401) {
      saveError.value = 'Сессия истекла. Войдите снова.'
      return
    }
    if (!response.ok) throw new Error('Не удалось загрузить страницу')

    const page = await response.json()
    selectedPageId.value = page.id
    markdown.value = page.content || ''
    lastSavedAt.value = page.created_at || null
    await loadSourceStatus()
  } catch (error) {
    saveError.value = error.message || 'Ошибка загрузки страницы'
  }
}

const save = async () => {
  const text = markdown.value.trim()
  if (!text) return

  saving.value = true
  saveError.value = ''
  try {
    const isUpdate = Boolean(selectedPageId.value)
    const response = await api(
      isUpdate ? `/diary/pages/${selectedPageId.value}` : '/diary/pages',
      {
        method: isUpdate ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text })
      }
    )

    if (response.status === 401) {
      saveError.value = 'Сессия истекла. Войдите снова.'
      return
    }
    if (!response.ok) throw new Error('Не удалось сохранить страницу')

    const savedPage = await response.json()
    selectedPageId.value = savedPage.id
    markdown.value = savedPage.content || text
    lastSavedAt.value = savedPage.created_at || new Date().toISOString()
    emit('updated', savedPage)
    await loadSourceStatus()
    watchExtraction()
  } catch {
    saveError.value = 'Ошибка сохранения страницы'
  } finally {
    saving.value = false
  }
}

const loadSourceStatus = async () => {
  if (!selectedPageId.value) {
    sourceStatus.value = null
    return
  }
  try {
    const response = await api(`/knowledge/sources/diary/${selectedPageId.value}/status`)
    if (response.status === 404) {
      sourceStatus.value = null
      return
    }
    if (response.ok) sourceStatus.value = await response.json()
  } catch {
    // статус — вспомогательная информация
  }
}

const stopPolling = () => {
  if (pollingStop.value) {
    pollingStop.value()
    pollingStop.value = null
  }
}

const watchExtraction = async () => {
  if (!selectedPageId.value || pollingStop.value) return
  const pageId = selectedPageId.value
  let stopped = false
  pollingStop.value = () => { stopped = true }
  await pollSourceStatus(api, {
    sourceType: 'diary',
    refId: pageId,
    onUpdate: (payload) => {
      if (!stopped && selectedPageId.value === pageId) sourceStatus.value = payload
    }
  })
  pollingStop.value = null
}

const reextract = async () => {
  if (!selectedPageId.value || extracting.value) return
  extracting.value = true
  saveError.value = ''
  try {
    const response = await api(`/knowledge/sources/diary/${selectedPageId.value}/extract`, {
      method: 'POST'
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось запустить извлечение')
    }
    sourceStatus.value = await response.json()
    watchExtraction()
  } catch (error) {
    saveError.value = error.message || 'Ошибка запуска извлечения'
  } finally {
    extracting.value = false
  }
}

watch(
  () => props.pageId,
  async (pageId) => {
    await loadSelectedPage(pageId)
  },
  { immediate: true }
)

onBeforeUnmount(stopPolling)
</script>

<style scoped>
.prose-preview :deep(h1) {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0.25rem 0 0.75rem;
}

.prose-preview :deep(h2) {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem;
}

.prose-preview :deep(p) {
  margin: 0.5rem 0;
  color: rgb(var(--c-text));
}

.prose-preview :deep(ul) {
  margin: 0.5rem 0;
  padding-left: 1.25rem;
  list-style: disc;
}

.prose-preview :deep(ol) {
  margin: 0.5rem 0;
  padding-left: 1.25rem;
  list-style: decimal;
}

.prose-preview :deep(li) {
  margin: 0.25rem 0;
}

.prose-preview :deep(code) {
  padding: 0.1rem 0.25rem;
  border-radius: 0.375rem;
  background: rgb(var(--c-input) / 0.6);
}

.prose-preview :deep(pre) {
  overflow: auto;
  padding: 0.75rem;
  border-radius: 0.75rem;
  border: 1px solid rgb(var(--c-border) / 0.4);
  background: rgb(var(--c-input) / 0.6);
}

.prose-preview :deep(a) {
  color: rgb(var(--c-accent));
  text-decoration: underline;
}
</style>
