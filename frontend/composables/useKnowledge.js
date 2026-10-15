import { computed, ref } from 'vue'
import { authFetch } from '~/utils/auth-fetch'

/*
 * KNOWLEDGE API — the client half of the knowledge tabs.
 *
 * Four screens read the same objects and the same extraction status: the
 * manuscripts highlight what a page taught, the knowledge tab shows it from a
 * chosen perspective, the diary shows whether its page was read, the chat will
 * search it. All of them need three things, so they live here rather than in
 * each page:
 *
 *   - one place that knows the endpoints and the error shape;
 *   - one shared cache of *objects* («Я», «Баба Галя», …), because the selector
 *     appears on several tabs and must not refetch on every mount;
 *   - one polling helper for extraction statuses, because a status is a poll
 *     and not a fetch: it stops on a terminal state, on a timeout and on unmount.
 *
 * `useKnowledgeSubjects()` keeps the chosen object in localStorage: «Я» is the
 * default, but a reader who is studying their grandmother's diary should not
 * have to re-pick her on every tab change.
 */

export const SUBJECT_STORAGE_KEY = 'life_logs_knowledge_subject'

const subjects = ref([])
const subjectsLoaded = ref(false)
const subjectsLoading = ref(false)
const subjectsError = ref('')

let subjectsPromise = null

const TERMINAL_STATUSES = ['done', 'failed']

/** One object payload as the selector and the author dialog need it. */
export const useKnowledgeSubjects = () => {
  const selectedId = ref(null)
  const { public: { apiBaseUrl } } = useRuntimeConfig()

  const api = (path, init) => authFetch(apiBaseUrl, `${apiBaseUrl}${path}`, init)

  const load = async ({ force = false } = {}) => {
    if (subjectsLoaded.value && !force) return subjects.value
    if (subjectsPromise && !force) return subjectsPromise
    subjectsLoading.value = true
    subjectsError.value = ''
    subjectsPromise = (async () => {
      try {
        const response = await api('/knowledge/objects')
        if (response.status === 401) {
          subjectsError.value = 'Войдите, чтобы видеть объекты знаний.'
          return []
        }
        if (!response.ok) throw new Error('Не удалось загрузить объекты')
        subjects.value = await response.json()
        subjectsLoaded.value = true
        return subjects.value
      } catch (error) {
        subjectsError.value = error.message || 'Ошибка загрузки объектов'
        return []
      } finally {
        subjectsLoading.value = false
        subjectsPromise = null
      }
    })()
    return subjectsPromise
  }

  const storedId = () => {
    if (!import.meta.client) return null
    const raw = localStorage.getItem(SUBJECT_STORAGE_KEY)
    const value = Number(raw)
    return Number.isInteger(value) && value > 0 ? value : null
  }

  const remember = (entityId) => {
    selectedId.value = entityId
    if (import.meta.client) {
      if (entityId) localStorage.setItem(SUBJECT_STORAGE_KEY, String(entityId))
      else localStorage.removeItem(SUBJECT_STORAGE_KEY)
    }
  }

  /** Picks the stored object, else «Я», else the first one. */
  const selectDefault = () => {
    const remembered = storedId()
    const self = subjects.value.find((item) => item.is_self)
    const chosen =
      (remembered && subjects.value.find((item) => item.id === remembered)) || self || subjects.value[0]
    if (chosen) remember(chosen.id)
    return chosen || null
  }

  const selected = computed(
    () => subjects.value.find((item) => item.id === selectedId.value) || null
  )

  const createSubject = async ({ name, aliases = [], description = '' }) => {
    const response = await api('/knowledge/objects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, aliases, description })
    })
    if (response.status === 409) {
      const error = new Error('Объект с таким именем уже есть')
      error.status = 409
      throw error
    }
    if (!response.ok) throw new Error('Не удалось создать объект')
    const created = await response.json()
    subjects.value = [...subjects.value, created]
    return created
  }

  const renameSubject = async (entityId, { name, aliases }) => {
    const response = await api(`/knowledge/objects/${entityId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, aliases })
    })
    if (!response.ok) throw new Error('Не удалось переименовать объект')
    const updated = await response.json()
    subjects.value = subjects.value.map((item) => (item.id === updated.id ? updated : item))
    return updated
  }

  /** Raw request, exposed so pages share the token/refresh handling. */
  const request = api

  return {
    subjects,
    subjectsLoaded,
    subjectsLoading,
    subjectsError,
    selectedId,
    selected,
    load,
    selectDefault,
    remember,
    createSubject,
    renameSubject,
    request
  }
}

/**
 * Polls one extraction status until it settles.
 *
 * Returns a stop function: a page that navigates away or reopens the page must
 * be able to cancel the interval, and a status that never settles (a worker
 * that died) must give up instead of polling forever.
 */
export const pollSourceStatus = async (
  request,
  { sourceType, refId, intervalMs = 1500, timeoutMs = 120000, onUpdate }
) => {
  const started = Date.now()
  let stopped = false
  const stop = () => {
    stopped = true
  }

  while (!stopped) {
    let payload = null
    try {
      const response = await request(`/knowledge/sources/${sourceType}/${refId}/status`)
      if (response.status === 404) {
        stop()
        return null
      }
      if (response.ok) payload = await response.json()
    } catch {
      // сеть мигнула — попробуем на следующем тике
    }

    if (payload && onUpdate) onUpdate(payload)
    if (payload && TERMINAL_STATUSES.includes(payload.status)) {
      stop()
      return payload
    }
    if (Date.now() - started > timeoutMs) {
      stop()
      return payload
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  return null
}

/** Human labels shared by the badges on the manuscripts and diary tabs. */
export const sourceStatusLabel = (status) => ({
  pending: 'ожидает извлечения',
  running: 'извлекаем знания…',
  done: 'знания готовы',
  failed: 'не удалось извлечь'
}[status] || status)

export const sourceStatusTone = (status) => ({
  pending: 'bg-app-input/60 text-app-muted',
  running: 'bg-sky-500/20 text-sky-300',
  done: 'bg-emerald-500/20 text-emerald-300',
  failed: 'bg-app-error/20 text-app-error'
}[status] || 'bg-app-input/60 text-app-muted')
