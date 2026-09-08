<template>
  <div class="flex flex-col gap-6">
    <header class="border-b border-app-border/40 pb-4">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 class="text-2xl font-semibold">Знания</h1>
          <p class="text-sm text-app-muted">Цели, связи, события и привычки, извлечённые из дневника</p>
        </div>
        <button
          type="button"
          class="inline-flex items-center justify-center rounded-full bg-app-primary px-5 py-2 text-sm font-semibold text-app-text transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
          :disabled="loading"
          @click="loadKnowledge"
        >
          {{ loading ? 'Обновление...' : 'Обновить' }}
        </button>
      </div>
    </header>

    <p v-if="error" class="rounded-xl border border-app-border/40 bg-app-input/40 px-4 py-3 text-sm text-app-error">
      {{ error }}
    </p>

    <p v-else-if="loading && !loaded" class="text-sm text-app-muted">Загрузка знаний...</p>

    <template v-else>
      <!-- Цели -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Цели</p>

        <p v-if="!knowledge.goals.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Целей пока нет. Напишите в дневнике о своих планах — они появятся здесь.
        </p>

        <div v-else class="grid gap-3 md:grid-cols-2">
          <article
            v-for="goal in knowledge.goals"
            :key="goal.id"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4"
          >
            <div class="flex items-start justify-between gap-2">
              <h3 class="text-base font-semibold text-app-text">{{ goal.title }}</h3>
              <span class="rounded-full px-2.5 py-0.5 text-[11px] font-semibold" :class="goalStatusClass(goal.status)">
                {{ goalStatusLabel(goal.status) }}
              </span>
            </div>
            <p v-if="goal.description" class="mt-1 text-sm text-app-muted">{{ goal.description }}</p>

            <div v-if="goal.progress_notes.length" class="mt-3 space-y-2 border-l border-app-border/40 pl-3">
              <div v-for="note in goal.progress_notes" :key="note.id" class="text-sm">
                <div class="flex items-center gap-2">
                  <span class="rounded-full px-2 py-0.5 text-[10px] font-semibold" :class="progressKindClass(note.progress_kind)">
                    {{ progressKindLabel(note.progress_kind) }}
                  </span>
                  <span class="text-[11px] text-app-muted">{{ formatDate(note.created_at) }}</span>
                </div>
                <p v-if="note.note" class="mt-0.5 text-app-text">{{ note.note }}</p>
                <p v-if="note.source_text" class="mt-0.5 text-xs italic text-app-muted">«{{ note.source_text }}»</p>
              </div>
            </div>
          </article>
        </div>
      </section>

      <!-- Связи -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Круг общения</p>

        <p v-if="!knowledge.relations.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Связи пока не определены. Упоминайте людей в дневнике — карта связей построится сама.
        </p>

        <div v-else class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <article
            v-for="relation in knowledge.relations"
            :key="relation.id"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4"
          >
            <div class="flex items-center gap-3">
              <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-app-primary/20 text-sm font-semibold text-app-accent">
                {{ initials(relation.person_name) }}
              </div>
              <div class="min-w-0">
                <p class="truncate text-sm font-semibold text-app-text">{{ relation.person_name }}</p>
                <p class="text-xs text-app-accent">{{ relation.relation_type }}</p>
              </div>
            </div>
            <p v-if="relation.evidence_text" class="mt-2 text-xs italic text-app-muted">«{{ relation.evidence_text }}»</p>
          </article>
        </div>
      </section>

      <!-- События -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Важные события</p>

        <p v-if="!knowledge.events.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Событий пока нет.
        </p>

        <div v-else class="space-y-2">
          <article
            v-for="event in knowledge.events"
            :key="event.id"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4"
          >
            <div class="flex flex-wrap items-center justify-between gap-2">
              <h3 class="text-sm font-semibold text-app-text">{{ event.title }}</h3>
              <div class="flex items-center gap-2 text-[11px] text-app-muted">
                <span v-if="event.time_text">{{ event.time_text }}</span>
                <span>{{ formatDate(event.created_at) }}</span>
                <span
                  v-if="event.importance != null"
                  class="rounded-full bg-app-primary/15 px-2 py-0.5 font-semibold text-app-accent"
                >
                  важность {{ Math.round(event.importance * 100) }}%
                </span>
              </div>
            </div>
            <p v-if="event.description" class="mt-1 text-sm text-app-muted">{{ event.description }}</p>
            <p v-if="event.source_text" class="mt-1 text-xs italic text-app-muted">«{{ event.source_text }}»</p>
          </article>
        </div>
      </section>

      <!-- Привычки -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Привычки</p>

        <p v-if="!knowledge.habits.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Привычек пока нет.
        </p>

        <div v-else class="grid gap-3 md:grid-cols-2">
          <article
            v-for="habit in knowledge.habits"
            :key="habit.id"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4"
          >
            <div class="flex items-center justify-between gap-2">
              <h3 class="text-sm font-semibold text-app-text">{{ habit.title }}</h3>
              <span class="rounded-full bg-app-primary/15 px-2.5 py-0.5 text-[11px] font-semibold text-app-accent">
                {{ habit.logs.length }} отметок
              </span>
            </div>
            <div v-if="habit.logs.length" class="mt-2 space-y-1">
              <p v-for="log in habit.logs.slice(-3).reverse()" :key="log.id" class="text-xs text-app-muted">
                <span class="text-app-text">{{ formatDate(log.created_at) }}</span>
                <template v-if="log.note"> — {{ log.note }}</template>
              </p>
            </div>
          </article>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { authFetch } from '~/utils/auth-fetch'

const { public: { apiBaseUrl } } = useRuntimeConfig()

const knowledge = ref({ goals: [], relations: [], events: [], habits: [] })
const loading = ref(false)
const loaded = ref(false)
const error = ref('')

const loadKnowledge = async () => {
  loading.value = true
  error.value = ''
  try {
    const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/knowledge`)

    if (response.status === 401) {
      error.value = 'Войдите, чтобы видеть знания.'
      return
    }
    if (!response.ok) throw new Error('Не удалось загрузить знания')

    knowledge.value = await response.json()
    loaded.value = true
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки знаний'
  } finally {
    loading.value = false
  }
}

const goalStatusLabel = (status) => ({
  active: 'В работе',
  completed: 'Достигнута',
  abandoned: 'Отменена'
}[status] || status)

const goalStatusClass = (status) => ({
  active: 'bg-app-primary/20 text-app-accent',
  completed: 'bg-emerald-500/20 text-emerald-300',
  abandoned: 'bg-app-error/20 text-app-error'
}[status] || 'bg-app-input/60 text-app-muted')

const progressKindLabel = (kind) => ({
  started: 'старт',
  progress: 'прогресс',
  completed: 'достигнута',
  abandoned: 'отменена',
  mentioned: 'упоминание'
}[kind] || kind)

const progressKindClass = (kind) => ({
  started: 'bg-app-primary/20 text-app-accent',
  progress: 'bg-sky-500/20 text-sky-300',
  completed: 'bg-emerald-500/20 text-emerald-300',
  abandoned: 'bg-app-error/20 text-app-error',
  mentioned: 'bg-app-input/60 text-app-muted'
}[kind] || 'bg-app-input/60 text-app-muted')

const initials = (name) => {
  const parts = String(name || '').trim().split(/\s+/)
  return parts.slice(0, 2).map((part) => part[0]?.toUpperCase() || '').join('') || '?'
}

const formatDate = (value) => {
  if (!value) return ''
  try {
    return new Date(value).toLocaleString('ru-RU', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return ''
  }
}

onMounted(loadKnowledge)
</script>
