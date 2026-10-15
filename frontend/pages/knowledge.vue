<template>
  <div class="flex flex-col gap-6">
    <header class="border-b border-app-border/40 pb-4">
      <div class="flex flex-wrap items-end justify-between gap-3">
        <div class="min-w-0">
          <h1 class="text-2xl font-semibold">Знания</h1>
          <p class="text-sm text-app-muted">
            Цели, круг общения, события, привычки и факты — от лица выбранного объекта
          </p>
        </div>

        <div class="flex flex-wrap items-center gap-2">
          <!-- Whose knowledge: «Я» by default, then the people of the archive -->
          <label class="flex items-center gap-2 text-xs text-app-muted">
            Объект
            <select
              :value="selectedId ?? ''"
              class="min-w-[12rem] rounded-xl border border-app-border/40 bg-app-input/60 px-3 py-2 text-sm text-app-text focus:border-app-primary focus:outline-none"
              :disabled="loadingObjects"
              @change="onSubjectChange($event.target.value)"
            >
              <option v-if="!subjects.length" value="">— нет объектов —</option>
              <option v-for="subject in subjects" :key="subject.id" :value="subject.id">
                {{ subject.name }}{{ subject.is_self ? ' (я)' : '' }}
              </option>
            </select>
          </label>

          <button
            type="button"
            class="inline-flex items-center justify-center rounded-full bg-app-primary px-5 py-2 text-sm font-semibold text-app-on-primary transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
            :disabled="loading"
            @click="loadKnowledge"
          >
            {{ loading ? 'Обновление...' : 'Обновить' }}
          </button>
        </div>
      </div>
    </header>

    <p v-if="error" class="rounded-xl border border-app-border/40 bg-app-input/40 px-4 py-3 text-sm text-app-error">
      {{ error }}
    </p>

    <p v-else-if="loading && !loaded" class="text-sm text-app-muted">Загрузка знаний...</p>

    <template v-else>
      <p v-if="subjectName" class="text-xs uppercase tracking-widest text-app-accent">
        Перспектива: {{ subjectName }}
        <span v-if="stats" class="ml-2 normal-case tracking-normal text-app-muted">
          {{ stats.facts }} фактов · {{ stats.relations }} связей · {{ stats.events }} событий ·
          {{ stats.sources }} источников обработано
        </span>
      </p>

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
                <button
                  v-if="note.source_text"
                  type="button"
                  class="mt-0.5 block text-left text-xs italic text-app-muted transition hover:text-app-accent"
                  :title="note.source_id ? 'Открыть источник' : 'Источник недоступен'"
                  :disabled="!note.source_id"
                  @click="openSource(note.source_id)"
                >
                  «{{ note.source_text }}»
                </button>
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
            :key="`${relation.direct ? 'edge' : 'path'}-${relation.entity_id}`"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4"
            :class="relation.direct ? '' : 'border-dashed'"
          >
            <div class="flex items-center gap-3">
              <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-app-primary/20 text-sm font-semibold text-app-accent">
                {{ initials(relation.person_name) }}
              </div>
              <div class="min-w-0">
                <p class="truncate text-sm font-semibold text-app-text">{{ relation.person_name }}</p>
                <!--
                  A relation read from another person's diary is not this
                  object's relation: the chain is shown as text («мама Бабы
                  Гали») and nothing is recomputed into «прабабушка».
                -->
                <p class="text-xs" :class="relation.direct ? 'text-app-accent' : 'text-app-muted'">
                  <template v-if="relation.direct">{{ relation.relation_type }}</template>
                  <template v-else>через: {{ relation.path_label }}</template>
                </p>
              </div>
            </div>
            <p v-if="relation.evidence_text" class="mt-2 text-xs italic text-app-muted">«{{ relation.evidence_text }}»</p>
            <button
              v-if="!relation.direct"
              type="button"
              class="mt-2 text-[11px] text-app-accent transition hover:underline"
              @click="openSourceByEntity(relation.entity_id)"
            >
              показать знания этого человека
            </button>
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
            <button
              v-if="event.source_text"
              type="button"
              class="mt-1 block text-left text-xs italic text-app-muted transition hover:text-app-accent"
              :title="sourceTitle(event.source_id)"
              @click="openSource(event.source_id)"
            >
              «{{ event.source_text }}»
            </button>
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

      <!-- Факты -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Факты</p>

        <p v-if="!knowledge.facts.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Фактов пока нет: они появляются, когда страница дневника или рукописи обработана.
        </p>

        <div v-else class="space-y-2">
          <article
            v-for="fact in knowledge.facts"
            :key="fact.id"
            class="rounded-xl border border-app-border/40 bg-app-panel/30 p-3"
          >
            <div class="flex flex-wrap items-center justify-between gap-2">
              <p class="text-sm text-app-text">
                <span class="font-semibold">{{ factObject(fact) }}</span>
                <span v-if="fact.time_text" class="text-app-muted"> · {{ fact.time_text }}</span>
              </p>
              <span v-if="fact.confidence != null" class="text-[11px] text-app-muted">
                уверенность {{ Math.round(fact.confidence * 100) }}%
              </span>
            </div>
            <button
              type="button"
              class="mt-1 block text-left text-xs italic text-app-muted transition hover:text-app-accent"
              :title="sourceTitle(fact.source_id)"
              @click="openSource(fact.source_id)"
            >
              «{{ fact.source_text }}»
            </button>
          </article>
        </div>
      </section>

      <!-- Персоны -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-widest text-app-accent">Персоны и объекты</p>

        <p v-if="!knowledge.entities.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Других персон пока нет.
        </p>

        <div v-else class="flex flex-wrap gap-2">
          <div
            v-for="person in knowledge.entities"
            :key="person.id"
            class="flex items-center gap-1"
          >
            <button
              type="button"
              class="flex items-center gap-2 rounded-full border border-app-border/40 px-3 py-1.5 text-sm transition hover:border-app-primary/60 hover:bg-app-input/40"
              :title="person.aliases.length ? `Ещё имена: ${person.aliases.join(', ')}` : 'Показать знания этого человека'"
              @click="selectSubject(person.id)"
            >
              <span class="font-semibold text-app-text">{{ person.name }}</span>
              <span v-if="person.aliases.length" class="text-[11px] text-app-muted">
                {{ person.aliases.join(', ') }}
              </span>
            </button>
            <button
              v-if="!person.is_self"
              type="button"
              class="flex items-center justify-center rounded-full border border-app-border/40 px-2 py-1.5 text-sm text-app-muted transition hover:border-app-primary/60 hover:text-app-accent"
              title="Этот объект — дубль существующего? Объединить с другим"
              @click="openMerge(person)"
            >
              ⤴
            </button>
          </div>
        </div>
      </section>

      <!-- Слияние объектов -->
      <div
        v-if="mergeSource"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
        @click.self="closeMerge"
      >
        <div class="w-full max-w-md rounded-2xl border border-app-border/60 bg-app-panel p-5 shadow-2xl">
          <p class="text-sm font-semibold">Объединить объект</p>
          <p class="mt-1 text-sm text-app-muted">
            Все знания объекта <b class="text-app-text">«{{ mergeSource.name }}»</b> будут перенесены в выбранный объект,
            а сам объект удалён. Используйте, когда извлечение создало дубль, не слившийся автоматически.
          </p>

          <label class="mt-4 block text-xs text-app-muted">
            В какой объект перенести
            <select
              v-model="mergeTargetId"
              class="mt-1 w-full rounded-xl border border-app-border/40 bg-app-input/60 px-3 py-2 text-sm text-app-text focus:border-app-primary focus:outline-none"
            >
              <option :value="null" disabled>— выберите объект —</option>
              <option
                v-for="subject in subjects"
                :key="subject.id"
                :value="subject.id"
                :disabled="subject.id === mergeSource.id"
              >
                {{ subject.name }}{{ subject.is_self ? ' (я)' : '' }}
              </option>
            </select>
          </label>

          <p v-if="mergeError" class="mt-3 text-sm text-app-error">{{ mergeError }}</p>

          <div class="mt-5 flex items-center justify-end gap-2">
            <button type="button" class="app-ghost px-4 py-2 text-sm" @click="closeMerge">Отмена</button>
            <button
              type="button"
              class="app-primary px-4 py-2 text-sm"
              :disabled="!mergeTargetId || mergeLoading"
              @click="applyMerge"
            >
              {{ mergeLoading ? 'Объединение…' : 'Объединить' }}
            </button>
          </div>
        </div>
      </div>

      <!-- Аналитика -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
          <p class="text-xs uppercase tracking-widest text-app-accent">Аналитика</p>
          <div class="flex items-center gap-2">
            <span
              v-if="analytics && analytics.stale"
              class="rounded-full bg-amber-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-amber-300"
              title="Знания изменились после генерации: аналитика устарела"
            >
              данные изменились
            </span>
            <span v-else-if="analytics && analytics.generated_at" class="text-[11px] text-app-muted">
              сгенерировано {{ formatDate(analytics.generated_at) }}
            </span>
            <button
              type="button"
              class="app-primary px-4 py-1.5 text-xs"
              :disabled="analyticsLoading"
              @click="refreshAnalytics"
            >
              {{ analyticsLoading ? 'Генерация…' : analyticsHasItems ? 'Обновить' : 'Сгенерировать' }}
            </button>
          </div>
        </div>

        <p v-if="analyticsError" class="rounded-xl border border-app-border/40 bg-app-input/40 px-3 py-2 text-sm text-app-error">
          {{ analyticsError }}
        </p>
        <p v-else-if="!analyticsHasItems" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Аналитики пока нет. Нажмите «Сгенерировать»: модель найдёт закономерности, вопросы и идеи
          по этим знаниям — каждый пункт со ссылками на записи.
        </p>
        <div v-else class="space-y-4">
          <div v-for="section in ANALYTICS_SECTIONS" :key="section.key">
            <template v-if="analytics.payload[section.key].length">
              <p class="mb-1.5 text-[11px] uppercase tracking-widest text-app-muted">{{ section.title }}</p>
              <ul class="space-y-2">
                <li
                  v-for="(item, index) in analytics.payload[section.key]"
                  :key="`${section.key}-${index}`"
                  class="rounded-xl border border-app-border/40 bg-app-panel/30 p-3"
                >
                  <p class="text-sm text-app-text">{{ item.text }}</p>
                  <div class="mt-1.5 flex flex-wrap gap-1.5">
                    <button
                      v-for="evidence in item.evidence"
                      :key="`${evidence.kind}-${evidence.id}`"
                      type="button"
                      class="rounded-full bg-app-input/60 px-2 py-0.5 text-[10px] text-app-muted transition hover:text-app-accent"
                      :title="evidence.label"
                      @click="openEvidence(evidence)"
                    >
                      {{ evidenceKindLabel(evidence.kind) }}: {{ shortLabel(evidence.label) }}
                    </button>
                  </div>
                </li>
              </ul>
            </template>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { authFetch } from '~/utils/auth-fetch'
import { useKnowledgeSubjects } from '~/composables/useKnowledge'

const { public: { apiBaseUrl } } = useRuntimeConfig()
const router = useRouter()

const {
  subjects,
  subjectsLoading,
  selectedId,
  selected,
  load: loadSubjects,
  selectDefault,
  remember,
  request
} = useKnowledgeSubjects()

const knowledge = ref({ goals: [], relations: [], events: [], habits: [], facts: [], entities: [] })
const stats = ref(null)
const subjectName = ref('')
const loading = ref(false)
const loaded = ref(false)
const error = ref('')

const analytics = ref(null)
const analyticsLoading = ref(false)
const analyticsError = ref('')

const ANALYTICS_SECTIONS = [
  { key: 'insights', title: 'Инсайты' },
  { key: 'questions', title: 'Вопросы' },
  { key: 'ideas', title: 'Идеи' },
  { key: 'conclusions', title: 'Выводы' }
]

const loadingObjects = computed(() => subjectsLoading.value)

const analyticsHasItems = computed(() =>
  Boolean(analytics.value) &&
  ANALYTICS_SECTIONS.some((section) => (analytics.value.payload?.[section.key] || []).length)
)

/** The same request path the manuscripts tab uses: token + refresh handling. */
const api = (path, init) => request(path, init)

const loadKnowledge = async () => {
  loading.value = true
  error.value = ''
  try {
    const query = selectedId.value ? `?subject_entity_id=${selectedId.value}` : ''
    const response = await api(`/knowledge${query}`)

    if (response.status === 401) {
      error.value = 'Войдите, чтобы видеть знания.'
      return
    }
    if (!response.ok) throw new Error('Не удалось загрузить знания')

    const payload = await response.json()
    knowledge.value = payload
    stats.value = payload.stats
    subjectName.value = payload.subject_name
    // the backend answers with the object it actually used (self by default),
    // so a first visit with an empty storage lands on «Я» without a second call
    if (payload.subject_entity_id) remember(payload.subject_entity_id)
    loaded.value = true
    await loadAnalytics()
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки знаний'
  } finally {
    loading.value = false
  }
}

const loadAnalytics = async () => {
  if (!selectedId.value) return
  analyticsError.value = ''
  try {
    const response = await api(`/knowledge/analytics?subject_entity_id=${selectedId.value}`)
    if (!response.ok) throw new Error('Не удалось загрузить аналитику')
    analytics.value = await response.json()
  } catch (err) {
    analyticsError.value = err.message || 'Ошибка загрузки аналитики'
  }
}

const refreshAnalytics = async () => {
  if (analyticsLoading.value) return
  analyticsLoading.value = true
  analyticsError.value = ''
  try {
    const response = await api('/knowledge/analytics/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject_entity_id: selectedId.value })
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось сгенерировать аналитику')
    }
    analytics.value = await response.json()
  } catch (err) {
    analyticsError.value = err.message || 'Ошибка генерации аналитики'
  } finally {
    analyticsLoading.value = false
  }
}

const onSubjectChange = async (value) => {
  const entityId = Number(value)
  remember(Number.isInteger(entityId) && entityId > 0 ? entityId : null)
  analytics.value = null
  await loadKnowledge()
}

const selectSubject = async (entityId) => {
  remember(entityId)
  analytics.value = null
  await loadKnowledge()
}

// --- слияние объектов (ручная дедупликация) ---

const mergeSource = ref(null)
const mergeTargetId = ref(null)
const mergeLoading = ref(false)
const mergeError = ref('')

const openMerge = (person) => {
  mergeSource.value = person
  mergeTargetId.value = null
  mergeError.value = ''
}

const closeMerge = () => {
  mergeSource.value = null
  mergeTargetId.value = null
  mergeError.value = ''
}

const applyMerge = async () => {
  if (!mergeSource.value || !mergeTargetId.value || mergeLoading.value) return
  mergeLoading.value = true
  mergeError.value = ''
  try {
    const response = await api('/knowledge/objects/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: mergeSource.value.id, target_id: mergeTargetId.value })
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось объединить объекты')
    }
    closeMerge()
    await loadSubjects()
    await loadKnowledge()
  } catch (err) {
    mergeError.value = err.message || 'Ошибка объединения'
  } finally {
    mergeLoading.value = false
  }
}

/**
 * Where a quote comes from: a manuscript page opens the reader on the line the
 * knowledge stands on, a diary page opens the entry. The knowledge response
 * carries the source id, not the page id, so the mapping is resolved lazily
 * through the source registry.
 */
const sourceIndex = ref({})

const rememberSource = (source) => {
  if (source && source.id) sourceIndex.value[source.id] = source
}

const openSource = async (sourceId) => {
  if (!sourceId) return
  let source = sourceIndex.value[sourceId]
  if (!source) {
    const response = await api(`/knowledge/source/${sourceId}`).catch(() => null)
    if (response && response.ok) {
      source = await response.json()
      rememberSource(source)
    }
  }
  if (!source) return
  router.push({
    path: '/library',
    query: { type: source.source_type, page: String(source.ref_id) }
  })
}

const sourceTitle = (sourceId) => {
  const source = sourceIndex.value[sourceId]
  if (!source) return 'Открыть источник'
  return source.source_type === 'htr'
    ? `Открыть страницу рукописи «${source.title || source.ref_id}»`
    : `Открыть запись дневника «${source.title || source.ref_id}»`
}

/** A person reached only through a chain: show their own knowledge. */
const openSourceByEntity = (entityId) => selectSubject(entityId)

const openEvidence = (evidence) => {
  if (evidence.kind === 'entity') {
    selectSubject(evidence.id)
    return
  }
  // other kinds open the section they belong to; the label is the anchor
  const target = evidence.kind === 'goal' ? 'Цели'
    : evidence.kind === 'event' ? 'Важные события'
      : evidence.kind === 'habit' ? 'Привычки'
        : evidence.kind === 'fact' ? 'Факты'
          : 'Круг общения'
  const element = [...document.querySelectorAll('section p')].find(
    (node) => node.textContent.trim().toUpperCase().startsWith(target.slice(0, 8).toUpperCase())
  )
  const section = element?.closest('section')
  if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const evidenceKindLabel = (kind) => ({
  fact: 'факт',
  goal: 'цель',
  event: 'событие',
  habit: 'привычка',
  relation: 'связь',
  entity: 'объект'
}[kind] || kind)

const shortLabel = (label) => {
  const text = String(label || '')
  return text.length > 60 ? `${text.slice(0, 60)}…` : text
}

const factObject = (fact) => {
  const object = fact.object_text || ''
  return `${fact.predicate}${object ? ` ${object}` : ''}`
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

onMounted(async () => {
  await loadSubjects()
  selectDefault()
  await loadKnowledge()
})
</script>
