<template>
  <div class="flex flex-col gap-6">
    <header class="flex flex-col gap-2 border-b border-app-border/40 pb-4 md:flex-row md:items-center md:justify-between">
      <div>
        <p class="text-sm uppercase tracking-widest text-app-accent">LIFE LOGS</p>
        <h1 class="text-3xl font-semibold">Мементо</h1>
      </div>
      <div class="flex flex-wrap items-center gap-4">
        <!--
          RAG (spec 11.3): the archive is searched by default, and the object
          selector narrows the search to one person's pages. Switching it off
          asks the model to answer from the conversation alone.
        -->
        <label class="flex items-center gap-2 text-sm font-medium text-app-muted">
          <input v-model="ragEnabled" type="checkbox" class="h-4 w-4 accent-current" />
          Искать по архиву
        </label>
        <label v-if="ragEnabled" class="flex items-center gap-2 text-sm font-medium text-app-muted">
          Объект
          <select
            v-model="subjectEntityId"
            class="rounded-md border border-app-border/60 bg-app-panel px-3 py-2 text-sm text-app-text focus:border-app-primary focus:outline-none"
            aria-label="Чьи записи искать"
          >
            <option :value="null">все записи</option>
            <option v-for="subject in subjects" :key="subject.id" :value="subject.id">
              {{ subject.name }}{{ subject.is_self ? ' (я)' : '' }}
            </option>
          </select>
        </label>
      <label class="flex items-center gap-2 text-sm font-medium text-app-muted">
        Тональность
        <select
          v-model="tone"
          class="rounded-md border border-app-border/60 bg-app-panel px-3 py-2 text-sm text-app-text focus:border-app-primary focus:outline-none"
          aria-label="Выберите тон дневника"
        >
          <option v-for="option in toneOptions" :key="option.value" :value="option.value">
            {{ option.label }}
          </option>
        </select>
      </label>
      </div>
    </header>

    <section class="flex h-[clamp(360px,68dvh,780px)] flex-col overflow-hidden rounded-2xl border border-app-border/40 bg-app-panel/60 shadow-lg shadow-app-primary/10 md:h-[clamp(420px,70dvh,840px)]">
      <div ref="chatContainer" class="no-scrollbar flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-6">
        <template v-if="chatStore.messages.length">
          <article
            v-for="message in chatStore.messages"
            :key="message.id"
            class="flex flex-col gap-2"
            :class="message.role === 'user' ? 'items-end' : 'items-start'"
          >
            <p class="text-xs uppercase tracking-wide text-app-muted">
              {{ message.role === 'user' ? 'Вы' : 'Дневник' }} — {{ formatTimestamp(message.timestamp) }}
            </p>
            <div
              class="max-w-xl whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed"
              :class="message.role === 'user' ? 'bg-app-user text-white' : 'bg-app-assistant text-app-text'"
            >
              {{ message.text }}
            </div>

            <!-- Where an answer came from: the reader must be able to check it -->
            <div v-if="message.role === 'assistant' && message.sources?.length" class="max-w-xl">
              <p class="mb-1 text-[10px] uppercase tracking-widest text-app-muted">Источники</p>
              <div class="flex flex-wrap gap-1.5">
                <button
                  v-for="source in message.sources"
                  :key="`${message.id}-${source.chunk_id}`"
                  type="button"
                  class="max-w-md truncate rounded-full border border-app-border/50 px-2.5 py-0.5 text-[11px] text-app-muted transition hover:border-app-primary/60 hover:text-app-accent"
                  :title="source.text"
                  @click="openSource(source)"
                >
                  {{ source.title }}: {{ shortText(source.text) }}
                </button>
              </div>
            </div>
          </article>
        </template>
        <p v-else class="rounded-xl border border-dashed border-app-border/50 px-4 py-6 text-center text-sm text-app-muted">
          Пока нет сообщений. Расскажите дневнику о планах и событиях за день.
        </p>
      </div>

      <footer class="shrink-0 border-t border-app-border/40 bg-app-panel/40 p-6">
        <form class="flex flex-col gap-4" @submit.prevent="handleSubmit">
          <textarea
            v-model="draft"
            class="min-h-[120px] resize-none rounded-2xl border border-app-border/40 bg-app-input/60 px-4 py-3 text-base text-app-text shadow-inner shadow-black/20 focus:border-app-primary focus:outline-none"
            placeholder="Что произошло сегодня? Какие цели движутся вперед?"
            @keydown="handleKeydown"
          ></textarea>
          <div class="flex flex-col gap-3 text-sm text-app-muted md:flex-row md:items-center md:justify-between">
            <p v-if="chatStore.error" class="text-app-error">{{ chatStore.error }}</p>
            <p v-else>Дневник сохранит вашу историю и поддержит тональность {{ toneLabel }}.</p>
            <button
              type="submit"
              class="inline-flex items-center justify-center rounded-full bg-app-primary px-6 py-2 font-semibold text-app-on-primary transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
              :disabled="chatStore.loading"
            >
              <svg
                v-if="chatStore.loading"
                class="mr-2 h-4 w-4 animate-spin text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path
                  class="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
              </svg>
              {{ chatStore.loading ? 'Дневник думает...' : 'Отправить' }}
            </button>
          </div>
        </form>
      </footer>
    </section>
  </div>
</template>

<script setup>
const router = useRouter()
const chatStore = useChatStore()
const draft = ref('')
const chatContainer = ref(null)

const toneOptions = [
  { value: 'critic', label: 'Критик' },
  { value: 'supportive', label: 'Поддержка друга' },
  { value: 'philosopher', label: 'Философ' }
]

const tone = computed({
  get: () => chatStore.tone,
  set: (value) => { chatStore.tone = value }
})

const toneLabel = computed(() => toneOptions.find((opt) => opt.value === tone.value)?.label ?? '')

// RAG controls live in the store so the setting survives a tab switch
const ragEnabled = computed({
  get: () => chatStore.ragEnabled,
  set: (value) => { chatStore.ragEnabled = value }
})

const subjectEntityId = computed({
  get: () => chatStore.subjectEntityId,
  set: (value) => { chatStore.subjectEntityId = value }
})

// Objects for the selector: «Я» first, then the people of the archive
const { subjects, load: loadSubjects, selectedId, selectDefault } = useKnowledgeSubjects()

const shortText = (text) => {
  const value = String(text || '').replace(/\s+/g, ' ').trim()
  return value.length > 70 ? `${value.slice(0, 70)}…` : value
}

/** A source of an answer opens the page it was read from. */
const openSource = (source) => {
  if (!source?.source_type) return
  router.push({
    path: '/library',
    query: { type: source.source_type, page: String(source.ref_id) }
  })
}

const scrollToBottom = async (behavior = 'smooth') => {
  await nextTick()
  const container = chatContainer.value
  if (!container) return
  container.scrollTo({ top: container.scrollHeight, behavior })
}

watch(
  () => chatStore.messages,
  () => { scrollToBottom() },
  { deep: true }
)

onMounted(async () => {
  await Promise.all([chatStore.loadHistory(), loadSubjects()])
  selectDefault()
  // the object chosen on the knowledge tab is the one the chat searches by
  subjectEntityId.value = selectedId.value
  scrollToBottom('auto')
})

const handleSubmit = () => {
  if (!draft.value.trim() || chatStore.loading) return
  chatStore.sendMessage(draft.value)
  draft.value = ''
}

const handleKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSubmit()
  }
}

const formatTimestamp = (timestamp) => {
  if (!timestamp) return 'сейчас'
  try {
    return new Date(timestamp).toLocaleString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short'
    })
  } catch {
    return 'сейчас'
  }
}
</script>
