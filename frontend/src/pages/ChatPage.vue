<template>
  <div class="flex min-h-[calc(100vh-2rem)] flex-col gap-6">
    <header class="flex flex-col gap-2 border-b border-app-border/40 pb-4 md:flex-row md:items-center md:justify-between">
      <div>
        <p class="text-sm uppercase tracking-widest text-app-accent">LIFE LOGS</p>
        <h1 class="text-3xl font-semibold">Мементо</h1>
      </div>
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
    </header>

    <section class="flex flex-1 flex-col overflow-hidden rounded-2xl border border-app-border/40 bg-app-panel/60 shadow-lg shadow-app-primary/10">
      <div ref="chatContainer" class="no-scrollbar flex flex-1 flex-col gap-4 overflow-y-auto p-6">
        <template v-if="messages.length">
          <article
            v-for="message in messages"
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
          </article>
        </template>
        <p v-else class="rounded-xl border border-dashed border-app-border/50 px-4 py-6 text-center text-sm text-app-muted">
          Пока нет сообщений. Расскажите дневнику о планах и событиях за день.
        </p>
      </div>

      <footer class="border-t border-app-border/40 bg-app-panel/40 p-6">
        <form class="flex flex-col gap-4" @submit.prevent="handleSubmit">
          <textarea
            v-model="draft"
            class="min-h-[120px] rounded-2xl border border-app-border/40 bg-app-input/60 px-4 py-3 text-base text-app-text shadow-inner shadow-black/20 focus:border-app-primary focus:outline-none"
            placeholder="Что произошло сегодня? Какие цели движутся вперед?"
            @keydown="handleKeydown"
          ></textarea>
          <div class="flex flex-col gap-3 text-sm text-app-muted md:flex-row md:items-center md:justify-between">
            <p v-if="error" class="text-app-error">{{ error }}</p>
            <p v-else>Дневник сохранит вашу историю и поддержит тональность {{ toneLabel }}.</p>
            <button
              type="submit"
              class="inline-flex items-center justify-center rounded-full bg-app-primary px-6 py-2 font-semibold text-app-text transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
              :disabled="loading"
            >
              <svg
                v-if="loading"
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
              {{ loading ? 'Дневник думает...' : 'Отправить' }}
            </button>
          </div>
        </form>
      </footer>
    </section>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { useStore } from 'vuex';

const store = useStore();
const draft = ref('');
const chatContainer = ref(null);

const toneOptions = [
  { value: 'critic', label: 'Критик' },
  { value: 'supportive', label: 'Поддержка друга' },
  { value: 'philosopher', label: 'Философ' }
];

const messages = computed(() => store.state.messages);
const loading = computed(() => store.state.loading);
const error = computed(() => store.state.error);
const tone = computed({
  get: () => store.state.tone,
  set: (value) => store.commit('setTone', value)
});
const toneLabel = computed(() => toneOptions.find((opt) => opt.value === tone.value)?.label ?? '');

const scrollToBottom = async (behavior = 'smooth') => {
  await nextTick();
  const container = chatContainer.value;
  if (!container) {
    return;
  }
  container.scrollTo({
    top: container.scrollHeight,
    behavior
  });
};

watch(
  messages,
  () => {
    scrollToBottom();
  },
  { deep: true }
);

onMounted(async () => {
  await store.dispatch('loadHistory');
  scrollToBottom('auto');
});

const handleSubmit = () => {
  if (!draft.value.trim() || loading.value) {
    return;
  }
  store.dispatch('sendMessage', draft.value);
  draft.value = '';
};

const handleKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSubmit();
  }
};

const formatTimestamp = (timestamp) => {
  if (!timestamp) {
    return 'сейчас';
  }
  try {
    return new Date(timestamp).toLocaleString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short'
    });
  } catch {
    return 'сейчас';
  }
};
</script>
