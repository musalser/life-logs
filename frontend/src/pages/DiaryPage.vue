<template>
  <div class="flex flex-col gap-6">
    <header class="border-b border-app-border/40 pb-4">
      <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h1 class="text-2xl font-semibold">Дневник</h1>
        </div>

        <div class="flex flex-col items-start gap-2 md:items-end">
          <div class="flex items-center gap-2">
            <button
              type="button"
              class="inline-flex items-center justify-center rounded-full bg-app-primary px-5 py-2 text-sm font-semibold text-app-text transition hover:bg-app-primary/80 disabled:cursor-not-allowed disabled:bg-app-primary/40"
              :disabled="!markdown.trim() || saving"
              @click="save"
            >
              {{ saving ? 'Сохранение...' : 'Сохранить' }}
            </button>
          </div>
          <p class="text-xs text-app-muted">{{ savedLabel }}</p>
        </div>
      </div>
    </header>

    <section class="flex flex-col gap-4">
      <div class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-wide text-app-muted">Markdown</p>
        <textarea
          v-model="markdown"
          class="min-h-[60vh] w-full resize-none rounded-xl border border-app-border/40 bg-app-input/60 px-4 py-3 text-base text-app-text shadow-inner shadow-black/20 focus:border-app-primary focus:outline-none"
          placeholder="# Что произошло сегодня?

- Главное событие дня…
- Маленькая победа…

## Цели
- …"
        ></textarea>
      </div>

      <div class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
        <p class="mb-3 text-xs uppercase tracking-wide text-app-muted">Превью</p>

        <div
          v-if="sanitizedHtml"
          class="prose-preview"
          v-html="sanitizedHtml"
        ></div>
        <p v-else class="text-sm text-app-muted">Начните печатать слева, чтобы увидеть превью.</p>
      </div>
    </section>

    <p v-if="saveError" class="rounded-xl border border-app-border/40 bg-app-input/40 px-4 py-3 text-sm text-app-error">
      {{ saveError }}
    </p>

    <details class="rounded-2xl border border-app-border/40 bg-app-panel/40 p-4">
      <summary class="cursor-pointer select-none text-sm font-semibold text-app-text">
        Шпаргалка по Markdown
      </summary>
      <div class="mt-3 grid gap-3 text-sm text-app-muted md:grid-cols-2">
        <div class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4">
          <p class="mb-2 text-xs uppercase tracking-wide text-app-muted">Заголовки</p>
          <pre class="no-scrollbar overflow-x-auto whitespace-pre-wrap rounded-lg border border-app-border/40 bg-app-input/60 p-3 text-app-text"><code># Заголовок 1
## Заголовок 2
### Заголовок 3</code></pre>
        </div>

        <div class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4">
          <p class="mb-2 text-xs uppercase tracking-wide text-app-muted">Списки</p>
          <pre class="no-scrollbar overflow-x-auto whitespace-pre-wrap rounded-lg border border-app-border/40 bg-app-input/60 p-3 text-app-text"><code>- Пункт
- Ещё пункт

1. Первый
2. Второй</code></pre>
        </div>

        <div class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4">
          <p class="mb-2 text-xs uppercase tracking-wide text-app-muted">Выделение</p>
          <pre class="no-scrollbar overflow-x-auto whitespace-pre-wrap rounded-lg border border-app-border/40 bg-app-input/60 p-3 text-app-text"><code>**жирный**
*курсив*
~~зачёркнутый~~
`инлайн код`</code></pre>
        </div>

        <div class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4">
          <p class="mb-2 text-xs uppercase tracking-wide text-app-muted">Ссылки и цитаты</p>
          <pre class="no-scrollbar overflow-x-auto whitespace-pre-wrap rounded-lg border border-app-border/40 bg-app-input/60 p-3 text-app-text"><code>[текст ссылки](https://example.com)

> Это цитата
> в две строки</code></pre>
        </div>

        <div class="rounded-xl border border-app-border/40 bg-app-panel/30 p-4 md:col-span-2">
          <p class="mb-2 text-xs uppercase tracking-wide text-app-muted">Код-блок</p>
          <pre class="no-scrollbar overflow-x-auto whitespace-pre-wrap rounded-lg border border-app-border/40 bg-app-input/60 p-3 text-app-text"><code>```js
console.log('hello')
```</code></pre>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({
  gfm: true,
  breaks: true
});

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const AUTH_TOKEN_KEY = 'life_logs_auth_token';
const route = useRoute();
const router = useRouter();

const markdown = ref('');
const lastSavedAt = ref(null);
const saving = ref(false);
const saveError = ref('');
const selectedPageId = ref(null);

const sanitizedHtml = computed(() => {
  const source = markdown.value.trim();
  if (!source) {
    return '';
  }
  const raw = marked.parse(source);
  return DOMPurify.sanitize(raw);
});

const savedLabel = computed(() => {
  if (!lastSavedAt.value) {
    return 'Не сохранено';
  }
  try {
    return `Сохранено: ${new Date(lastSavedAt.value).toLocaleString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short'
    })}`;
  } catch {
    return 'Сохранено';
  }
});

const getAuthToken = () => localStorage.getItem(AUTH_TOKEN_KEY) || '';

const loadSelectedPage = async (pageQuery) => {
  saveError.value = '';
  const pageId = Number(pageQuery);
  if (!Number.isInteger(pageId)) {
    selectedPageId.value = null;
    markdown.value = '';
    lastSavedAt.value = null;
    return;
  }

  const token = getAuthToken();
  if (!token) {
    saveError.value = 'Войдите в аккаунт, чтобы открыть страницу.';
    return;
  }

  try {
    const response = await fetch(`${apiBaseUrl}/diary/pages/${pageId}`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });

    if (response.status === 401) {
      saveError.value = 'Сессия истекла. Войдите снова.';
      return;
    }

    if (!response.ok) {
      throw new Error('Не удалось загрузить страницу');
    }

    const page = await response.json();
    selectedPageId.value = page.id;
    markdown.value = page.content || '';
    lastSavedAt.value = page.created_at || null;
  } catch (error) {
    saveError.value = error.message || 'Ошибка загрузки страницы';
  }
};

const save = async () => {
  const text = markdown.value.trim();
  if (!text) {
    return;
  }

  saving.value = true;
  saveError.value = '';
  try {
    const token = getAuthToken();
    if (!token) {
      saveError.value = 'Войдите в аккаунт, чтобы сохранять страницы.';
      return;
    }

    const isUpdate = Boolean(selectedPageId.value);
    const response = await fetch(
      isUpdate ? `${apiBaseUrl}/diary/pages/${selectedPageId.value}` : `${apiBaseUrl}/diary/pages`,
      {
        method: isUpdate ? 'PUT' : 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ content: text })
      }
    );

    if (response.status === 401) {
      saveError.value = 'Сессия истекла. Войдите снова.';
      return;
    }

    if (!response.ok) {
      throw new Error('Не удалось сохранить страницу');
    }

    const savedPage = await response.json();
    selectedPageId.value = savedPage.id;
    markdown.value = savedPage.content || text;
    if (String(route.query.page || '') !== String(savedPage.id)) {
      await router.replace({
        name: 'diary',
        query: { page: String(savedPage.id) }
      });
    }

    lastSavedAt.value = savedPage.created_at || new Date().toISOString();
    window.dispatchEvent(new CustomEvent('diary-pages-updated'));
  } catch {
    saveError.value = 'Ошибка сохранения страницы';
  } finally {
    saving.value = false;
  }
};

watch(
  () => route.query.page,
  async (pageQuery) => {
    await loadSelectedPage(pageQuery);
  },
  { immediate: true }
);
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
  color: #f8fafc;
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
  background: rgba(2, 6, 23, 0.6);
}

.prose-preview :deep(pre) {
  overflow: auto;
  padding: 0.75rem;
  border-radius: 0.75rem;
  border: 1px solid rgba(31, 41, 55, 0.4);
  background: rgba(2, 6, 23, 0.6);
}

.prose-preview :deep(a) {
  color: #a5b4fc;
  text-decoration: underline;
}
</style>
