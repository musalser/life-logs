<template>
  <div class="min-h-screen bg-app-bg text-app-text">
    <div class="mx-auto max-w-6xl px-4 py-6 md:px-8">
      <header class="mb-6 rounded-2xl border border-app-border/40 bg-app-panel/60 p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center gap-4">
            <div>
              <p class="text-xs uppercase tracking-widest text-app-accent">LIFE LOGS</p>
              <p class="text-lg font-semibold">Журнал</p>
            </div>

            <nav class="flex items-center gap-2">
              <NuxtLink to="/chat" class="nav-link" active-class="nav-link--active">Чат</NuxtLink>
              <NuxtLink to="/diary" class="nav-link" active-class="nav-link--active">Дневник</NuxtLink>
            </nav>
          </div>

          <div class="relative">
            <button
              type="button"
              class="user-btn"
              aria-label="Меню пользователя"
              @click="menuOpen = !menuOpen"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="h-5 w-5">
                <path d="M20 21a8 8 0 00-16 0"></path>
                <circle cx="12" cy="8" r="4"></circle>
              </svg>
            </button>

            <div v-if="menuOpen" class="menu-card">
              <p v-if="isAuthenticated" class="menu-user">{{ authUser }}</p>
              <button
                v-if="!isAuthenticated"
                type="button"
                class="menu-action"
                @click="openAuthModal('login')"
              >
                Войти
              </button>
              <button
                v-if="!isAuthenticated"
                type="button"
                class="menu-action"
                @click="openAuthModal('register')"
              >
                Зарегистрироваться
              </button>
              <button
                v-if="isAuthenticated"
                type="button"
                class="menu-action"
                @click="handleLogout"
              >
                Выйти
              </button>
            </div>
          </div>
        </div>
      </header>

      <div class="grid grid-cols-1 gap-6 md:grid-cols-[260px_1fr]">
        <aside class="rounded-2xl border border-app-border/40 bg-app-panel/60 p-4">
          <div class="mb-3 flex items-center justify-between gap-2">
            <p class="text-xs uppercase tracking-widest text-app-accent">История страниц</p>
            <p v-if="isDiaryRoute && isAuthenticated" class="text-xs text-app-muted">{{ diaryPages.length }} шт.</p>
          </div>

          <p v-if="!isDiaryRoute" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
            История доступна в разделе «Дневник».
          </p>

          <p v-else-if="!isAuthenticated" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
            Войдите, чтобы видеть историю страниц.
          </p>

          <p v-else-if="diaryHistoryError" class="mb-2 rounded-lg border border-app-border/40 bg-app-input/40 px-3 py-2 text-sm text-app-error">
            {{ diaryHistoryError }}
          </p>

          <div v-if="isDiaryRoute && isAuthenticated && diaryPages.length" class="no-scrollbar max-h-[70vh] space-y-2 overflow-y-auto pr-1">
            <button
              v-for="page in diaryPages"
              :key="page.id"
              type="button"
              class="w-full rounded-xl border px-3 py-2 text-left transition"
              :class="
                activePageId === page.id
                  ? 'border-app-primary/70 bg-app-primary/10'
                  : 'border-app-border/40 bg-app-panel/30 hover:border-app-primary/40 hover:bg-app-panel/50'
              "
              @click="openDiaryPage(page.id)"
            >
              <div class="flex items-center justify-between gap-2">
                <p class="text-sm font-medium text-app-text">{{ page.title }}</p>
                <div class="flex items-center gap-1">
                  <p class="text-[11px] text-app-muted">{{ formatTimestamp(page.created_at) }}</p>
                  <button
                    type="button"
                    class="ml-1 inline-flex items-center justify-center rounded-lg px-1.5 py-0.5 text-app-muted transition hover:bg-app-error/20 hover:text-app-error"
                    title="Удалить страницу"
                    @click.stop="handleDeleteDiaryPage(page.id)"
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5">
                      <path d="M6 2h12l2 2v2H2V4l2-2zm2 6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H8zM10 10h4v10h-4V10z"></path>
                    </svg>
                  </button>
                </div>
              </div>
              <p class="mt-1 line-clamp-2 text-xs text-app-muted">{{ previewText(page.content) }}</p>
            </button>
          </div>

          <p v-else-if="isDiaryRoute && isAuthenticated && !diaryHistoryLoading" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
            История пока пустая.
          </p>

          <p v-if="isDiaryRoute && diaryHistoryLoading" class="mt-2 text-xs text-app-muted">Загрузка...</p>
        </aside>

        <main class="rounded-2xl border border-app-border/40 bg-app-panel/30 p-4 md:p-6">
          <slot />
        </main>
      </div>
    </div>

    <div
      v-if="authModalOpen"
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4"
      @click.self="closeAuthModal"
    >
      <section class="w-full max-w-md rounded-2xl border border-app-border/50 bg-app-panel p-6 shadow-2xl">
        <header class="mb-4">
          <h2 class="text-xl font-semibold">
            {{ authMode === 'login' ? 'Вход' : 'Регистрация' }}
          </h2>
          <p class="mt-1 text-sm text-app-muted">
            {{ authMode === 'login' ? 'Введите логин и пароль' : 'Создайте новый аккаунт' }}
          </p>
        </header>

        <form class="space-y-3" @submit.prevent="submitAuth">
          <input
            v-model="authForm.username"
            class="auth-input"
            type="text"
            autocomplete="username"
            placeholder="Логин"
            required
          />

          <input
            v-model="authForm.password"
            class="auth-input"
            type="password"
            autocomplete="current-password"
            placeholder="Пароль"
            minlength="6"
            required
          />

          <input
            v-if="authMode === 'register'"
            v-model="authForm.name"
            class="auth-input"
            type="text"
            autocomplete="name"
            placeholder="Имя (необязательно)"
          />

          <p v-if="authStore.error" class="text-sm text-app-error">{{ authStore.error }}</p>

          <div class="mt-2 flex items-center justify-between gap-2">
            <button type="button" class="auth-secondary" @click="closeAuthModal">Отмена</button>
            <button type="submit" class="auth-primary" :disabled="authStore.loading">
              {{ authStore.loading ? 'Отправка...' : authMode === 'login' ? 'Войти' : 'Зарегистрироваться' }}
            </button>
          </div>
        </form>

        <button
          type="button"
          class="mt-4 text-sm text-app-muted underline underline-offset-4"
          @click="switchAuthMode"
        >
          {{ authMode === 'login' ? 'Нет аккаунта? Зарегистрироваться' : 'Уже есть аккаунт? Войти' }}
        </button>
      </section>
    </div>
  </div>
</template>

<script setup>
import { authFetch } from '~/utils/auth-fetch'

const authStore = useAuthStore()
const chatStore = useChatStore()
const route = useRoute()
const router = useRouter()
const { public: { apiBaseUrl } } = useRuntimeConfig()

const menuOpen = ref(false)
const authModalOpen = ref(false)
const diaryPages = ref([])
const diaryHistoryLoading = ref(false)
const diaryHistoryError = ref('')
const authMode = ref('login')
const authForm = reactive({ username: '', password: '', name: '' })

const isAuthenticated = computed(() => authStore.isAuthenticated)
const authUser = computed(() => authStore.user || 'Пользователь')
const isDiaryRoute = computed(() => route.name === 'diary')
const activePageId = computed(() => {
  const parsed = Number(route.query.page)
  return Number.isInteger(parsed) ? parsed : null
})

const previewText = (value) => value?.trim() || 'Пустая страница'

const formatTimestamp = (timestamp) => {
  if (!timestamp) return '—'
  try {
    return new Date(timestamp).toLocaleString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short'
    })
  } catch {
    return '—'
  }
}

const loadDiaryPages = async () => {
  if (!isDiaryRoute.value) return

  diaryHistoryLoading.value = true
  diaryHistoryError.value = ''
  try {
    if (!authStore.token) {
      diaryPages.value = []
      return
    }

    const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/diary/pages?limit=100&offset=0`)

    if (response.status === 401) {
      diaryPages.value = []
      diaryHistoryError.value = 'Сессия истекла. Войдите снова.'
      return
    }

    if (!response.ok) throw new Error('Не удалось загрузить историю')

    diaryPages.value = await response.json()
  } catch (error) {
    diaryHistoryError.value = error.message || 'Ошибка загрузки истории'
  } finally {
    diaryHistoryLoading.value = false
  }
}

const handleDiaryPagesUpdated = async () => {
  await loadDiaryPages()
}

const openDiaryPage = (pageId) => {
  router.push({ name: 'diary', query: { page: String(pageId) } })
}

const resetAuthForm = () => {
  authForm.username = ''
  authForm.password = ''
  authForm.name = ''
}

const openAuthModal = (mode) => {
  authMode.value = mode
  menuOpen.value = false
  authStore.clearError()
  authModalOpen.value = true
}

const closeAuthModal = () => {
  authModalOpen.value = false
  authStore.clearError()
  resetAuthForm()
}

const switchAuthMode = () => {
  authMode.value = authMode.value === 'login' ? 'register' : 'login'
  authStore.clearError()
}

const submitAuth = async () => {
  try {
    if (authMode.value === 'login') {
      await authStore.login(authForm.username, authForm.password)
    } else {
      await authStore.register(authForm.username, authForm.password, authForm.name)
    }
    await chatStore.loadHistory()
    await loadDiaryPages()
    closeAuthModal()
  } catch {
    // Error text is managed by authStore.error
  }
}

const handleLogout = async () => {
  await authStore.logout()
  chatStore.resetChat()
  diaryPages.value = []
  diaryHistoryError.value = ''
  menuOpen.value = false
}

const handleDeleteDiaryPage = async (pageId) => {
  if (!confirm('Вы уверены? Эту страницу нельзя будет восстановить.')) return

  try {
    const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/diary/pages/${pageId}`, {
      method: 'DELETE',
    })

    if (!response.ok) throw new Error('Не удалось удалить страницу')

    if (activePageId.value === pageId) {
      router.push({ name: 'diary' })
    }
    await loadDiaryPages()
  } catch (error) {
    diaryHistoryError.value = error.message || 'Ошибка удаления страницы'
  }
}

watch(
  () => [route.name, authStore.token],
  async () => {
    await loadDiaryPages()
  },
  { immediate: true }
)

onMounted(() => {
  authStore.tryRestoreSession()
  window.addEventListener('diary-pages-updated', handleDiaryPagesUpdated)
})

onBeforeUnmount(() => {
  window.removeEventListener('diary-pages-updated', handleDiaryPagesUpdated)
})
</script>

<style scoped>
.user-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  border: 1px solid rgba(99, 102, 241, 0.45);
  width: 2.25rem;
  height: 2.25rem;
  color: #e2e8f0;
  background: rgba(15, 23, 42, 0.55);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.user-btn:hover {
  background: rgba(30, 41, 59, 0.75);
  border-color: rgba(99, 102, 241, 0.7);
}

.menu-card {
  position: absolute;
  top: calc(100% + 0.5rem);
  right: 0;
  z-index: 20;
  width: 12rem;
  border-radius: 0.75rem;
  border: 1px solid rgba(99, 102, 241, 0.35);
  background: rgba(15, 23, 42, 0.95);
  padding: 0.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.menu-user {
  margin: 0;
  padding: 0.45rem 0.55rem;
  font-size: 0.78rem;
  color: #93c5fd;
  border-bottom: 1px solid rgba(148, 163, 184, 0.25);
}

.menu-action {
  border: 0;
  border-radius: 0.5rem;
  padding: 0.5rem 0.6rem;
  text-align: left;
  color: #e2e8f0;
  background: transparent;
}

.menu-action:hover {
  background: rgba(99, 102, 241, 0.24);
}

.nav-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-radius: 0.75rem;
  border: 1px solid rgba(31, 41, 55, 0.4);
  padding: 0.5rem 0.85rem;
  font-size: 0.875rem;
  color: #f8fafc;
  background: rgba(15, 23, 42, 0.3);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.nav-link:hover {
  background: rgba(15, 23, 42, 0.6);
  border-color: rgba(99, 102, 241, 0.4);
}

.nav-link--active {
  background: rgba(99, 102, 241, 0.18);
  border-color: rgba(99, 102, 241, 0.55);
}

.auth-input {
  width: 100%;
  border-radius: 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: rgba(15, 23, 42, 0.45);
  color: #f8fafc;
  padding: 0.65rem 0.85rem;
}

.auth-input:focus {
  outline: none;
  border-color: rgba(99, 102, 241, 0.8);
}

.auth-primary,
.auth-secondary {
  border-radius: 9999px;
  border: 1px solid transparent;
  padding: 0.5rem 1rem;
  font-weight: 600;
}

.auth-primary {
  background: rgba(99, 102, 241, 0.95);
  color: #fff;
}

.auth-primary:disabled {
  opacity: 0.55;
}

.auth-secondary {
  background: rgba(15, 23, 42, 0.5);
  border-color: rgba(148, 163, 184, 0.4);
  color: #e2e8f0;
}
</style>
