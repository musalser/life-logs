<template>
  <div class="min-h-screen bg-app-bg text-app-text">
    <div class="mx-auto max-w-[1800px] px-4 py-6 md:px-6">
      <header class="mb-6 rounded-2xl border border-app-border/40 bg-app-panel/60 p-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center gap-4">
            <div>
              <p class="text-xs uppercase tracking-widest text-app-accent">LIFE LOGS</p>
              <p class="text-lg font-semibold">Журнал</p>
            </div>

            <nav class="flex items-center gap-2">
              <TabNavLink to="/chat" label="Чат" glyph="chat" />
              <TabNavLink to="/library" label="Записи" glyph="library" />
              <TabNavLink to="/knowledge" label="Знания" glyph="knowledge" />
            </nav>
          </div>

          <div class="flex items-center gap-2">
            <ThemeToggle />

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
        </div>
      </header>

      <main>
        <slot />
      </main>
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
const authStore = useAuthStore()
const chatStore = useChatStore()

const menuOpen = ref(false)
const authModalOpen = ref(false)
const authMode = ref('login')
const authForm = reactive({ username: '', password: '', name: '' })

const isAuthenticated = computed(() => authStore.isAuthenticated)
const authUser = computed(() => authStore.user || 'Пользователь')

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
    closeAuthModal()
  } catch {
    // Error text is managed by authStore.error
  }
}

const handleLogout = async () => {
  await authStore.logout()
  chatStore.resetChat()
  menuOpen.value = false
}

onMounted(() => {
  authStore.tryRestoreSession()
})
</script>

<style scoped>
.user-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  border: 1px solid rgb(var(--c-primary) / 0.45);
  width: 2.25rem;
  height: 2.25rem;
  color: rgb(var(--c-on-surface));
  background: rgb(var(--c-surface) / 0.55);
  transition: background 0.15s ease, border-color 0.15s ease;
}

.user-btn:hover {
  background: rgb(var(--c-soft) / 0.75);
  border-color: rgb(var(--c-primary) / 0.7);
}

.menu-card {
  position: absolute;
  top: calc(100% + 0.5rem);
  right: 0;
  z-index: 20;
  width: 12rem;
  border-radius: 0.75rem;
  border: 1px solid rgb(var(--c-primary) / 0.35);
  background: rgb(var(--c-surface) / 0.97);
  padding: 0.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  box-shadow: 0 10px 30px rgb(0 0 0 / 0.25);
}

.menu-user {
  margin: 0;
  padding: 0.45rem 0.55rem;
  font-size: 0.78rem;
  color: rgb(var(--c-accent));
  border-bottom: 1px solid rgb(var(--c-line) / 0.25);
}

.menu-action {
  border: 0;
  border-radius: 0.5rem;
  padding: 0.5rem 0.6rem;
  text-align: left;
  color: rgb(var(--c-on-surface));
  background: transparent;
}

.menu-action:hover {
  background: rgb(var(--c-primary) / 0.24);
}

.auth-input {
  width: 100%;
  border-radius: 0.75rem;
  border: 1px solid rgb(var(--c-line) / 0.45);
  background: rgb(var(--c-surface) / 0.45);
  color: rgb(var(--c-text));
  padding: 0.65rem 0.85rem;
}

.auth-input:focus {
  outline: none;
  border-color: rgb(var(--c-primary) / 0.8);
}

.auth-primary,
.auth-secondary {
  border-radius: 9999px;
  border: 1px solid transparent;
  padding: 0.5rem 1rem;
  font-weight: 600;
}

.auth-primary {
  background: rgb(var(--c-primary) / 0.95);
  color: rgb(var(--c-on-primary));
}

.auth-primary:disabled {
  opacity: 0.55;
}

.auth-secondary {
  background: rgb(var(--c-surface) / 0.5);
  border-color: rgb(var(--c-line) / 0.4);
  color: rgb(var(--c-on-surface));
}
</style>
