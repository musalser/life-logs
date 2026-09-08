import {
  AUTH_TOKEN_EVENT,
  AUTH_TOKEN_KEY,
  AUTH_USER_KEY,
  clearAuthStorage,
  refreshAccessToken,
  setAccessToken,
} from '~/utils/auth-fetch'

export const useAuthStore = defineStore('auth', () => {
  const { public: { apiBaseUrl } } = useRuntimeConfig()

  const token = ref(import.meta.client ? localStorage.getItem(AUTH_TOKEN_KEY) || '' : '')
  const user = ref(import.meta.client ? localStorage.getItem(AUTH_USER_KEY) || '' : '')
  const loading = ref(false)
  const error = ref(null)

  const isAuthenticated = computed(() => Boolean(token.value))

  if (import.meta.client) {
    window.addEventListener(AUTH_TOKEN_EVENT, (event) => {
      token.value = event.detail?.token || ''
      user.value = localStorage.getItem(AUTH_USER_KEY) || ''
    })
  }

  function clearError() {
    error.value = null
  }

  async function login(username, password) {
    loading.value = true
    clearError()
    try {
      const response = await fetch(`${apiBaseUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ username, password })
      })
      if (!response.ok) throw new Error('Неверный логин или пароль')
      const payload = await response.json()
      token.value = payload.access_token
      user.value = username
      setAccessToken(payload.access_token)
      localStorage.setItem(AUTH_USER_KEY, username)
    } catch (err) {
      error.value = err.message ?? 'Не удалось войти'
      throw err
    } finally {
      loading.value = false
    }
  }

  async function register(username, password, name, tone = 'coach') {
    loading.value = true
    clearError()
    try {
      const response = await fetch(`${apiBaseUrl}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ username, password, name, tone })
      })
      if (!response.ok) {
        if (response.status === 409) throw new Error('Пользователь с таким логином уже существует')
        throw new Error('Не удалось зарегистрироваться')
      }
      await login(username, password)
    } catch (err) {
      error.value = err.message ?? 'Ошибка регистрации'
      throw err
    } finally {
      loading.value = false
    }
  }

  async function tryRestoreSession() {
    if (token.value) return true
    try {
      const nextToken = await refreshAccessToken(apiBaseUrl)
      if (!nextToken) return false
      token.value = nextToken
      return true
    } catch {
      return false
    }
  }

  async function logout() {
    try {
      await fetch(`${apiBaseUrl}/auth/logout`, {
        method: 'POST',
        credentials: 'include'
      })
    } catch {
      // Local cleanup should still happen even if backend call fails.
    }
    token.value = ''
    user.value = ''
    clearAuthStorage()
  }

  return {
    token,
    user,
    loading,
    error,
    isAuthenticated,
    clearError,
    login,
    register,
    tryRestoreSession,
    logout,
  }
})
