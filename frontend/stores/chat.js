import { authFetch } from '~/utils/auth-fetch'

export const useChatStore = defineStore('chat', () => {
  const { public: { apiBaseUrl } } = useRuntimeConfig()

  const messages = ref([])
  const tone = ref('critic')
  const loading = ref(false)
  const error = ref(null)

  const createMessageId = () =>
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)

  function clearError() {
    error.value = null
  }

  async function loadHistory() {
    try {
      clearError()
      const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/chat/history`)
      if (response.status === 401) {
        messages.value = []
        error.value = 'Войдите, чтобы загрузить историю'
        return
      }
      if (!response.ok) throw new Error('Не удалось загрузить сохраненную историю')

      const entries = await response.json()
      const historyMessages = []
      entries.forEach((entry) => {
        historyMessages.push({
          id: `entry-${entry.id}-user`,
          role: 'user',
          text: entry.text,
          timestamp: entry.created_at
        })
        if (entry.reply?.trim()) {
          historyMessages.push({
            id: `entry-${entry.id}-assistant`,
            role: 'assistant',
            text: entry.reply,
            timestamp: entry.created_at
          })
        }
      })
      messages.value = historyMessages
    } catch (err) {
      error.value = err.message ?? 'Не удалось загрузить историю'
    }
  }

  async function sendMessage(text) {
    const trimmed = text.trim()
    if (!trimmed) return

    clearError()
    messages.value.push({
      id: createMessageId(),
      role: 'user',
      text: trimmed,
      timestamp: new Date().toISOString()
    })
    loading.value = true

    const assistantMessageId = createMessageId()
    messages.value.push({
      id: assistantMessageId,
      role: 'assistant',
      text: '',
      timestamp: null,
      streaming: true
    })

    try {
      const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: trimmed, tone: tone.value, stream: true })
      })

      if (!response.ok) throw new Error('Не удалось получить ответ дневника')
      if (!response.body) throw new Error('Браузер не поддерживает потоковые ответы')

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let received = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        if (chunk) {
          received += chunk
          const msg = messages.value.find((m) => m.id === assistantMessageId)
          if (msg) msg.text += chunk
        }
      }

      const tail = decoder.decode()
      if (tail) {
        received += tail
        const msg = messages.value.find((m) => m.id === assistantMessageId)
        if (msg) msg.text += tail
      }

      const finalMsg = messages.value.find((m) => m.id === assistantMessageId)
      if (finalMsg) {
        if (!received.trim()) finalMsg.text = 'Ответ пустой'
        finalMsg.streaming = false
        finalMsg.timestamp = new Date().toISOString()
      }
    } catch (err) {
      error.value = err.message ?? 'Сервис временно недоступен'
      const msg = messages.value.find((m) => m.id === assistantMessageId)
      if (msg) {
        msg.text = '⚠️ Дневник не смог ответить. Попробуйте позже.'
        msg.streaming = false
        msg.timestamp = new Date().toISOString()
      }
    } finally {
      loading.value = false
    }
  }

  function resetChat() {
    messages.value = []
  }

  return { messages, tone, loading, error, clearError, loadHistory, sendMessage, resetChat }
})
