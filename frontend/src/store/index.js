import { createStore } from 'vuex';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const AUTH_TOKEN_KEY = 'life_logs_auth_token';
const AUTH_USER_KEY = 'life_logs_auth_user';

const getInitialToken = () => localStorage.getItem(AUTH_TOKEN_KEY) || '';
const getInitialUser = () => localStorage.getItem(AUTH_USER_KEY) || '';

const createMessageId = () =>
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export default createStore({
  state: () => ({
    messages: [],
    tone: 'critic',
    loading: false,
    error: null,
    authToken: getInitialToken(),
    authUser: getInitialUser(),
    authLoading: false,
    authError: null
  }),
  mutations: {
    addMessage(state, message) {
      state.messages.push(message);
    },
    setMessages(state, messages) {
      state.messages = messages;
    },
    appendToMessage(state, { id, chunk }) {
      const target = state.messages.find((msg) => msg.id === id);
      if (target) {
        target.text = `${target.text || ''}${chunk}`;
      }
    },
    finalizeMessage(state, { id, timestamp }) {
      const target = state.messages.find((msg) => msg.id === id);
      if (target) {
        target.streaming = false;
        target.timestamp = timestamp;
      }
    },
    overwriteMessage(state, { id, text }) {
      const target = state.messages.find((msg) => msg.id === id);
      if (target) {
        target.text = text;
      }
    },
    setTone(state, tone) {
      state.tone = tone;
    },
    setLoading(state, value) {
      state.loading = value;
    },
    setError(state, error) {
      state.error = error;
    },
    clearError(state) {
      state.error = null;
    },
    setAuthLoading(state, value) {
      state.authLoading = value;
    },
    setAuthError(state, error) {
      state.authError = error;
    },
    clearAuthError(state) {
      state.authError = null;
    },
    setAuth(state, { token, username }) {
      state.authToken = token;
      state.authUser = username;
      localStorage.setItem(AUTH_TOKEN_KEY, token);
      localStorage.setItem(AUTH_USER_KEY, username);
    },
    clearAuth(state) {
      state.authToken = '';
      state.authUser = '';
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
    },
    resetChat(state) {
      state.messages = [];
    }
  },
  actions: {
    async login({ commit }, { username, password }) {
      commit('setAuthLoading', true);
      commit('clearAuthError');
      try {
        const response = await fetch(`${apiBaseUrl}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ username, password })
        });

        if (!response.ok) {
          throw new Error('Неверный логин или пароль');
        }

        const payload = await response.json();
        commit('setAuth', {
          token: payload.access_token,
          username
        });
      } catch (error) {
        commit('setAuthError', error.message ?? 'Не удалось войти');
        throw error;
      } finally {
        commit('setAuthLoading', false);
      }
    },
    async register({ commit, dispatch }, { username, password, name, tone = 'coach' }) {
      commit('setAuthLoading', true);
      commit('clearAuthError');
      try {
        const response = await fetch(`${apiBaseUrl}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ username, password, name, tone })
        });

        if (!response.ok) {
          if (response.status === 409) {
            throw new Error('Пользователь с таким логином уже существует');
          }
          throw new Error('Не удалось зарегистрироваться');
        }

        await dispatch('login', { username, password });
      } catch (error) {
        commit('setAuthError', error.message ?? 'Ошибка регистрации');
        throw error;
      } finally {
        commit('setAuthLoading', false);
      }
    },
    logout({ commit }) {
      commit('clearAuth');
      commit('resetChat');
      commit('clearAuthError');
    },
    async loadHistory({ commit }) {
      try {
        commit('clearError');
        const headers = {};
        const token = localStorage.getItem(AUTH_TOKEN_KEY);
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }

        const response = await fetch(`${apiBaseUrl}/chat/history`, { headers });
        if (response.status === 401) {
          commit('setMessages', []);
          commit('setError', 'Войдите, чтобы загрузить историю');
          return;
        }
        if (!response.ok) {
          throw new Error('Не удалось загрузить сохраненную историю');
        }
        const entries = await response.json();
        const historyMessages = [];
        entries.forEach((entry) => {
          historyMessages.push({
            id: `entry-${entry.id}-user`,
            role: 'user',
            text: entry.text,
            timestamp: entry.created_at
          });
          if (entry.reply && entry.reply.trim()) {
            historyMessages.push({
              id: `entry-${entry.id}-assistant`,
              role: 'assistant',
              text: entry.reply,
              timestamp: entry.created_at
            });
          }
        });
        commit('setMessages', historyMessages);
      } catch (error) {
        commit('setError', error.message ?? 'Не удалось загрузить историю');
      }
    },
    async sendMessage({ commit, state }, text) {
      const trimmed = text.trim();
      if (!trimmed) {
        return;
      }

      commit('clearError');
      commit('addMessage', {
        id: createMessageId(),
        role: 'user',
        text: trimmed,
        timestamp: new Date().toISOString()
      });
      commit('setLoading', true);

      const assistantMessageId = createMessageId();
      commit('addMessage', {
        id: assistantMessageId,
        role: 'assistant',
        text: '',
        timestamp: null,
        streaming: true
      });

      try {
        const response = await fetch(`${apiBaseUrl}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(state.authToken ? { Authorization: `Bearer ${state.authToken}` } : {})
          },
          body: JSON.stringify({
            message: trimmed,
            tone: state.tone,
            stream: true
          })
        });

        if (!response.ok) {
          throw new Error('Не удалось получить ответ дневника');
        }

        if (!response.body) {
          throw new Error('Браузер не поддерживает потоковые ответы');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let received = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }
          const chunk = decoder.decode(value, { stream: true });
          if (chunk) {
            received += chunk;
            commit('appendToMessage', {
              id: assistantMessageId,
              chunk
            });
          }
        }

        const tail = decoder.decode();
        if (tail) {
          received += tail;
          commit('appendToMessage', {
            id: assistantMessageId,
            chunk: tail
          });
        }

        if (!received.trim()) {
          commit('overwriteMessage', {
            id: assistantMessageId,
            text: 'Ответ пустой'
          });
        }

        commit('finalizeMessage', {
          id: assistantMessageId,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        commit('setError', error.message ?? 'Сервис временно недоступен');
        commit('overwriteMessage', {
          id: assistantMessageId,
          text: '⚠️ Дневник не смог ответить. Попробуйте позже.'
        });
        commit('finalizeMessage', {
          id: assistantMessageId,
          timestamp: new Date().toISOString()
        });
      } finally {
        commit('setLoading', false);
      }
    }
  }
});
