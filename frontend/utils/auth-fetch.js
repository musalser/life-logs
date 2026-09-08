export const AUTH_TOKEN_KEY = 'life_logs_auth_token'
export const AUTH_USER_KEY = 'life_logs_auth_user'
export const AUTH_TOKEN_EVENT = 'life-logs-auth-token-updated'

let refreshPromise = null

const buildHeaders = (headers = {}) => {
  if (headers instanceof Headers) {
    return new Headers(headers)
  }
  return new Headers(headers)
}

export const clearAuthStorage = () => {
  if (!import.meta.client) return
  localStorage.removeItem(AUTH_TOKEN_KEY)
  localStorage.removeItem(AUTH_USER_KEY)
  window.dispatchEvent(new CustomEvent(AUTH_TOKEN_EVENT, { detail: { token: '' } }))
}

export const getAccessToken = () => {
  if (!import.meta.client) return ''
  return localStorage.getItem(AUTH_TOKEN_KEY) || ''
}

export const setAccessToken = (token) => {
  if (!import.meta.client) return
  if (!token) {
    localStorage.removeItem(AUTH_TOKEN_KEY)
    window.dispatchEvent(new CustomEvent(AUTH_TOKEN_EVENT, { detail: { token: '' } }))
    return
  }
  localStorage.setItem(AUTH_TOKEN_KEY, token)
  window.dispatchEvent(new CustomEvent(AUTH_TOKEN_EVENT, { detail: { token } }))
}

export const refreshAccessToken = async (apiBaseUrl) => {
  const response = await fetch(`${apiBaseUrl}/auth/refresh`, {
    method: 'POST',
    credentials: 'include'
  })

  if (!response.ok) {
    setAccessToken('')
    return ''
  }

  const payload = await response.json()
  const token = payload?.access_token || ''
  setAccessToken(token)
  return token
}

const doRefresh = async (apiBaseUrl) => {
  if (!refreshPromise) {
    refreshPromise = refreshAccessToken(apiBaseUrl)
      .catch(() => '')
      .finally(() => {
        refreshPromise = null
      })
  }
  return refreshPromise
}

export const authFetch = async (apiBaseUrl, input, init = {}, retryOnUnauthorized = true) => {
  const headers = buildHeaders(init.headers)
  const existingAuth = headers.get('Authorization')
  if (!existingAuth) {
    const token = getAccessToken()
    if (token) headers.set('Authorization', `Bearer ${token}`)
  }

  const requestInit = {
    ...init,
    headers,
    credentials: 'include'
  }

  let response = await fetch(input, requestInit)
  if (response.status !== 401 || !retryOnUnauthorized) {
    return response
  }

  const nextToken = await doRefresh(apiBaseUrl)
  if (!nextToken) {
    clearAuthStorage()
    return response
  }

  const retryHeaders = buildHeaders(init.headers)
  retryHeaders.set('Authorization', `Bearer ${nextToken}`)
  response = await fetch(input, {
    ...init,
    headers: retryHeaders,
    credentials: 'include'
  })
  return response
}
