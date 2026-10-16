import { ref } from 'vue'
import { authFetch } from '~/utils/auth-fetch'

/*
 * ЕДИНАЯ БИБЛИОТЕКА «Записи» — клиентская половина дерева
 * авторы (объекты) → вложенные папки → страницы.
 *
 * Дерево — общий для вкладки «Записи» кэш: редактор сообщает об изменениях, а
 * здесь перечитывается `GET /library/tree` и выполняются операции над папками,
 * страницами и авторами. Синглтон на уровне модуля — по образцу useKnowledgeSubjects.
 */

const tree = ref({ authors: [] })
const loading = ref(false)
const error = ref('')
let promise = null

export const useLibrary = () => {
  const { public: { apiBaseUrl } } = useRuntimeConfig()
  const api = (path, init) => authFetch(apiBaseUrl, `${apiBaseUrl}${path}`, init)

  const load = async ({ force = false } = {}) => {
    if (promise && !force) return promise
    loading.value = true
    error.value = ''
    promise = (async () => {
      try {
        const response = await api('/library/tree')
        if (response.status === 401) {
          error.value = 'Войдите, чтобы видеть записи.'
          tree.value = { authors: [] }
          return tree.value
        }
        if (!response.ok) throw new Error('Не удалось загрузить библиотеку')
        tree.value = await response.json()
        return tree.value
      } catch (e) {
        error.value = e.message || 'Ошибка загрузки библиотеки'
        return tree.value
      } finally {
        loading.value = false
        promise = null
      }
    })()
    return promise
  }

  const createAuthor = async (name) => {
    const response = await api('/htr/authors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось создать автора')
    }
    const created = await response.json()
    await load({ force: true })
    return created
  }

  const createFolder = async ({ name, ownerEntityId, parentId }) => {
    const response = await api('/library/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim(), owner_entity_id: ownerEntityId, parent_id: parentId }),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось создать папку')
    }
    await load({ force: true })
    return response.json()
  }

  const renameFolder = async (folderId, name) => {
    const response = await api(`/library/folders/${folderId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось переименовать папку')
    }
    await load({ force: true })
  }

  const deleteFolder = async (folderId) => {
    const response = await api(`/library/folders/${folderId}`, { method: 'DELETE' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось удалить папку')
    }
    await load({ force: true })
  }

  const movePage = async (page, folderId) => {
    const response = await api(`/library/pages/${page.type}/${page.ref_id}/folder`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_id: folderId }),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось переместить страницу')
    }
    await load({ force: true })
  }

  const deletePage = async (page) => {
    const path = page.type === 'diary' ? `/diary/pages/${page.ref_id}` : `/htr/pages/${page.ref_id}`
    const response = await api(path, { method: 'DELETE' })
    if (!response.ok && response.status !== 404) {
      throw new Error('Не удалось удалить страницу')
    }
    await load({ force: true })
  }

  const uploadPage = async (authorId, file, folderId = null) => {
    const form = new FormData()
    form.append('file', file)
    if (folderId != null) form.append('folder_id', String(folderId))
    const response = await api(`/htr/authors/${authorId}/pages`, { method: 'POST', body: form })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || `Не удалось загрузить «${file.name}»`)
    }
    const created = await response.json()
    await load({ force: true })
    return created
  }

  /*
   * PDF: каждая страница файла становится страницей рукописи, а все они —
   * папкой с именем файла (сервер создаёт её сам). Ответ содержит id страниц
   * в порядке документа, чтобы сразу открыть первую.
   */
  const uploadPdf = async (authorId, file, parentFolderId = null) => {
    const form = new FormData()
    form.append('file', file)
    if (parentFolderId != null) form.append('parent_folder_id', String(parentFolderId))
    const response = await api(`/htr/authors/${authorId}/pdf`, { method: 'POST', body: form })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || `Не удалось разобрать «${file.name}»`)
    }
    const created = await response.json()
    await load({ force: true })
    return created
  }

  return {
    tree,
    loading,
    error,
    load,
    createAuthor,
    createFolder,
    renameFolder,
    deleteFolder,
    movePage,
    deletePage,
    uploadPage,
    uploadPdf,
  }
}
