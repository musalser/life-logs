import { computed, ref } from 'vue'
import { authFetch } from '~/utils/auth-fetch'

/*
 * Граф связей: один запрос на весь архив.
 *
 * Корень выбирает клиент — бэкенд отдаёт граф целиком вместе с уровнями внутри
 * компонент родства и с диагностикой. Поэтому смена «от кого смотрим» не ходит
 * на сервер: уровни уже посчитаны, клиент лишь вычитает уровень выбранного.
 *
 * ETag — версия знаний: после перепрогона страницы или слияния объектов она
 * меняется, и граф обновляется; без изменений сервер отвечает 304 и картинка не
 * перерисовывается зря.
 */

const graph = ref(null)
const etag = ref('')
const loading = ref(false)
const error = ref('')

export const useRelationGraph = () => {
  const {
    public: { apiBaseUrl },
  } = useRuntimeConfig()

  const load = async ({ force = false } = {}) => {
    if (graph.value && !force) return graph.value
    loading.value = true
    error.value = ''
    try {
      const headers = force || !etag.value ? {} : { 'If-None-Match': etag.value }
      const response = await authFetch(apiBaseUrl, `${apiBaseUrl}/knowledge/graph`, { headers })
      if (response.status === 304) return graph.value
      if (!response.ok) throw new Error(`не удалось получить граф связей (${response.status})`)
      etag.value = response.headers.get('etag') || ''
      graph.value = await response.json()
      return graph.value
    } catch (cause) {
      error.value = cause?.message || 'не удалось получить граф связей'
      return graph.value
    } finally {
      loading.value = false
    }
  }

  const nodes = computed(() => graph.value?.nodes ?? [])
  const edges = computed(() => graph.value?.edges ?? [])
  const conflicts = computed(() => graph.value?.conflicts ?? [])
  const unplaced = computed(() => graph.value?.unplaced ?? [])

  const names = computed(() => new Map(nodes.value.map((node) => [node.id, node.name])))
  const nameOf = (id) => names.value.get(id) ?? `#${id}`

  const selectedId = ref(null)
  const selected = computed(() => nodes.value.find((node) => node.id === selectedId.value) ?? null)

  /** Подсветка: выбранный человек и его прямые соседи, остальное — фон. */
  const highlight = computed(() => {
    const id = selectedId.value
    const set = new Set()
    if (id === null || id === undefined) return set
    set.add(id)
    for (const edge of edges.value) {
      if (edge.from_id === id) set.add(edge.to_id)
      if (edge.to_id === id) set.add(edge.from_id)
    }
    return set
  })

  return {
    graph,
    nodes,
    edges,
    conflicts,
    unplaced,
    loading,
    error,
    selectedId,
    selected,
    highlight,
    names,
    nameOf,
    load,
  }
}
