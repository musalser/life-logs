<template>
  <div class="flex flex-col gap-4">
    <header class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 class="text-2xl font-semibold">Записи</h1>
        <p class="text-sm text-app-muted">
          Дневник и рукописи в одном месте: дерево авторов и папок слева, страница — справа
        </p>
      </div>
      <p v-if="selection" class="text-xs text-app-muted">
        {{ selection.type === 'diary' ? 'Запись дневника' : 'Страница рукописи' }}
        <span v-if="selection.ref_id">#{{ selection.ref_id }}</span>
        <span v-else>— новая</span>
      </p>
    </header>

    <div class="grid grid-cols-1 gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <aside class="h-[calc(100vh_-_190px)] overflow-hidden rounded-2xl border border-app-border/40 bg-app-panel/60 p-3 lg:sticky lg:top-4">
        <LibraryTree ref="treeRef" :active-page="activePage" @select="onSelect" />
      </aside>

      <main class="rounded-2xl border border-app-border/40 bg-app-panel/30 p-4 md:p-5">
        <DiaryEditor
          v-if="selection?.type === 'diary'"
          :page-id="selection.ref_id"
          @updated="onDiaryUpdated"
        />
        <ManuscriptEditor
          v-else-if="selection?.type === 'htr'"
          :page-id="selection.ref_id"
          @changed="reloadTree"
          @navigate="onSelect"
        />
        <div v-else class="flex min-h-[50vh] flex-col items-center justify-center gap-2 text-center">
          <p class="text-sm text-app-muted">Выберите страницу в дереве слева</p>
          <p class="text-xs text-app-muted">
            или создайте запись дневника (✎ у «Я») / загрузите страницы или PDF (↑ у автора или папки)
          </p>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
const route = useRoute()
const router = useRouter()

const treeRef = ref(null)
const selection = ref(null) // { type: 'diary' | 'htr', ref_id: number | null }

const activePage = computed(() => {
  if (!selection.value) return null
  return { type: selection.value.type, ref_id: selection.value.ref_id ?? null }
})

const onSelect = (page) => {
  if (!page) {
    selection.value = null
    router.replace({ query: {} })
    return
  }
  selection.value = { type: page.type, ref_id: page.ref_id ?? null }
  const query = {}
  if (page.type) query.type = page.type
  if (page.ref_id) query.page = String(page.ref_id)
  router.replace({ query })
}

const onDiaryUpdated = (page) => {
  selection.value = { type: 'diary', ref_id: page.id }
  router.replace({ query: { type: 'diary', page: String(page.id) } })
  reloadTree()
}

const reloadTree = () => treeRef.value?.reload()

onMounted(() => {
  const type = route.query.type
  const pageId = Number(route.query.page)
  if (type === 'diary' || type === 'htr') {
    selection.value = {
      type,
      ref_id: Number.isInteger(pageId) && pageId > 0 ? pageId : null,
    }
  }
})
</script>
