<template>
  <div class="flex h-full flex-col gap-2">
    <div class="flex items-center justify-between gap-2">
      <p class="text-xs uppercase tracking-widest text-app-accent">Записи</p>
      <div class="flex items-center gap-1">
        <button type="button" class="lib-tool" title="Обновить" :disabled="loading" @click="reload">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="h-3.5 w-3.5">
            <path d="M4 12a8 8 0 1 1 2.3 5.7"></path>
            <path d="M4 20v-5h5"></path>
          </svg>
        </button>
        <button type="button" class="lib-tool lib-tool--primary" title="Добавить автора рукописи" @click="createAuthor">
          + автор
        </button>
      </div>
    </div>

    <p v-if="error" class="rounded-lg border border-app-border/40 bg-app-error/10 px-2 py-1.5 text-xs text-app-error">{{ error }}</p>
    <p v-if="notice" class="rounded-lg border border-app-border/40 bg-app-panel/60 px-2 py-1.5 text-xs text-app-muted">{{ notice }}</p>
    <p v-if="uploading" class="text-xs text-app-muted">{{ uploading }}</p>
    <p v-if="loading" class="text-xs text-app-muted">Загрузка…</p>

    <div v-else class="no-scrollbar -mr-1 flex-1 space-y-2 overflow-y-auto pr-1">
      <p v-if="!authors.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-4 text-center text-xs text-app-muted">
        Пока пусто. Добавьте автора рукописи — «Я» для дневника уже есть.
      </p>

      <section
        v-for="author in authors"
        :key="`author-${author.entity_id}`"
        class="lib-author"
        @dragover.prevent="authorDropTarget = author"
        @dragleave="authorDropTarget = null"
        @drop.prevent="onAuthorDrop(author)"
      >
        <div class="lib-author__row" :class="{ 'lib-drop': authorDropTarget === author }">
          <button
            type="button"
            class="lib-caret"
            :title="collapsed[author.entity_id] ? 'Развернуть' : 'Свернуть'"
            @click="collapsed[author.entity_id] = !collapsed[author.entity_id]"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3" :class="{ 'lib-caret--closed': collapsed[author.entity_id] }">
              <path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z" />
            </svg>
          </button>
          <span class="lib-author__name">{{ author.name }}</span>
          <span v-if="author.is_self" class="lib-badge">я</span>
          <div class="lib-actions">
            <button v-if="author.is_self" type="button" title="Новая запись дневника" @click.stop="$emit('select', { type: 'diary', ref_id: null })">✎</button>
            <button v-if="author.author_id" type="button" title="Загрузить страницы или PDF" @click.stop="pickUpload(author, null)">↑</button>
            <button type="button" title="Новая папка" @click.stop="newFolder(author.entity_id, null)">+</button>
          </div>
        </div>

        <div v-if="!collapsed[author.entity_id]" class="lib-author__children">
          <LibraryFolder
            v-for="folder in author.folders"
            :key="folder.id"
            :folder="folder"
            :collapsed="collapsed"
            :active-page="activePage"
            @select="$emit('select', $event)"
            @delete="deletePage"
            @drag-start="onDragStart"
            @drag-end="onDragEnd"
            @drop-page="onDropToFolder"
            @upload="(folder) => pickUpload(author, folder)"
            @new-subfolder="(folder) => newFolder(author.entity_id, folder.id)"
            @rename="renameFolder"
            @delete-folder="deleteFolder"
          />
          <LibraryPageRow
            v-for="page in author.pages"
            :key="`${page.type}-${page.ref_id}`"
            :page="page"
            :active="isActive(page)"
            @select="$emit('select', $event)"
            @delete="deletePage"
            @drag-start="onDragStart"
            @drag-end="onDragEnd"
          />
          <p v-if="!author.folders.length && !author.pages.length" class="lib-empty">
            {{ author.is_self ? 'Записей пока нет.' : 'Страниц пока нет.' }}
          </p>
        </div>
      </section>
    </div>

    <input ref="uploadInput" type="file" accept="image/*,application/pdf,.pdf" multiple class="hidden" @change="onUpload" />
  </div>
</template>

<script setup>
import { useLibrary } from '~/composables/useLibrary'

const props = defineProps({
  activePage: { type: Object, default: null }
})
const emit = defineEmits(['select'])

const {
  tree,
  loading,
  error,
  load,
  createAuthor: apiCreateAuthor,
  createFolder,
  renameFolder: apiRenameFolder,
  deleteFolder: apiDeleteFolder,
  movePage,
  deletePage: apiDeletePage,
  uploadPage,
  uploadPdf,
} = useLibrary()

const collapsed = reactive({})
const draggedPage = ref(null)
const authorDropTarget = ref(null)
const uploadInput = ref(null)
//: { author, folderId } — куда попадёт выбранный файл (folderId = null → корень)
const uploadTarget = ref(null)
const uploading = ref('')
const notice = ref('')

const authors = computed(() => tree.value.authors || [])

const reload = () => load({ force: true })

const isActive = (page) =>
  props.activePage && props.activePage.type === page.type && props.activePage.ref_id === page.ref_id

const clearError = () => { error.value = '' }

const createAuthor = async () => {
  const name = prompt('Имя автора рукописи (например, «Дед»):')
  if (name === null) return
  if (!name.trim()) return
  clearError()
  try {
    await apiCreateAuthor(name)
  } catch (e) {
    error.value = e.message || 'Не удалось создать автора'
  }
}

const newFolder = async (ownerEntityId, parentId) => {
  const name = prompt('Имя папки:')
  if (name === null) return
  if (!name.trim()) return
  clearError()
  try {
    await createFolder({ name, ownerEntityId, parentId })
  } catch (e) {
    error.value = e.message || 'Не удалось создать папку'
  }
}

const renameFolder = async (folder) => {
  const name = prompt('Имя папки:', folder.name)
  if (name === null) return
  if (!name.trim() || name.trim() === folder.name) return
  clearError()
  try {
    await apiRenameFolder(folder.id, name)
  } catch (e) {
    error.value = e.message || 'Не удалось переименовать папку'
  }
}

const deleteFolder = async (folder) => {
  if (!confirm(`Удалить папку «${folder.name}»? Вложенные папки и страницы останутся — они перейдут на уровень выше.`)) return
  clearError()
  try {
    await apiDeleteFolder(folder.id)
  } catch (e) {
    error.value = e.message || 'Не удалось удалить папку'
  }
}

const deletePage = async (page) => {
  const label = page.title || page.ref_id
  if (!confirm(`Удалить «${label}»?`)) return
  clearError()
  try {
    await apiDeletePage(page)
    emit('select', null)
  } catch (e) {
    error.value = e.message || 'Не удалось удалить страницу'
  }
}

const pickUpload = (author, folder) => {
  uploadTarget.value = { author, folderId: folder ? folder.id : null }
  uploadInput.value?.click()
}

const PDF_FILE = /\.pdf$/i
/** PDF picks up whole manuscripts; anything else is a page image. */
const isPdfFile = (file) => file.type === 'application/pdf' || PDF_FILE.test(file.name)

const onUpload = async (event) => {
  const files = Array.from(event.target.files || [])
  event.target.value = ''
  const target = uploadTarget.value
  uploadTarget.value = null
  if (!files.length || !target?.author?.author_id) return
  // one import at a time: the picker is not the only way a change event arrives
  if (uploading.value) return
  clearError()
  notice.value = ''
  const { author, folderId } = target
  let selected = null
  let pdfPages = 0
  try {
    for (const file of files) {
      if (isPdfFile(file)) {
        uploading.value = `Разбираю «${file.name}»…`
        // каждая страница PDF становится страницей, все они — в папке с именем файла
        const imported = await uploadPdf(author.author_id, file, folderId)
        pdfPages += imported.page_count || 0
        // открываем первую страницу файла: с неё и начинают читать
        if (imported.page_ids?.length) selected = imported.page_ids[0]
      } else {
        uploading.value = `Загружаю «${file.name}»…`
        const created = await uploadPage(author.author_id, file, folderId)
        selected = created.page_id
      }
    }
    if (selected != null) emit('select', { type: 'htr', ref_id: selected })
    if (pdfPages) {
      const where = folderId != null ? 'в выбранную папку' : 'в папку с именем файла'
      notice.value = `PDF разобран: страниц — ${pdfPages}, ${where}.`
    }
  } catch (e) {
    error.value = e.message || 'Не удалось загрузить страницу'
  } finally {
    uploading.value = ''
  }
}

// --- drag & drop ---

const onDragStart = (page) => {
  draggedPage.value = page
}
const onDragEnd = () => {
  draggedPage.value = null
  authorDropTarget.value = null
}
const onDropToFolder = async (folderId) => {
  const page = draggedPage.value
  if (!page) return
  clearError()
  try {
    await movePage(page, folderId)
  } catch (e) {
    error.value = e.message || 'Не удалось переместить страницу'
  } finally {
    onDragEnd()
  }
}
const onAuthorDrop = async (author) => {
  authorDropTarget.value = null
  const page = draggedPage.value
  if (!page) return
  clearError()
  try {
    await movePage(page, null)
  } catch (e) {
    error.value = e.message || 'Не удалось переместить страницу'
  } finally {
    onDragEnd()
  }
}

onMounted(() => load())
defineExpose({ reload })
</script>

<style scoped>
.lib-tool {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border-radius: 0.5rem;
  border: 1px solid rgb(var(--c-line) / 0.4);
  background: rgb(var(--c-surface) / 0.5);
  color: rgb(var(--c-on-surface));
  padding: 0.25rem 0.55rem;
  font-size: 0.72rem;
  transition: background 0.12s ease;
}
.lib-tool:hover:not(:disabled) {
  background: rgb(var(--c-soft) / 0.7);
}
.lib-tool--primary {
  border-color: rgb(var(--c-primary) / 0.5);
  color: rgb(var(--c-accent));
}

.lib-author {
  display: flex;
  flex-direction: column;
}
.lib-author__row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.4rem;
  border-radius: 0.55rem;
  border: 1px solid transparent;
  background: rgb(var(--c-panel) / 0.6);
}
.lib-author__row:hover {
  background: rgb(var(--c-input) / 0.4);
}
.lib-drop {
  border-color: rgb(var(--c-primary) / 0.7);
  background: rgb(var(--c-primary) / 0.14);
}
.lib-caret {
  flex: none;
  display: inline-flex;
  align-items: center;
  color: rgb(var(--c-muted));
  background: transparent;
  border: 0;
  padding: 0;
}
.lib-caret--closed {
  transform: rotate(-90deg);
}
.lib-author__name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.85rem;
  font-weight: 700;
}
.lib-badge {
  flex: none;
  border-radius: 999px;
  background: rgb(var(--c-primary) / 0.2);
  color: rgb(var(--c-accent));
  padding: 0 0.4rem;
  font-size: 0.62rem;
  font-weight: 700;
}
.lib-actions {
  display: flex;
  flex: none;
  align-items: center;
  gap: 0.05rem;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.lib-author__row:hover .lib-actions {
  opacity: 1;
  pointer-events: auto;
}
.lib-actions button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 0.3rem;
  color: rgb(var(--c-muted));
  background: transparent;
  font-size: 0.75rem;
}
.lib-actions button:hover {
  background: rgb(var(--c-soft) / 0.7);
  color: rgb(var(--c-on-surface));
}
.lib-author__children {
  margin-left: 0.4rem;
  padding-left: 0.45rem;
  border-left: 1px solid rgb(var(--c-line) / 0.3);
  display: flex;
  flex-direction: column;
  gap: 0.05rem;
}
.lib-empty {
  padding: 0.2rem 0.5rem;
  font-size: 0.72rem;
  color: rgb(var(--c-muted));
  opacity: 0.7;
}
</style>
