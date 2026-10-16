<template>
  <div class="lib-folder">
    <div
      class="lib-folder__row"
      :class="{ 'lib-drop': isDropTarget }"
      @dragover.prevent="isDropTarget = true"
      @dragleave="isDropTarget = false"
      @drop.prevent="onDrop"
    >
      <button
        type="button"
        class="lib-caret"
        :title="collapsed[folder.id] ? 'Развернуть' : 'Свернуть'"
        @click="collapsed[folder.id] = !collapsed[folder.id]"
      >
        <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3" :class="{ 'lib-caret--closed': collapsed[folder.id] }">
          <path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z" />
        </svg>
      </button>
      <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5 lib-folder__icon" aria-hidden="true">
        <path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z" />
      </svg>
      <span class="lib-folder__name">{{ folder.name }}</span>
      <span class="lib-count">{{ folder.pages.length }}</span>
      <div class="lib-actions">
        <button type="button" title="Загрузить страницы или PDF в эту папку" @click.stop="$emit('upload', folder)">↑</button>
        <button type="button" title="Новая подпапка" @click.stop="$emit('new-subfolder', folder)">+</button>
        <button type="button" title="Переименовать" @click.stop="$emit('rename', folder)">✎</button>
        <button type="button" title="Удалить папку (содержимое останется)" @click.stop="$emit('delete-folder', folder)">✕</button>
      </div>
    </div>

    <div v-if="!collapsed[folder.id]" class="lib-folder__children">
      <LibraryFolder
        v-for="child in folder.children"
        :key="child.id"
        :folder="child"
        :collapsed="collapsed"
        :active-page="activePage"
        @select="$emit('select', $event)"
        @delete="$emit('delete', $event)"
        @drag-start="$emit('drag-start', $event)"
        @drag-end="$emit('drag-end', $event)"
        @drop-page="$emit('drop-page', $event)"
        @upload="$emit('upload', $event)"
        @new-subfolder="$emit('new-subfolder', $event)"
        @rename="$emit('rename', $event)"
        @delete-folder="$emit('delete-folder', $event)"
      />
      <LibraryPageRow
        v-for="page in folder.pages"
        :key="`${page.type}-${page.ref_id}`"
        :page="page"
        :active="isActive(page)"
        @select="$emit('select', $event)"
        @delete="$emit('delete', $event)"
        @drag-start="$emit('drag-start', $event)"
        @drag-end="$emit('drag-end', $event)"
      />
      <p v-if="!folder.children.length && !folder.pages.length" class="lib-empty">Пусто</p>
    </div>
  </div>
</template>

<script setup>
defineOptions({ name: 'LibraryFolder' })

const props = defineProps({
  folder: { type: Object, required: true },
  collapsed: { type: Object, required: true },
  activePage: { type: Object, default: null }
})
const emit = defineEmits([
  'select', 'delete', 'drag-start', 'drag-end', 'drop-page',
  'upload', 'new-subfolder', 'rename', 'delete-folder',
])

const isDropTarget = ref(false)

const onDrop = () => {
  isDropTarget.value = false
  emit('drop-page', props.folder.id)
}

const isActive = (page) =>
  props.activePage && props.activePage.type === page.type && props.activePage.ref_id === page.ref_id
</script>

<style scoped>
.lib-folder__row {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.24rem 0.4rem;
  border-radius: 0.5rem;
  border: 1px solid transparent;
}
.lib-folder__row:hover {
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
.lib-caret__closed {
  transform: rotate(-90deg);
}
.lib-folder__icon {
  flex: none;
  color: rgb(var(--c-accent));
  opacity: 0.7;
}
.lib-folder__name {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.8rem;
  font-weight: 600;
}
.lib-count {
  flex: none;
  font-size: 0.68rem;
  color: rgb(var(--c-muted));
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
.lib-folder__row:hover .lib-actions {
  opacity: 1;
  pointer-events: auto;
}
.lib-actions button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.15rem;
  height: 1.15rem;
  border-radius: 0.3rem;
  color: rgb(var(--c-muted));
  background: transparent;
  font-size: 0.7rem;
}
.lib-actions button:hover {
  background: rgb(var(--c-soft) / 0.7);
  color: rgb(var(--c-on-surface));
}
.lib-folder__children {
  margin-left: 0.85rem;
  padding-left: 0.4rem;
  border-left: 1px solid rgb(var(--c-line) / 0.25);
  display: flex;
  flex-direction: column;
  gap: 0.05rem;
}
.lib-empty {
  padding: 0.2rem 0.5rem;
  font-size: 0.7rem;
  color: rgb(var(--c-muted));
  opacity: 0.7;
}
</style>
