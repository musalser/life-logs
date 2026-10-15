<template>
  <div
    role="button"
    tabindex="0"
    :draggable="true"
    class="lib-page"
    :class="{ 'lib-page--active': active }"
    :title="page.type === 'htr' ? 'Страница рукописи' : 'Запись дневника'"
    @click="$emit('select', page)"
    @keydown.enter.prevent="$emit('select', page)"
    @dragstart="$emit('drag-start', page, $event)"
    @dragend="$emit('drag-end', page, $event)"
  >
    <span class="lib-page__icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3">
        <path v-if="page.type === 'diary'" d="M12 6.9C9.7 5.2 6.1 4.9 3.8 5.7V18.3c2.3-.8 5.9-.5 8.2 1.2 2.3-1.7 5.9-2 8.2-1.2V5.7c-2.3-.8-5.9-.5-8.2 1.2z" />
        <path v-else d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 7V3.5L18.5 9H13z" />
      </svg>
    </span>
    <span class="lib-page__label">{{ page.title }}</span>
    <span v-if="page.status === 'CONFIRMED'" class="lib-page__lock" title="Подтверждена">
      <LockIcon :closed="true" class="h-3 w-3" />
    </span>
    <button
      type="button"
      class="lib-page__delete"
      title="Удалить страницу"
      @click.stop="$emit('delete', page)"
    >
      <svg viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3">
        <path d="M6 2h12l2 2v2H2V4l2-2zm2 6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H8zM10 10h4v10h-4V10z"></path>
      </svg>
    </button>
  </div>
</template>

<script setup>
defineProps({
  page: { type: Object, required: true },
  active: { type: Boolean, default: false }
})
defineEmits(['select', 'delete', 'drag-start', 'drag-end'])
</script>

<style scoped>
.lib-page {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.28rem 0.45rem;
  border-radius: 0.5rem;
  border: 1px solid transparent;
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
}
.lib-page:hover {
  background: rgb(var(--c-input) / 0.45);
  border-color: rgb(var(--c-primary) / 0.3);
}
.lib-page--active {
  background: rgb(var(--c-primary) / 0.16);
  border-color: rgb(var(--c-primary) / 0.55);
}
.lib-page__icon {
  flex: none;
  color: rgb(var(--c-accent));
  opacity: 0.8;
}
.lib-page__label {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.8rem;
}
.lib-page__lock {
  flex: none;
  color: rgb(var(--c-ok-text-soft));
  display: inline-flex;
}
.lib-page__delete {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.35rem;
  padding: 0.1rem 0.2rem;
  color: rgb(var(--c-muted));
  background: transparent;
  opacity: 0;
  pointer-events: none;
  transition: background 0.12s ease, color 0.12s ease, opacity 0.12s ease;
}
.lib-page:hover .lib-page__delete {
  opacity: 1;
  pointer-events: auto;
}
.lib-page__delete:hover {
  background: rgb(var(--c-error) / 0.2);
  color: rgb(var(--c-error));
}
</style>
