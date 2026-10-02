<template>
  <button
    type="button"
    class="theme-toggle"
    :title="isLight ? 'Включить тёмную тему' : 'Включить светлую тему (Solarized Light)'"
    :aria-label="isLight ? 'Включить тёмную тему' : 'Включить светлую тему'"
    :aria-pressed="isLight"
    @click="toggleTheme"
  >
    <span class="theme-toggle__icon" aria-hidden="true">
      <!-- a sun in dark mode (offer the light), a moon in light mode -->
      <svg v-if="!isLight" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
        <circle cx="12" cy="12" r="4"></circle>
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6L17 7M7 17l-1.4 1.4"></path>
      </svg>
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z"></path>
      </svg>
    </span>
    <span class="theme-toggle__label">{{ isLight ? 'Светлая' : 'Тёмная' }}</span>
  </button>
</template>

<script setup>
const { theme, toggleTheme } = useTheme()

/*
 * The icon and the label are theme-dependent, and the server always renders the
 * dark pair — its `v-if` branches therefore disagreed with the client's stored
 * theme and hydration reported a mismatch inside this button. Rendering the
 * server's shape until the component is mounted keeps the two identical; the
 * palette itself already comes from the wrapper, so only these two glyphs settle
 * a moment later.
 */
const mounted = ref(false)
const isLight = computed(() => mounted.value && theme.value === 'light')

onMounted(() => {
  mounted.value = true
})
</script>

<style scoped>
.theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  border-radius: 9999px;
  border: 1px solid rgb(var(--c-primary) / 0.45);
  background: rgb(var(--c-surface) / 0.6);
  color: rgb(var(--c-on-surface));
  padding: 0.35rem 0.7rem;
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.theme-toggle:hover {
  background: rgb(var(--c-soft) / 0.8);
  border-color: rgb(var(--c-primary) / 0.75);
}

.theme-toggle__icon {
  display: inline-flex;
  width: 1rem;
  height: 1rem;
}

.theme-toggle__icon svg {
  width: 100%;
  height: 100%;
}

@media (max-width: 640px) {
  .theme-toggle__label {
    display: none;
  }
}
</style>
