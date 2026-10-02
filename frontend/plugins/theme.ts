/*
 * The head script decides the palette before the first paint: it runs while the
 * document is still parsing and writes `data-theme` onto <html>, so a reload with
 * the light theme does not flash dark and the CSS variables are already right for
 * the first frame.
 *
 * It is intentionally dependency-free and wrapped in try/catch: where
 * localStorage is unavailable (private mode) the OS preference decides. The rule
 * is the same one `resolveTheme()` in composables/useTheme.js implements for the
 * app state — keep the two in step.
 */
export const SHARED_THEME_INIT_SCRIPT = `(function () {
  try {
    var stored = localStorage.getItem('life-logs-theme');
    var theme =
      stored === 'light' || stored === 'dark'
        ? stored
        : window.matchMedia('(prefers-color-scheme: light)').matches
          ? 'light'
          : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.style.colorScheme = theme;
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.documentElement.style.colorScheme = 'dark';
  }
})();`

export default defineNuxtPlugin((nuxtApp) => {
  const { initTheme, syncDocumentTheme } = useTheme()

  /*
   * No `htmlAttrs` here on purpose. The server has no access to the stored
   * preference, so rendering the attribute into <html> made hydration reconcile
   * the server's `dark` back over the client's `light`. The wrapper in app.vue
   * carries the attribute the app actually renders; this script only prepares the
   * document element for the very first frame.
   */
  useHead({
    script: [{ innerHTML: SHARED_THEME_INIT_SCRIPT, tagPosition: 'head' }]
  })

  nuxtApp.hook('app:mounted', () => {
    const theme = initTheme()
    syncDocumentTheme(theme)
  })
})
