/*
 * The colour theme: dark (the palette the app shipped with) or light
 * (Solarized Light).
 *
 * The attribute lives on the app wrapper (`app.vue`), not on `<html>`: Vue has to
 * own the element whose attribute it renders, otherwise hydration reconciles the
 * server's value back over whatever the client wrote and the switch appears dead
 * (measured: the inline script set `light`, hydration reverted it to `dark`, and
 * clicking the toggle changed nothing on screen).
 *
 * Two things decide the first palette:
 *   - SHARED_THEME_INIT_SCRIPT in plugins/theme.ts runs in <head> and writes the
 *     same attribute onto <html> before the first paint, so a light-theme reload
 *     does not flash dark. Both places resolve the preference with the same code.
 *   - `resolveTheme()` below is that same rule as a function, used when the app
 *     mounts to seed the state.
 */
export const resolveTheme = () => {
  if (!import.meta.client) return 'dark'
  try {
    const stored = window.localStorage.getItem('life-logs-theme')
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // private mode: fall through to the OS preference
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export const useTheme = () => {
  const theme = useState('theme', resolveTheme)

  /** Keep the standalone <html> attribute (first paint, color-scheme) in step. */
  const syncDocumentTheme = (value) => {
    if (!import.meta.client) return
    document.documentElement.setAttribute('data-theme', value)
    document.documentElement.style.colorScheme = value
  }

  const applyTheme = (value) => {
    const next = value === 'light' ? 'light' : 'dark'
    theme.value = next

    if (import.meta.client) {
      try {
        window.localStorage.setItem('life-logs-theme', next)
      } catch {
        // private mode: the theme simply will not survive a reload
      }
    }

    return next
  }

  const initTheme = () => applyTheme(resolveTheme())

  const toggleTheme = () => applyTheme(theme.value === 'light' ? 'dark' : 'light')

  return { theme, applyTheme, initTheme, toggleTheme, syncDocumentTheme }
}
