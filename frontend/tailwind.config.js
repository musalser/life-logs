export default {
  content: [
    './components/**/*.{vue,js,ts}',
    './layouts/**/*.vue',
    './pages/**/*.vue',
    './app.vue',
    './plugins/**/*.{js,ts}',
    './nuxt.config.{js,ts}'
  ],
  theme: {
    extend: {
      colors: {
        /*
         * Every app colour comes from a CSS variable defined in
         * assets/css/main.css and re-declared per theme, so `bg-app-panel/60`
         * works in both the dark palette and Solarized Light without touching
         * the markup.
         */
        'app-bg': 'rgb(var(--c-bg) / <alpha-value>)',
        'app-surface': 'rgb(var(--c-surface) / <alpha-value>)',
        'app-panel': 'rgb(var(--c-panel) / <alpha-value>)',
        'app-border': 'rgb(var(--c-border) / <alpha-value>)',
        'app-text': 'rgb(var(--c-text) / <alpha-value>)',
        'app-muted': 'rgb(var(--c-muted) / <alpha-value>)',
        'app-dim': 'rgb(var(--c-dim) / <alpha-value>)',
        'app-soft': 'rgb(var(--c-soft) / <alpha-value>)',
        'app-on-surface': 'rgb(var(--c-on-surface) / <alpha-value>)',
        'app-accent': 'rgb(var(--c-accent) / <alpha-value>)',
        'app-accent-strong': 'rgb(var(--c-accent-strong) / <alpha-value>)',
        'app-primary': 'rgb(var(--c-primary) / <alpha-value>)',
        'app-on-primary': 'rgb(var(--c-on-primary) / <alpha-value>)',
        'app-user': 'rgb(var(--c-user) / <alpha-value>)',
        'app-assistant': 'rgb(var(--c-assistant) / <alpha-value>)',
        'app-input': 'rgb(var(--c-input) / <alpha-value>)',
        'app-error': 'rgb(var(--c-error) / <alpha-value>)',
        'app-ok': 'rgb(var(--c-ok) / <alpha-value>)',
        'app-ok-text': 'rgb(var(--c-ok-text) / <alpha-value>)',
        'app-ok-text-soft': 'rgb(var(--c-ok-text-soft) / <alpha-value>)',
        'app-warn': 'rgb(var(--c-warn) / <alpha-value>)',
        'app-warn-text': 'rgb(var(--c-warn-text) / <alpha-value>)',
        'app-oov': 'rgb(var(--c-oov) / <alpha-value>)',
        'app-oov-text': 'rgb(var(--c-oov-text) / <alpha-value>)',
        /*
         * The status chips in the manuscripts page are written with Tailwind's
         * stock names; re-pointing the family at the token keeps them on
         * palette (and readable) in Solarized Light.
         */
        emerald: {
          DEFAULT: 'rgb(var(--c-chip-ok-bg) / <alpha-value>)',
          100: 'rgb(var(--c-chip-ok-text-soft) / <alpha-value>)',
          200: 'rgb(var(--c-chip-ok-text-soft) / <alpha-value>)',
          300: 'rgb(var(--c-chip-ok-text) / <alpha-value>)',
          400: 'rgb(var(--c-chip-ok-border) / <alpha-value>)',
          500: 'rgb(var(--c-chip-ok-bg) / <alpha-value>)'
        },
        amber: {
          DEFAULT: 'rgb(var(--c-chip-warn) / <alpha-value>)',
          200: 'rgb(var(--c-chip-warn) / <alpha-value>)',
          300: 'rgb(var(--c-chip-warn) / <alpha-value>)',
          400: 'rgb(var(--c-chip-warn) / <alpha-value>)'
        },
        violet: {
          DEFAULT: 'rgb(var(--c-oov) / <alpha-value>)',
          300: 'rgb(var(--c-chip-oov-text) / <alpha-value>)'
        },
        marble: '#EDE8DF',
        gold: '#C9A227',
        bronze: '#8C6A3F',
        obsidian: '#0B0B0F'
      },
      fontFamily: {
        body: ['Inter', 'system-ui', 'sans-serif'],
        heading: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        display: ['Cinzel', 'Times New Roman', 'serif'],
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif']
      }
    }
  },
  plugins: []
};
