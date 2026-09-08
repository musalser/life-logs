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
        'app-bg': '#020617',
        'app-surface': '#0f172a',
        'app-panel': '#111827',
        'app-border': '#1f2937',
        'app-text': '#f8fafc',
        'app-muted': '#94a3b8',
        'app-accent': '#a5b4fc',
        'app-primary': '#6366f1',
        'app-user': '#4f46e5',
        'app-assistant': '#1e293b',
        'app-input': '#020617',
        'app-error': '#f87171',
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
