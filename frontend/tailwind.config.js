export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
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
        'app-error': '#f87171'
      },
      fontFamily: {
        body: ['Inter', 'system-ui', 'sans-serif'],
        heading: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif']
      }
    }
  },
  plugins: []
};
