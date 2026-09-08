export default defineNuxtConfig({
  modules: ['@pinia/nuxt'],
  css: ['~/assets/css/main.css'],
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {}
    }
  },
  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.NUXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
    }
  },
  compatibilityDate: '2026-07-08'
})
