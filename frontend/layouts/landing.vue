<template>
  <div class="min-h-screen overflow-x-clip bg-obsidian font-serif text-marble antialiased selection:bg-gold/30">
    <slot />
  </div>
</template>

<script setup>
useHead({
  htmlAttrs: { lang: 'en' },
  link: [
    { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
    { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
    {
      rel: 'stylesheet',
      href: 'https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400;1,500&display=swap'
    }
  ]
})
</script>
