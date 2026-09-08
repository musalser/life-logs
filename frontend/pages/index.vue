<template>
  <main>
    <LandingHeroSection />
    <LandingOracleSection />
    <LandingMentorsSection />
    <LandingAscentSection />
    <LandingCtaSection />

    <footer
      class="flex items-center justify-between px-6 py-8 font-display text-[11px] uppercase tracking-[0.3em] text-marble/35"
    >
      <span>Life Logs · MMXXVI</span>
      <nav class="flex gap-6">
        <NuxtLink to="/chat" class="transition-colors hover:text-gold">Chat</NuxtLink>
        <NuxtLink to="/diary" class="transition-colors hover:text-gold">Diary</NuxtLink>
      </nav>
    </footer>
  </main>
</template>

<script setup>
import Lenis from 'lenis'
import 'lenis/dist/lenis.css'

definePageMeta({ layout: 'landing' })

const title = 'Life Logs — Become who you were meant to be'
const description =
  'A private AI-read diary: the Oracle parses your entries, a council of mentors coaches your habits, and every goal becomes an ascent.'

useSeoMeta({
  title,
  description,
  ogTitle: title,
  ogDescription: description,
  ogImage: '/landing/temple-backdrop.jpg',
  twitterCard: 'summary_large_image'
})

useHead({
  link: [
    { rel: 'preload', as: 'image', href: '/landing/temple-backdrop.jpg' },
    { rel: 'preload', as: 'image', href: '/landing/hero-statue.png' },
    { rel: 'preload', as: 'image', href: '/landing/column-left.png' },
    { rel: 'preload', as: 'image', href: '/landing/column-right.png' },
    { rel: 'preload', as: 'image', href: '/landing/fog-1.png' }
  ]
})

const { $gsap, $ScrollTrigger } = useNuxtApp()
let lenis = null

function rafLenis(time) {
  lenis?.raf(time * 1000)
}

function refreshTriggers() {
  $ScrollTrigger.refresh()
}

onMounted(() => {
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    lenis = new Lenis({ lerp: 0.12 })
    lenis.on('scroll', $ScrollTrigger.update)
    $gsap.ticker.add(rafLenis)
    $gsap.ticker.lagSmoothing(0)
  }

  // re-measure pinned sections once images are in
  if (document.readyState === 'complete') refreshTriggers()
  else window.addEventListener('load', refreshTriggers, { once: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('load', refreshTriggers)
  $gsap.ticker.remove(rafLenis)
  lenis?.destroy()
  lenis = null
})
</script>
