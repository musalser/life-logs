<template>
  <section ref="root" class="relative overflow-hidden bg-obsidian">
    <div class="relative md:h-[100svh] md:overflow-hidden">
      <header class="px-6 pt-20 text-center md:absolute md:inset-x-0 md:top-0 md:z-10 md:pt-16">
        <p data-l="h" class="landing-eyebrow">The Council</p>
        <h2 data-l="h" class="mt-4 font-display text-3xl font-semibold text-marble sm:text-5xl">
          Counsel of the Mentors
        </h2>
      </header>

      <div data-l="track" class="flex flex-col md:h-full md:w-[300vw] md:flex-row">
        <article
          v-for="m in mentors"
          :key="m.name"
          class="mentor-panel flex w-full flex-col items-center justify-center gap-8 px-6 py-16 md:h-full md:w-screen md:flex-row md:gap-16 md:py-0 md:pt-24"
        >
          <img
            :src="m.img"
            :alt="m.name"
            loading="lazy"
            decoding="async"
            class="mentor-bust h-[36svh] w-auto max-w-full object-contain drop-shadow-[0_25px_60px_rgba(0,0,0,0.8)] md:h-[52svh]"
          />
          <div class="max-w-md text-center md:text-left">
            <p class="mentor-kicker font-display text-sm tracking-[0.35em] text-gold/70">{{ m.numeral }}</p>
            <h3 class="mentor-name mt-3 font-display text-2xl text-marble sm:text-4xl">{{ m.name }}</h3>
            <p class="mentor-desc mt-4 font-serif text-lg italic text-marble/70">{{ m.text }}</p>
          </div>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
const mentors = [
  {
    numeral: 'I',
    name: 'The Coach',
    img: '/landing/bust-coach.png',
    text: 'Forges discipline. He builds your routines, counts your streaks, and never lets a good habit die quietly.'
  },
  {
    numeral: 'II',
    name: 'The Sage',
    img: '/landing/bust-sage.png',
    text: 'Sees the pattern. He weighs months of your entries and returns them as insight you can act on.'
  },
  {
    numeral: 'III',
    name: 'The Healer',
    img: '/landing/bust-healer.png',
    text: 'Keeps the balance. She hears what you leave unsaid, names the weight you carry, and counsels rest before you break.'
  }
]

const root = ref(null)
const { $gsap } = useNuxtApp()
let mm

onMounted(() => {
  const el = (name) => root.value.querySelector(`[data-l="${name}"]`)
  const q = (sel) => root.value.querySelectorAll(sel)

  mm = $gsap.matchMedia()
  mm.add(
    {
      full: '(min-width: 768px) and (prefers-reduced-motion: no-preference)',
      lite: '(max-width: 767.98px) and (prefers-reduced-motion: no-preference)'
    },
    (ctx) => {
      const { full } = ctx.conditions
      const panels = [...q('.mentor-panel')]
      const panelParts = (panel) => panel.querySelectorAll('img, .mentor-kicker, .mentor-name, .mentor-desc')

      if (full) {
        const tl = $gsap.timeline({
          scrollTrigger: { trigger: root.value, start: 'top top', end: '+=250%', pin: true, scrub: 1 }
        })
        tl.from(q('[data-l="h"]'), { opacity: 0, y: 30, stagger: 0.1, duration: 0.3 })
          .from(panelParts(panels[0]), { opacity: 0, y: 40, stagger: 0.06, duration: 0.35 }, 0.1)
          .to(el('track'), { xPercent: -100 / 3, ease: 'power1.inOut', duration: 1 }, '+=0.3')
          .from(panelParts(panels[1]), { opacity: 0, y: 40, stagger: 0.06, duration: 0.35 }, '<+0.6')
          .to(el('track'), { xPercent: -200 / 3, ease: 'power1.inOut', duration: 1 }, '+=0.3')
          .from(panelParts(panels[2]), { opacity: 0, y: 40, stagger: 0.06, duration: 0.35 }, '<+0.6')
          .to({}, { duration: 0.3 })
      } else {
        panels.forEach((panel) => {
          $gsap.from(panelParts(panel), {
            scrollTrigger: { trigger: panel, start: 'top 75%' },
            opacity: 0,
            y: 36,
            stagger: 0.08,
            duration: 0.9,
            ease: 'power3.out'
          })
        })
      }
    }
  )
})

onBeforeUnmount(() => mm?.revert())
</script>
