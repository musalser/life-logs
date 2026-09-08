<template>
  <section ref="root" class="relative overflow-hidden bg-obsidian">
    <div class="relative flex min-h-[100svh] items-center overflow-hidden">
      <div data-l="stairs" class="absolute inset-0 will-change-transform">
        <img
          src="/landing/stairs-ascent.jpg"
          alt=""
          loading="lazy"
          decoding="async"
          class="h-full w-full object-cover object-bottom opacity-80"
        />
      </div>
      <div class="pointer-events-none absolute inset-0 bg-gradient-to-r from-obsidian/80 via-obsidian/30 to-transparent"></div>
      <div
        data-l="glow"
        class="pointer-events-none absolute inset-x-0 top-[-20%] h-[60%] bg-[radial-gradient(ellipse_at_top,rgba(201,162,39,0.35),transparent_65%)] opacity-30"
      ></div>
      <div class="pointer-events-none absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-obsidian to-transparent"></div>
      <div class="pointer-events-none absolute inset-x-0 bottom-0 h-36 bg-gradient-to-t from-obsidian to-transparent"></div>

      <div class="relative z-10 mx-auto grid w-full max-w-6xl grid-cols-1 items-center gap-14 px-6 py-24 md:grid-cols-2">
        <div>
          <p data-l="h" class="landing-eyebrow">The Ascent</p>
          <h2 data-l="h" class="mt-4 font-display text-3xl font-semibold text-marble sm:text-5xl">
            Every day is a step
          </h2>
          <p data-l="h" class="mt-5 max-w-md font-serif text-lg italic text-marble/70">
            Name a goal and the temple keeps the count — each entry a stair, each streak a landing on the climb.
          </p>

          <ol class="mt-10 space-y-5">
            <li v-for="step in steps" :key="step.numeral" class="ascent-step flex items-baseline gap-4">
              <span class="font-display text-sm text-gold/70">{{ step.numeral }}</span>
              <span class="font-serif text-xl text-marble/85">{{ step.text }}</span>
            </li>
          </ol>
        </div>

        <div class="relative mx-auto aspect-square w-full max-w-sm">
          <img
            data-l="wreath"
            src="/landing/laurel-wreath.png"
            alt="Golden laurel wreath"
            loading="lazy"
            decoding="async"
            class="absolute inset-0 h-full w-full object-contain drop-shadow-[0_0_40px_rgba(201,162,39,0.2)]"
          />
          <svg class="absolute inset-[13%] -rotate-90" viewBox="0 0 100 100" aria-hidden="true">
            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(237,232,223,0.08)" stroke-width="1.5" />
            <circle
              data-l="ring"
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#C9A227"
              stroke-width="1.5"
              stroke-linecap="round"
              pathLength="100"
              style="stroke-dasharray: 100; stroke-dashoffset: 0"
            />
          </svg>
          <div class="absolute inset-0 flex flex-col items-center justify-center text-center">
            <p class="font-display text-5xl text-marble"><span data-l="count">100</span>%</p>
            <p class="mt-2 font-display text-[10px] uppercase tracking-[0.35em] text-gold/70">to the summit</p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const steps = [
  { numeral: 'I', text: 'Name the goal' },
  { numeral: 'II', text: 'Log the days' },
  { numeral: 'III', text: 'Hold the streak' },
  { numeral: 'IV', text: 'Stand on the summit' }
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
      const ringTweens = (tl, pos) =>
        tl
          .fromTo(el('ring'), { strokeDashoffset: 100 }, { strokeDashoffset: 0, duration: 1, ease: 'none' }, pos)
          .fromTo(
            el('count'),
            { textContent: 0 },
            { textContent: 100, snap: { textContent: 1 }, duration: 1, ease: 'none' },
            pos
          )

      if (full) {
        const tl = $gsap.timeline({
          scrollTrigger: { trigger: root.value, start: 'top top', end: '+=160%', pin: true, scrub: 1 }
        })
        tl.from(q('[data-l="h"]'), { opacity: 0, y: 30, stagger: 0.1 })
          .from(q('.ascent-step'), { opacity: 0, x: -30, stagger: 0.18 }, 0.15)
          .from(el('wreath'), { opacity: 0, scale: 0.85 }, 0.2)
          .to(el('stairs'), { yPercent: -8, scale: 1.06, ease: 'none', duration: 1.6 }, 0)
          .to(el('glow'), { opacity: 1, duration: 1.6 }, 0)
        ringTweens(tl, 0.35)
      } else {
        $gsap.from([...q('[data-l="h"]'), ...q('.ascent-step')], {
          scrollTrigger: { trigger: root.value, start: 'top 70%' },
          opacity: 0,
          y: 30,
          stagger: 0.09,
          duration: 0.9,
          ease: 'power3.out'
        })
        const tl = $gsap.timeline({ scrollTrigger: { trigger: el('wreath'), start: 'top 75%' } })
        tl.from(el('wreath'), { opacity: 0, scale: 0.85, duration: 0.8, ease: 'power3.out' })
        ringTweens(tl, 0.2)
      }
    }
  )
})

onBeforeUnmount(() => mm?.revert())
</script>
