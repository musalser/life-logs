<template>
  <section ref="root" class="relative overflow-hidden bg-obsidian">
    <div
      class="mx-auto grid min-h-[100svh] max-w-6xl grid-cols-1 items-center gap-12 px-6 py-24 md:grid-cols-2 md:gap-16 md:py-0"
    >
      <div>
        <p data-l="h" class="landing-eyebrow">The Oracle</p>
        <h2 data-l="h" class="mt-4 font-display text-3xl font-semibold text-marble sm:text-5xl">
          The Oracle reads your scrolls
        </h2>
        <p data-l="h" class="mt-5 max-w-md font-serif text-lg italic text-marble/70">
          Write freely, in your own words. The Oracle parses every entry and turns loose ink into living knowledge.
        </p>

        <ul class="mt-10 space-y-4 font-serif text-lg text-marble/85">
          <li class="oracle-line">“Slept badly again — six hours, restless.”</li>
          <li class="oracle-line">“Morning run, five kilometres along the river.”</li>
          <li class="oracle-line">“Felt anxious before the stand-up. Breathed through it.”</li>
          <li class="oracle-line">“Finished chapter twelve of the book. Finally.”</li>
          <li class="oracle-line">“Coffee with Anna — we talked about the move.”</li>
        </ul>
      </div>

      <div class="relative mx-auto flex aspect-square w-full max-w-md items-center justify-center md:max-w-lg">
        <div
          data-l="orb"
          class="absolute left-1/2 top-[38%] h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gold/40 blur-3xl"
        ></div>
        <img
          data-l="hand"
          src="/landing/oracle-hand.png"
          alt="Marble hand holding a glowing orb"
          loading="lazy"
          decoding="async"
          class="relative h-full w-full object-contain drop-shadow-[0_10px_50px_rgba(201,162,39,0.25)]"
        />
        <span class="oracle-fact left-[2%] top-[10%]">Sleep · 6 h, restless</span>
        <span class="oracle-fact right-[-2%] top-[22%]">Run · 5 km</span>
        <span class="oracle-fact left-[-4%] top-[56%]">Mood · anxious → calm</span>
        <span class="oracle-fact right-[2%] top-[66%]">Reading · ch. XII done</span>
        <span class="oracle-fact left-[28%] top-[-3%]">Social · coffee with Anna</span>
      </div>
    </div>
  </section>
</template>

<script setup>
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
      const lines = q('.oracle-line')
      const facts = q('.oracle-fact')

      if (full) {
        $gsap
          .timeline({
            scrollTrigger: { trigger: root.value, start: 'top top', end: '+=170%', pin: true, scrub: 1 }
          })
          .from(q('[data-l="h"]'), { opacity: 0, y: 32, stagger: 0.08 })
          .from(lines, { opacity: 0, y: 24, stagger: 0.1 }, 0.1)
          .from(el('hand'), { opacity: 0, y: 60 }, 0.15)
          .fromTo(el('orb'), { opacity: 0.15, scale: 0.7 }, { opacity: 0.9, scale: 1.25 }, 0.3)
          .to(lines, { opacity: 0.18, x: 28, filter: 'blur(4px)', stagger: 0.08 }, '+=0.15')
          .from(facts, { opacity: 0, scale: 0.5, ease: 'back.out(1.8)', stagger: 0.09 }, '<+0.1')
      } else {
        $gsap.from([...q('[data-l="h"]'), ...lines], {
          scrollTrigger: { trigger: root.value, start: 'top 70%' },
          opacity: 0,
          y: 30,
          stagger: 0.08,
          duration: 0.9,
          ease: 'power3.out'
        })
        $gsap.from([el('hand'), ...facts], {
          scrollTrigger: { trigger: el('hand'), start: 'top 80%' },
          opacity: 0,
          y: 40,
          stagger: 0.08,
          duration: 0.9,
          ease: 'power3.out'
        })
      }
    }
  )
})

onBeforeUnmount(() => mm?.revert())
</script>

<style scoped>
.oracle-line {
  border-left: 1px solid rgba(201, 162, 39, 0.3);
  padding-left: 1rem;
}

.oracle-fact {
  position: absolute;
  border: 1px solid rgba(201, 162, 39, 0.4);
  background: rgba(11, 11, 15, 0.8);
  backdrop-filter: blur(4px);
  padding: 0.3rem 0.8rem;
  font-family: Cinzel, serif;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #c9a227;
  white-space: nowrap;
}
</style>
