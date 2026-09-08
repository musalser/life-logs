<template>
  <section ref="root" class="relative overflow-hidden bg-obsidian">
    <div class="relative flex min-h-[100svh] items-center justify-center overflow-hidden">
      <div data-l="doors" class="absolute inset-0 will-change-transform">
        <img
          src="/landing/temple-doors.jpg"
          alt=""
          loading="lazy"
          decoding="async"
          class="h-full w-full object-cover"
        />
      </div>
      <div data-l="dim" class="absolute inset-0 bg-obsidian/70"></div>

      <div
        data-l="seam"
        class="pointer-events-none absolute left-1/2 top-0 h-full w-[3px] -translate-x-1/2 bg-gradient-to-b from-gold/0 via-gold to-gold/0 opacity-40 blur-[2px]"
      ></div>
      <div
        data-l="seam-glow"
        class="pointer-events-none absolute left-1/2 top-0 h-full w-28 -translate-x-1/2 bg-gold/25 opacity-0 blur-3xl"
      ></div>

      <div class="pointer-events-none absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-obsidian to-transparent"></div>

      <div
        class="pointer-events-none absolute left-1/2 top-1/2 h-[80vh] w-[110vh] -translate-x-1/2 -translate-y-1/2 bg-[radial-gradient(ellipse_at_center,rgba(11,11,15,0.75),transparent_70%)]"
      ></div>

      <div class="relative z-10 flex flex-col items-center px-6 text-center">
        <p data-l="c" class="landing-eyebrow">The Threshold</p>
        <h2
          data-l="c"
          class="mt-5 font-display text-4xl font-semibold text-marble [text-shadow:0_4px_40px_rgba(0,0,0,0.9)] sm:text-6xl"
        >
          The doors are open
        </h2>
        <p data-l="c" class="mt-5 max-w-xl font-serif text-lg italic text-marble/75 sm:text-xl">
          Begin your chronicle. Write one honest line tonight — the temple will remember it forever.
        </p>
        <NuxtLink to="/chat" data-l="c" class="btn-gold mt-10">Enter the Temple</NuxtLink>
        <p data-l="c" class="mt-4 font-display text-[10px] uppercase tracking-[0.35em] text-marble/40">
          No rites required
        </p>
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

      if (full) {
        $gsap
          .timeline({
            scrollTrigger: { trigger: root.value, start: 'top top', end: '+=130%', pin: true, scrub: 1 }
          })
          .to(el('doors'), { scale: 1.14, ease: 'none', duration: 1 }, 0)
          .to(el('dim'), { opacity: 0.6, ease: 'none', duration: 1 }, 0)
          .to(el('seam'), { opacity: 1, scaleX: 20, ease: 'power1.in', duration: 1 }, 0)
          .to(el('seam-glow'), { opacity: 1, ease: 'power1.in', duration: 1 }, 0)
          .from(q('[data-l="c"]'), { opacity: 0, y: 40, stagger: 0.12, duration: 0.4 }, 0.25)
      } else {
        $gsap.from(q('[data-l="c"]'), {
          scrollTrigger: { trigger: root.value, start: 'top 65%' },
          opacity: 0,
          y: 36,
          stagger: 0.1,
          duration: 0.9,
          ease: 'power3.out'
        })
      }
    }
  )
})

onBeforeUnmount(() => mm?.revert())
</script>
