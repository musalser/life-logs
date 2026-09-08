<template>
  <section ref="root" class="relative h-[100svh] min-h-[640px] overflow-hidden">
    <div data-l="backdrop" class="absolute inset-0 will-change-transform">
      <img
        src="/landing/temple-backdrop.jpg"
        alt=""
        fetchpriority="high"
        class="h-full w-full scale-105 object-cover opacity-80"
      />
    </div>

    <div data-l="rays" class="god-rays pointer-events-none absolute inset-0 opacity-60 mix-blend-screen"></div>

    <div class="absolute inset-0 flex items-end justify-center">
      <img
        data-l="statue"
        src="/landing/hero-statue.png"
        alt="Heroic marble statue"
        fetchpriority="high"
        class="h-[80%] w-auto max-w-none object-contain drop-shadow-[0_30px_80px_rgba(0,0,0,0.85)] will-change-transform"
      />
    </div>

    <img
      data-l="col-left"
      src="/landing/column-left.png"
      alt=""
      class="absolute -left-[10%] bottom-[-8%] h-[90%] w-auto max-w-none object-contain will-change-transform sm:left-[-4%]"
    />
    <img
      data-l="col-right"
      src="/landing/column-right.png"
      alt=""
      class="absolute -right-[14%] top-[-18%] h-[70%] w-auto max-w-none -scale-x-100 object-contain will-change-transform sm:right-[-7%]"
    />

    <!-- fog-1 used twice: back band + mirrored front band -->
    <div data-l="fog-a" class="absolute inset-x-[-15%] bottom-[-4%] h-[42%] will-change-transform">
      <img data-l="fog-a-img" src="/landing/fog-1.png" alt="" class="h-full w-full object-cover opacity-50" />
    </div>
    <div data-l="fog-b" class="absolute inset-x-[-20%] bottom-[-10%] h-[52%] will-change-transform">
      <img data-l="fog-b-img" src="/landing/fog-1.png" alt="" class="h-full w-full -scale-x-100 object-cover opacity-70" />
    </div>

    <div class="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(11,11,15,0.85)_100%)]"></div>
    <div class="pointer-events-none absolute inset-x-0 bottom-0 h-44 bg-gradient-to-b from-transparent to-obsidian"></div>

    <div data-l="content" class="relative z-10 flex h-full flex-col items-center justify-center px-6 pb-[8vh] text-center">
      <p class="landing-eyebrow">Life Logs</p>
      <h1
        class="mt-6 font-display text-4xl font-semibold leading-tight text-marble [text-shadow:0_4px_40px_rgba(0,0,0,0.9)] sm:text-6xl lg:text-7xl"
      >
        Become who you<br class="hidden sm:block" />
        were meant to be
      </h1>
      <p class="mt-6 max-w-2xl font-serif text-lg italic text-marble/75 sm:text-2xl">
        Your diary, read by an oracle. Your habits, forged by mentors. Your ascent, measured in days.
      </p>
      <NuxtLink to="/chat" class="btn-gold mt-10">Enter the Temple</NuxtLink>
    </div>

    <div data-l="cue" class="absolute inset-x-0 bottom-6 z-10 flex flex-col items-center gap-2 text-marble/50">
      <span class="font-display text-[10px] uppercase tracking-[0.4em]">Descend</span>
      <span class="h-10 w-px animate-pulse bg-gradient-to-b from-marble/60 to-transparent"></span>
    </div>
  </section>
</template>

<script setup>
const root = ref(null)
const { $gsap } = useNuxtApp()
let mm

onMounted(() => {
  const el = (name) => root.value.querySelector(`[data-l="${name}"]`)

  mm = $gsap.matchMedia()
  mm.add(
    {
      full: '(min-width: 768px) and (prefers-reduced-motion: no-preference)',
      lite: '(max-width: 767.98px) and (prefers-reduced-motion: no-preference)'
    },
    (ctx) => {
      const { full } = ctx.conditions

      $gsap
        .timeline({ defaults: { ease: 'power3.out' } })
        .from(el('statue'), { opacity: 0, y: 80, duration: 1.6 })
        .from([el('col-left'), el('col-right')], { opacity: 0, y: 60, duration: 1.4 }, 0.15)
        .from(el('content').children, { opacity: 0, y: 36, stagger: 0.12, duration: 1 }, 0.45)
        .from(el('cue'), { opacity: 0, duration: 1 }, 1.2)

      // endless fog drift, independent of scroll
      $gsap.to(el('fog-a-img'), { xPercent: 7, duration: 26, repeat: -1, yoyo: true, ease: 'sine.inOut' })
      $gsap.to(el('fog-b-img'), { xPercent: -9, duration: 34, repeat: -1, yoyo: true, ease: 'sine.inOut' })

      if (full) {
        $gsap
          .timeline({
            defaults: { ease: 'none' },
            scrollTrigger: { trigger: root.value, start: 'top top', end: 'bottom top', scrub: true }
          })
          .to(el('backdrop'), { yPercent: 14, scale: 1.06 }, 0)
          .to(el('statue'), { yPercent: 7 }, 0)
          .to([el('col-left'), el('col-right')], { yPercent: -16 }, 0)
          .to(el('fog-a'), { yPercent: -22, opacity: 0.4 }, 0)
          .to(el('fog-b'), { yPercent: -34 }, 0)
          .to(el('content'), { yPercent: -30, opacity: 0 }, 0)
          .to(el('rays'), { opacity: 0.15 }, 0)
      }
    }
  )
})

onBeforeUnmount(() => mm?.revert())
</script>

<style scoped>
.god-rays {
  background:
    linear-gradient(115deg, transparent 42%, rgba(201, 162, 39, 0.16) 47%, transparent 52%),
    linear-gradient(100deg, transparent 55%, rgba(237, 232, 223, 0.09) 60%, transparent 66%),
    linear-gradient(125deg, transparent 28%, rgba(201, 162, 39, 0.1) 33%, transparent 39%);
}
</style>
