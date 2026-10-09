<template>
  <Teleport to="body">
    <!--
      The cut is a focus pull, not a curtain: the page behind goes soft and dark,
      one object stands in the middle of it, and then the blur lifts off the new
      page. No text, nothing travels across the screen — the only things that move
      are the depth of field and the object's slow push-in.

      It is decorative — the route change it hides is announced by the page itself
      — so it stays out of the a11y tree, but it does own the pointer while it is
      up, which is what stops a second click from queueing a second cut.
    -->
    <div
      v-if="visible"
      ref="root"
      class="cine"
      :class="`cine--${plateFit}`"
      :data-phase="phase"
      aria-hidden="true"
    >
      <!--
        The blur lives on its own layer, and this layer is the only one with
        `backdrop-filter`: an ancestor with `contain: paint`, `filter`, `opacity`
        or `isolation` would become the backdrop root and the veil would end up
        blurring an empty box instead of the page. So `.cine` above carries none
        of those.
      -->
      <div ref="veil" class="cine__veil"></div>

      <div class="cine__stage">
        <div ref="glow" class="cine__glow"></div>
        <img
          v-if="scene?.plate"
          ref="art"
          class="cine__art"
          :src="scene.plate"
          alt=""
          decoding="async"
        />
        <video
          v-if="plateKind === 'video'"
          ref="videoEl"
          class="cine__art"
          :src="scene.video"
          muted
          playsinline
          loop
          preload="auto"
        ></video>
        <canvas v-else-if="plateKind === 'frames'" ref="canvasEl" class="cine__art cine__art--seq"></canvas>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import {
  finishTransition,
  prefersReducedMotion,
  registerCurtainDriver,
  useCinematicTransition,
  warmAllScenesLater
} from '~/composables/useCinematicTransition'

/*
 * THE CINEMATIC CUT — the DOM and GSAP half of the tab change.
 *
 * Two halves of one idea: the outgoing page loses focus (a `backdrop-filter`
 * blur under a near-black veil) and the incoming one gains it. The object in the
 * middle belongs to the destination, appears with the blur and leaves with it, so
 * the eye follows one thing across the swap. Timings are collected in CUT.
 *
 * The composable owns *when* this happens (router guard covers, `page:finish`
 * reveals); this component only knows *how*. It registers itself as the driver,
 * so if it is ever unmounted the transition degrades to a plain push.
 */

const CUT = {
  blurIn: 0.5, // the page going soft and dark
  settle: 0.55, // the veil is opaque; the route may change now
  out: 0.6, // the new page coming back into focus
  pushIn: 0.85 // the object's camera move, which runs through both halves
}

const { $gsap } = useNuxtApp()
const { phase, enabled } = useCinematicTransition()

const root = ref(null)
const veil = ref(null)
const glow = ref(null)
const art = ref(null)
const videoEl = ref(null)
const canvasEl = ref(null)

const visible = ref(false)
const scene = shallowRef(null)
const frameImages = shallowRef([])

const plateKind = computed(() => {
  if (scene.value?.video) return 'video'
  if (scene.value?.frames) return 'frames'
  return 'image'
})
const plateFit = computed(() => scene.value?.fit === 'cover' ? 'cover' : 'figure')

let coverTimeline = null
let liftTimeline = null
let releaseDriver = null

/* ---------------------------------------------------------------------------
 * Plates with real motion (optional): an image sequence scrubbed by the same
 * timeline that fades the veil, or a video simply played. The still frame is
 * always rendered underneath, so a sequence that is still downloading shows the
 * object rather than an empty middle.
 * ------------------------------------------------------------------------- */

const frameSets = new Map()

function frameSource(pattern, index) {
  /* frame-001.webp — ffmpeg numbers from 1, so the reader's first frame is 1. */
  return pattern.replace(/%0?(\d*)d/, (_, width) => String(index + 1).padStart(Number(width) || 1, '0'))
}

function loadFrames(target) {
  if (frameSets.has(target.path)) return frameSets.get(target.path)
  const { pattern, count } = target.frames
  const pending = Promise.all(
    Array.from({ length: count }, (_, index) => new Promise((resolve) => {
      const image = new Image()
      image.decoding = 'async'
      image.onload = () => resolve(image)
      image.onerror = () => resolve(null)
      image.src = frameSource(pattern, index)
    }))
  ).then((images) => images.filter(Boolean))
  frameSets.set(target.path, pending)
  return pending
}

/*
 * A canvas has no intrinsic size to be laid out by `max-height`/`max-width` the
 * way an <img> does, so the "contain" box is computed here from the frame's own
 * aspect and the viewport, and the frames are drawn into it at device pixels.
 */
function frameBox(image) {
  const maxHeight = window.innerHeight * 0.74
  const maxWidth = window.innerWidth * 0.76
  const aspect = image.width / image.height
  let height = maxHeight
  let width = height * aspect
  if (width > maxWidth) {
    width = maxWidth
    height = width / aspect
  }
  return { width, height }
}

function paintFrame(progress) {
  const canvas = canvasEl.value
  const frames = frameImages.value
  if (!canvas || !frames.length) return

  const index = Math.min(frames.length - 1, Math.max(0, Math.round(progress * (frames.length - 1))))
  const image = frames[index]
  const box = frameBox(image)
  const dpr = Math.min(2, window.devicePixelRatio || 1)

  canvas.style.width = `${box.width}px`
  canvas.style.height = `${box.height}px`
  if (canvas.width !== Math.round(box.width * dpr)) canvas.width = Math.round(box.width * dpr)
  if (canvas.height !== Math.round(box.height * dpr)) canvas.height = Math.round(box.height * dpr)

  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, box.width, box.height)
  ctx.drawImage(image, 0, 0, box.width, box.height)
}

/* ---------------------------------------------------------------------------
 * Driver: cover → (router changes the page) → reveal.
 * ------------------------------------------------------------------------- */

async function cover(next) {
  if (prefersReducedMotion()) return
  scene.value = next
  visible.value = true
  frameImages.value = []

  await nextTick() // the layers have to exist before a timeline can address them

  const gsap = $gsap
  if (!gsap || !root.value || !veil.value) {
    /* No timeline driver (ancient browser, script error): drop the veil rather
       than leave an invisible full-screen layer holding the pointer. */
    visible.value = false
    scene.value = null
    return
  }

  if (next.frames) {
    /* The cut must not wait for 96 frames; whichever arrived in ~0.9 s is used. */
    const settled = await Promise.race([
      loadFrames(next),
      new Promise((resolve) => setTimeout(() => resolve(null), 900))
    ])
    if (settled?.length) {
      frameImages.value = settled
      paintFrame(0)
    }
  }
  if (videoEl.value) videoEl.value.play?.().catch(() => {})

  await new Promise((resolve) => {
    const scrub = { frame: 0 }
    let resolved = false
    const settle = () => {
      if (resolved) return
      resolved = true
      resolve()
    }
    coverTimeline = gsap.timeline({ onComplete: settle, defaults: { ease: 'power2.out' } })
    const tl = coverTimeline

    /* Both layers are born at `opacity: 0` in CSS, so the frame between mounting
       them and this timeline cannot flash a covered screen. */
    tl.fromTo(veil.value, { opacity: 0 }, { opacity: 1, duration: CUT.blurIn }, 0)
    tl.fromTo(glow.value, { opacity: 0, scale: 0.88 }, { opacity: 1, scale: 1, duration: CUT.pushIn }, 0.05)
    if (art.value) {
      tl.fromTo(art.value, { opacity: 0, scale: 1.09 }, { opacity: 1, scale: 1, duration: CUT.pushIn }, 0.08)
    }
    if (frameImages.value.length) {
      tl.to(scrub, { frame: 0.5, duration: CUT.settle + 0.2, ease: 'none', onUpdate: () => paintFrame(scrub.frame) }, 0)
    }

    /* The route may change as soon as the veil is opaque — not when the object has
       finished its push-in, which is slower and does not need to hold the cut. */
    tl.call(settle, [], CUT.settle)
  })

  coverTimeline = null
}

async function reveal() {
  if (prefersReducedMotion() || !visible.value) return
  const gsap = $gsap
  if (!gsap || !veil.value || !root.value) return

  await new Promise((resolve) => {
    liftTimeline = gsap.timeline({ onComplete: resolve })
    const tl = liftTimeline
    const scrub = { frame: frameImages.value.length ? 0.5 : 0 }

    tl.to(veil.value, { opacity: 0, duration: CUT.out, ease: 'power2.inOut' }, 0)
    tl.to(glow.value, { opacity: 0, duration: CUT.out * 0.8, ease: 'power1.in' }, 0)
    if (art.value) {
      tl.to(art.value, { opacity: 0, scale: 1.07, duration: CUT.out, ease: 'power1.inOut' }, 0)
    }
    if (frameImages.value.length) {
      tl.to(scrub, { frame: 1, duration: CUT.out, ease: 'none', onUpdate: () => paintFrame(scrub.frame) }, 0)
    }
  })

  videoEl.value?.pause?.()
  liftTimeline = null
  visible.value = false
  scene.value = null
  frameImages.value = []
}

/* Skipping the cut fast-forwards whichever half is playing; the page change
   itself is never skipped, so Escape cannot strand the reader on a blurred,
   objectless screen. */
function skip() {
  if (coverTimeline) coverTimeline.progress(1)
  if (liftTimeline) liftTimeline.progress(1)
}

const onKeydown = (event) => {
  if (event.key === 'Escape') skip()
}

onMounted(() => {
  /* Switched off in app.config.ts: no driver, no key listener, and — the part
     that actually matters on a slow connection — no preloading of 1.5 MB of
     objects nobody is going to see. */
  if (!enabled.value) return

  releaseDriver = registerCurtainDriver({ cover, reveal })
  window.addEventListener('keydown', onKeydown)
  /* The reader will pull one of these tabs in a moment; fetching them during the
     first idle slots means the first cut is not a middle waiting on a picture. */
  warmAllScenesLater()
})

onBeforeUnmount(() => {
  releaseDriver?.()
  window.removeEventListener('keydown', onKeydown)
  coverTimeline?.kill()
  liftTimeline?.kill()
  coverTimeline = null
  liftTimeline = null
  /* If the veil dies mid-cut (HMR, a page-level error), the app must not stay
     covered: clearing the state lets the router move on uncovered. */
  finishTransition()
})
</script>

<style scoped>
/*
 * No `contain`, no `filter`, no `opacity` here: any of them would make this
 * element the backdrop root and the veil would blur an empty box instead of the
 * page. `overflow: hidden` is enough to keep the object's push-in inside.
 */
.cine {
  position: fixed;
  inset: 0;
  z-index: 90;
  overflow: hidden;
  pointer-events: auto;
  cursor: progress;
}

.cine__veil {
  position: absolute;
  inset: 0;
  opacity: 0;
  /* Light enough that the blurred page can still be read as a page: the cut is a
     loss of focus, not a blackout. */
  background: radial-gradient(ellipse 78% 68% at 50% 50%, rgba(2, 2, 2, 0.7), rgba(2, 2, 2, 0.9));
  /*
   * Measured on this machine (headless, software rasterisation): the whole cut
   * runs at 62 fps without this line and 24 fps with it; `saturate()` next to the
   * blur costs another 4 fps for no visible gain, so the blur stands alone. On a
   * GPU-composited browser the blur is one cached layer, not a per-frame
   * re-rasterisation — this is the pessimistic end of the range. If a machine
   * does struggle, deleting this line leaves a smooth dark fade.
   */
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);
  will-change: opacity;
}

.cine__stage {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* A quiet halo so a cut-out does not float in a void. */
.cine__glow {
  position: absolute;
  width: min(78vh, 78vw);
  height: min(78vh, 78vw);
  border-radius: 50%;
  opacity: 0;
  background: radial-gradient(circle, rgba(201, 162, 39, 0.18), rgba(201, 162, 39, 0.04) 45%, transparent 66%);
}

.cine__art {
  position: relative;
  width: auto;
  height: auto;
  max-height: 74vh;
  max-width: 76vw;
  opacity: 0;
  will-change: transform, opacity;
}

.cine__art--seq {
  max-width: none;
  max-height: none;
}

/* A full-bleed scene (a photograph with no cut-out) still fades with the blur. */
.cine--cover .cine__art {
  width: 100vw;
  height: 100vh;
  max-width: none;
  max-height: none;
  object-fit: cover;
}

@media (prefers-reduced-motion: reduce) {
  .cine {
    display: none;
  }
}
</style>
