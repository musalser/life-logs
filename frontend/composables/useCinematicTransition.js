import { ref } from 'vue'

/*
 * CINEMATIC TAB CHANGES — the data half.
 *
 * The four sections are not four views of one screen: the diary is something you
 * write, the manuscripts are a workshop, knowledge is the ledger distilled out of
 * both, and the chat is the voice that reads them back. A router push between
 * them swaps one wall of text for another and reads as a repaint, not as a move.
 *
 * So the change is staged as a cut: the page loses focus behind a blurred, dark
 * veil with one object standing in the middle of it, the route swaps while that
 * veil hides it, and the new page comes back into focus. This file owns
 *
 *   - TAB_SCENES: which object belongs to which section;
 *   - preloading, so the first cut is not a middle waiting on a picture;
 *   - the handshake with the router: `stageTransition()` covers, `finishTransition()`
 *     uncovers, and the DOM/GSAP half lives in components/CinematicCurtain.vue.
 *
 * Everything here is client-only by construction: the server never navigates and
 * the curtain registers its driver after mount. If the driver is missing (SSR,
 * a page without the curtain, reduced motion), both entry points are no-ops and
 * the router pushes as it always did.
 */

/*
 * The picture is chosen for what the section *is*, not for decoration:
 *   chat        — the sage who answers
 *   diary       — the figure who keeps the record
 *   knowledge   — the oracle's open hand: what the diary gives up when asked
 *   manuscripts — Vulcan at the anvil: pages are forged, not found
 *
 * `fit: 'figure'` is a cut-out standing on the veil (transparent PNG, centred,
 * object-fit: contain); `fit: 'cover'` is a photograph that fills the screen.
 *
 * A scene may instead carry real motion, which the curtain picks up
 * automatically — see docs/cinematic-tab-transitions.md:
 *   frames: { pattern: '/landing/chat/frame-%03d.webp', count: 96 }
 *   video:  '/landing/chat/plate.webm'
 */
export const TAB_SCENES = {
  chat: {
    path: '/chat',
    plate: '/landing/bust-sage.png',
    fit: 'figure'
  },
  diary: {
    path: '/diary',
    plate: '/landing/hero-statue.png',
    fit: 'figure'
  },
  knowledge: {
    path: '/knowledge',
    plate: '/landing/oracle-hand.png',
    fit: 'figure'
  },
  manuscripts: {
    path: '/manuscripts',
    plate: '/landing/forge-statue.png',
    fit: 'figure'
  }
}

const byPath = new Map(Object.values(TAB_SCENES).map((scene) => [scene.path, scene]))

/** `/diary/?page=3` and `/diary` are the same section. */
export const normalizePath = (path) =>
  String(path || '/').split(/[?#]/)[0].replace(/\/+$/, '') || '/'

export const sceneForPath = (path) => byPath.get(normalizePath(path)) ?? null

export const sceneList = () => Object.values(TAB_SCENES)

export const prefersReducedMotion = () =>
  import.meta.client && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/*
 * The objects are photographs, 300–500 KB each: warming them all at boot would
 * spend the diary's bandwidth on pictures the reader may never ask for. They are
 * fetched on hover/focus of the tab, and the rest are picked up one per idle slot
 * once the app has settled, so a cold click still lands on a loaded picture.
 */
const warmed = new Set()

export function warmScene(scene) {
  if (!import.meta.client || !cinematic.enabled.value || !scene?.plate || warmed.has(scene.path)) return
  warmed.add(scene.path)
  const image = new Image()
  image.decoding = 'async'
  image.src = scene.plate
}

export function warmAllScenesLater(delay = 2500) {
  if (!import.meta.client || !cinematic.enabled.value) return
  let pending = sceneList().filter((scene) => !warmed.has(scene.path))
  const tick = () => {
    const scene = pending.shift()
    if (!scene) return
    warmScene(scene)
    idle(tick)
  }
  setTimeout(() => idle(tick), delay)
}

const idle = (run) =>
  'requestIdleCallback' in window ? window.requestIdleCallback(run, { timeout: 1000 }) : setTimeout(run, 250)

/* --------------------------------------------------------------------------
 * The switch, and the state the UI can read (aria-busy, a test hook).
 *
 * `enabled` mirrors `cinematicTransitions` in app.config.ts. The plugin sets it
 * once at boot; everything else in this file asks it before doing any work, so a
 * disabled feature costs nothing at runtime — no driver, no guard, no preloading.
 * ------------------------------------------------------------------------ */

export const cinematic = {
  enabled: ref(true),
  active: ref(false),
  /* idle → cover → covered → reveal → idle */
  phase: ref('idle'),
  sceneKey: ref(null)
}

export const setCinematicEnabled = (value) => {
  cinematic.enabled.value = value !== false
}

export const cinematicEnabled = () => cinematic.enabled.value

export const useCinematicTransition = () => ({
  ...cinematic,
  scenes: TAB_SCENES,
  sceneForPath,
  sceneList,
  warmScene,
  prefersReducedMotion
})

/* --------------------------------------------------------------------------
 * Driver registry: the curtain component hands its timeline player over here.
 * ------------------------------------------------------------------------ */

let curtainDriver = null

export function registerCurtainDriver(driver) {
  curtainDriver = driver
  return () => {
    if (curtainDriver === driver) curtainDriver = null
  }
}

/* --------------------------------------------------------------------------
 * The handshake.
 *
 * The router (plugins/cinematic.client.ts) calls `stageTransition()` from a
 * beforeEach guard and waits for it, so the route only changes once the veil
 * covers the screen; `finishTransition()` runs from Nuxt's `page:finish`, which
 * fires after the new page has actually resolved and rendered.
 * ------------------------------------------------------------------------ */

let covering = null

export function stageTransition(path) {
  if (!import.meta.client || !cinematic.enabled.value || !curtainDriver) return Promise.resolve()
  if (prefersReducedMotion()) return Promise.resolve()

  const scene = sceneForPath(path)
  if (!scene) return Promise.resolve()

  /* A second click during a cut rides the blur that is already coming up. */
  if (covering) return covering

  cinematic.active.value = true
  cinematic.phase.value = 'cover'
  cinematic.sceneKey.value = scene.path

  covering = Promise.resolve(curtainDriver.cover(scene))
    .catch(() => {})
    .then(() => {
      cinematic.phase.value = 'covered'
    })

  return covering
}

export async function finishTransition() {
  covering = null
  if (!import.meta.client || !cinematic.active.value) return

  cinematic.phase.value = 'reveal'
  try {
    if (curtainDriver && !prefersReducedMotion()) await curtainDriver.reveal()
  } catch {
    /* a broken timeline must not leave the blur over the page */
  } finally {
    cinematic.active.value = false
    cinematic.phase.value = 'idle'
    cinematic.sceneKey.value = null
  }
}
