import { finishTransition, setCinematicEnabled, stageTransition } from '~/composables/useCinematicTransition'

/*
 * Staging the tab change in the router, not in the link.
 *
 * Intercepting the click would have covered only the links we remembered to
 * rewrite; a guard covers every way into a section — the header tabs, the "Enter
 * the Temple" button on the landing page, a bookmark, back/forward — and it
 * covers it at the only moment when it is safe to change the page: after the
 * plate has finished rising (the guard awaits the cover, so the route really does
 * change under it).
 *
 * The other half of the handshake is `page:finish`, which Nuxt fires once the new
 * page has resolved and rendered. That — not the navigation promise — is the
 * truth about "the new screen is on", so the veil lifts there. The watchdog is
 * the failure path: if a page never resolves, the reader is not left staring at a
 * blurred screen.
 *
 * The whole thing is behind `cinematicTransitions` in app.config.ts. When it is
 * off this plugin registers nothing at all — no guard, no watchdog, no hook —
 * so tab clicks are plain router pushes and the feature costs zero.
 */
export default defineNuxtPlugin((nuxtApp) => {
  const enabled = useAppConfig().cinematicTransitions !== false
  setCinematicEnabled(enabled)
  if (!enabled) return

  const router = useRouter()

  /* HMR re-runs plugins; a second guard would stage every cut twice. */
  const guarded = router as unknown as { __cinematicGuarded?: boolean }
  if (guarded.__cinematicGuarded) return
  guarded.__cinematicGuarded = true

  let watchdog: ReturnType<typeof setTimeout> | null = null

  const disarm = () => {
    if (watchdog) clearTimeout(watchdog)
    watchdog = null
  }

  const arm = () => {
    disarm()
    watchdog = setTimeout(() => {
      watchdog = null
      finishTransition()
    }, 5000)
  }

  router.beforeEach(async (to, from) => {
    /* Query-only moves (another diary page, another manuscript author) are edits
       inside one section, not a journey between two: they must not cut. */
    if (to.path === from.path) return
    await stageTransition(to.path)
  })

  router.afterEach((to, from) => {
    if (to.path !== from.path) arm()
  })

  nuxtApp.hook('page:finish', () => {
    disarm()
    finishTransition()
  })

  router.onError(() => {
    disarm()
    finishTransition()
  })
})
