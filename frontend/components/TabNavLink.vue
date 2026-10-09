<template>
  <!--
    A tab is a door, so it carries a small mark of what is behind it: a bubble for
    the chat, an open book for the diary, a star for the ledger of knowledge, a
    hammer for the manuscripts (pages are forged there, on the anvil the plate for
    that section shows).

    Navigation itself is deliberately *not* handled here — plugins/cinematic.client.ts
    stages every route change in a router guard, so this stays a plain link: it
    works without JavaScript, opens in a new tab on middle click, and cannot drift
    out of step with the back button.
  -->
  <NuxtLink
    :to="to"
    class="nav-link"
    active-class="nav-link--active"
    @mouseenter="preload"
    @focus="preload"
  >
    <span class="nav-link__glyph" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="currentColor">
        <template v-if="glyph === 'chat'">
          <path d="M4.5 4h15a1.5 1.5 0 011.5 1.5v10A1.5 1.5 0 0119.5 17H9.6L4 21.2V5.5A1.5 1.5 0 015.5 4z" />
        </template>
        <template v-else-if="glyph === 'diary'">
          <path d="M12 6.9C9.7 5.2 6.1 4.9 3.8 5.7V18.3c2.3-.8 5.9-.5 8.2 1.2 2.3-1.7 5.9-2 8.2-1.2V5.7c-2.3-.8-5.9-.5-8.2 1.2z" />
          <path d="M11.4 8.1h1.2v11.4h-1.2z" fill="#07070b" />
        </template>
        <template v-else-if="glyph === 'knowledge'">
          <path d="M12 2.8l2.6 5.8 6.3.8-4.6 4.3 1.2 6.2-5.5-3.1-5.5 3.1 1.2-6.2L3.1 9.4l6.3-.8z" />
        </template>
        <template v-else>
          <path d="M14.1 2.5l7.4 7.4-2.9 2.9-7.4-7.4z" />
          <path d="M11.5 8.7l3.8 3.8-8.2 8.2-3.8-3.8z" />
        </template>
      </svg>
    </span>
    <span class="nav-link__label">{{ label }}</span>
  </NuxtLink>
</template>

<script setup>
import { sceneForPath, warmScene } from '~/composables/useCinematicTransition'

const props = defineProps({
  to: { type: String, required: true },
  label: { type: String, required: true },
  glyph: { type: String, default: 'chat' }
})

/* Hovering a door is the cheapest moment to fetch the room behind it: by the
   time the plate rises, its photograph is already in the cache. */
const preload = () => warmScene(sceneForPath(props.to))
</script>

<style scoped>
.nav-link__glyph {
  display: inline-flex;
  width: 1rem;
  height: 1rem;
  flex: none;
  color: rgb(var(--c-accent));
  opacity: 0.75;
  transition: transform 0.25s ease, opacity 0.25s ease;
}

.nav-link__glyph svg {
  width: 100%;
  height: 100%;
}

.nav-link:hover .nav-link__glyph,
.nav-link--active .nav-link__glyph {
  opacity: 1;
  transform: translateY(-1px) rotate(-6deg);
}

.nav-link__label {
  white-space: nowrap;
}
</style>
