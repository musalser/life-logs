<template>
  <!--
    The theme attribute lives here, on the element Vue owns. Rendering it into
    <html> from useHead made hydration reconcile the server's `dark` back over the
    client's stored `light`, so the switch looked dead.
  -->
  <div :data-theme="theme">
    <NuxtLayout>
      <NuxtPage />
    </NuxtLayout>
  </div>
</template>

<script setup>
const { theme, initTheme, syncDocumentTheme } = useTheme()

/*
 * On mount, not in setup: the server cannot know the stored preference, so the
 * first client render must match the server's markup or hydration would patch the
 * attribute back (that is what made the switch look broken). By the time this
 * runs, the head script has already painted the right palette onto <html>, so
 * resolving the state here does not flash.
 */
onMounted(() => {
  const applied = initTheme()
  syncDocumentTheme(applied)
  watch(theme, (value) => syncDocumentTheme(value))
})
</script>
