<template>
  <!--
    A padlock for the page state: closed means the page is confirmed ground
    truth (its transcription is frozen and it feeds the training corpus), open
    means it is still being edited. Used as the button icon in the page header
    and as the state indicator in the page list.
  -->
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <rect x="4" y="10.5" width="16" height="10.5" rx="2" />
    <path v-if="closed" d="M8 10.5V7a4 4 0 0 1 8 0v3.5" />
    <path v-else d="M8 10.5V7a4 4 0 0 1 7.6-1.7" />
    <circle v-if="closed" cx="12" cy="15.5" r="1.4" fill="currentColor" stroke="none" />
  </svg>
</template>

<script setup lang="ts">
defineProps<{ closed?: boolean }>()
</script>
