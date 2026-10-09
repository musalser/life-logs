<template>
  <!--
    A line whose word markup no longer matches its text: the text was edited by
    hand, so the word boxes on the scan are stale. Used as a quiet icon after the
    line instead of a chip under it, with the explanation in the tooltip.
  -->
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.9"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path d="M12 4.5 3 19.5h18L12 4.5z" />
    <path d="M12 10v4.5" />
    <circle cx="12" cy="17.2" r="0.9" fill="currentColor" stroke="none" />
  </svg>
</template>
