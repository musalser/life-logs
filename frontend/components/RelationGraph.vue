<script setup>
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { drag } from 'd3-drag'
import { select } from 'd3-selection'
import { zoom } from 'd3-zoom'
import gsap from 'gsap'
import { buildTree, isKinLink, simulateGraph } from '~/utils/relation-layout'

/*
 * Карта связей: обычный режим — пружины (как в Obsidian), режим «Древо» — слои по
 * поколениям. Переход между ними анимируется одним скаляром: позиции считаются
 * интерполяцией двух раскладок, а не двумя разными разметками.
 *
 * Узлы без родства (ритуальные, неоднозначные, одинокие) в древо не попадают и
 * перечислены отдельным списком: ставить их на нулевой слой по умолчанию нельзя.
 */

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  edges: { type: Array, default: () => [] },
  conflicts: { type: Array, default: () => [] },
  unplaced: { type: Array, default: () => [] },
  selectedId: { type: Number, default: null },
  //: когда граф пуст, показываем подсказку, а не пустую сцену
  empty: { type: Boolean, default: false },
})

const emit = defineEmits(['select', 'refresh'])

const WIDTH = 1040
const HEIGHT = 660
const NODE_RADIUS = 26
const NODE_WIDTH = 148
const NODE_HEIGHT = 46

const mode = ref('graph')
const svgRef = ref(null)
const viewportRef = ref(null)
const morph = ref(0)
const showConflicts = ref(true)
const showUnplaced = ref(true)

const names = computed(() => new Map(props.nodes.map((node) => [node.id, node.name])))
const nameOf = (id) => names.value.get(id) ?? `#${id}`

//: рёбра родства — для древа; остальные видымы только в графе
const kinLinks = computed(() => props.edges.filter((edge) => isKinLink(edge)))

const force = shallowRef(new Map())
const tree = computed(() =>
  buildTree({
    nodes: props.nodes,
    links: props.edges,
    rootId: props.selectedId ?? rootFallback.value,
    nodeWidth: NODE_WIDTH,
    nodeHeight: NODE_HEIGHT,
  }),
)

/** Если объект не выбран, древо строится от самого «старшего» узла графа. */
const rootFallback = computed(() => {
  const persons = props.nodes.filter((node) => node.level !== null && node.level !== undefined)
  if (!persons.length) return props.nodes[0]?.id ?? null
  return persons.reduce((best, node) => (node.level > best.level ? node : best), persons[0]).id
})

const recomputeForce = () => {
  force.value = simulateGraph({
    nodes: props.nodes,
    links: props.edges,
    width: WIDTH,
    height: HEIGHT,
  })
}

onMounted(() => {
  recomputeForce()
  const svg = select(svgRef.value)
  const behaviour = zoom()
    .scaleExtent([0.3, 2.6])
    .on('zoom', (event) => {
      select(viewportRef.value).attr('transform', event.transform.toString())
    })
  svg.call(behaviour)
  cleanup = () => svg.on('.zoom', null)
})

let cleanup = null
onBeforeUnmount(() => cleanup?.())

watch(() => [props.nodes.length, props.edges.length], recomputeForce)
watch(
  () => [props.selectedId, tree.value],
  () => {
    if (mode.value === 'tree') morph.value = 0
  },
)

/** Плавный переход между режимами: один скаляр, из него — все позиции. */
watch(mode, (next) => {
  gsap.to(morph, {
    value: next === 'tree' ? 1 : 0,
    duration: 0.65,
    ease: 'power2.inOut',
  })
})

const manual = ref(new Map())

const positions = computed(() => {
  const result = new Map()
  const springPositions = force.value
  const treePositions = tree.value.positions
  const t = morph.value
  for (const node of props.nodes) {
    const spring = springPositions.get(node.id)
    const layered = treePositions.get(node.id)
    if (spring && layered) {
      result.set(node.id, {
        x: spring.x + (layered.x - spring.x) * t,
        y: spring.y + (layered.y - spring.y) * t,
        opacity: 1,
      })
      continue
    }
    const single = spring ?? layered
    if (!single) continue
    result.set(node.id, { ...single, opacity: spring && layered ? 1 : 0.35 })
  }
  for (const [id, point] of manual.value) {
    const current = result.get(id)
    if (current) result.set(id, { ...current, x: current.x + point.dx, y: current.y + point.dy })
  }
  return result
})

const visibleEdges = computed(() => {
  const t = morph.value
  return props.edges
    .filter((edge) => (t > 0.5 ? isKinLink(edge) : true))
    .map((edge) => {
      const from = positions.value.get(edge.from_id)
      const to = positions.value.get(edge.to_id)
      if (!from || !to) return null
      return { ...edge, x1: from.x, y1: from.y, x2: to.x, y2: to.y }
    })
    .filter(Boolean)
})

const drawnNodes = computed(() =>
  props.nodes
    .map((node) => ({ node, point: positions.value.get(node.id) }))
    .filter((item) => item.point),
)

const selectedName = computed(() => (props.selectedId ? nameOf(props.selectedId) : null))

const slotLabel = { father: 'отец', mother: 'мать', parent: 'родитель' }

const conflictRows = computed(() =>
  props.conflicts.map((conflict) => {
    if (conflict.kind === 'functional_parent') {
      const slots = Object.entries(conflict.duplicated_slots ?? {})
        .map(([slot, ids]) => `${slotLabel[slot] ?? slot}: ${ids.map(nameOf).join(', ')}`)
        .join('; ')
      return {
        key: `parent-${conflict.child}`,
        title: `у ${nameOf(conflict.child)} двое в одном родительском слоте`,
        details: `${slots}. Записаны родителями: ${(conflict.parents ?? []).map(nameOf).join(', ')}`,
      }
    }
    if (conflict.kind === 'anchor_named_pair') {
      return {
        key: `anchor-${conflict.anchor}-${conflict.named}`,
        title: `«${nameOf(conflict.anchor)}» и «${nameOf(conflict.named)}» — один человек?`,
        details: `оба претендуют на слот «${slotLabel[conflict.slot] ?? conflict.slot}» у ${nameOf(conflict.child)}`,
      }
    }
    if (conflict.kind === 'level_mismatch') {
      return {
        key: `level-${conflict.edge_id}`,
        title: `ребро ${nameOf(conflict.from_id)} → ${nameOf(conflict.to_id)} спорит с уровнями`,
        details: `ожидалось ${conflict.expected_delta}, получилось ${conflict.actual_delta}`,
      }
    }
    if (conflict.kind === 'kind_disagreement') {
      return {
        key: `kinds-${conflict.from_id}-${conflict.to_id}`,
        title: `источники называют связь по-разному`,
        details: `${(conflict.kinds ?? []).join(', ')} — оставлен «${conflict.kept}»`,
      }
    }
    return { key: JSON.stringify(conflict).slice(0, 40), title: conflict.kind, details: '' }
  }),
)

const unplacedRows = computed(() =>
  props.unplaced.map((item) => ({
    id: item.entity_id,
    name: nameOf(item.entity_id),
    reason:
      {
        ritual: 'крёстные связи — поколение не определено',
        ambiguous: 'связь не определена',
        isolated: 'вне родства',
        not_kin: 'не родня',
      }[item.reason] ?? item.reason,
  })),
)

const attachDrag = (element, node) => {
  if (!element) return
  select(element).call(
    drag()
      .on('start', () => {})
      .on('drag', (event) => {
        const next = new Map(manual.value)
        const current = next.get(node.id) ?? { dx: 0, dy: 0 }
        next.set(node.id, { dx: current.dx + event.dx, dy: current.dy + event.dy })
        manual.value = next
      }),
  )
}

const initials = (name) =>
  (name || '?')
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? '')
    .join('')

const isDimmed = (id) => {
  if (!props.selectedId) return false
  if (id === props.selectedId) return false
  const direct = props.edges.some(
    (edge) =>
      (edge.from_id === props.selectedId && edge.to_id === id) ||
      (edge.to_id === props.selectedId && edge.from_id === id),
  )
  return !direct
}
</script>

<template>
  <div class="space-y-3">
    <div class="flex flex-wrap items-center gap-2">
      <div class="flex rounded-full border border-app-border/40 bg-app-panel/40 p-0.5 text-xs">
        <button
          type="button"
          class="rounded-full px-3 py-1 transition"
          :class="mode === 'graph' ? 'bg-app-primary/25 text-app-accent' : 'text-app-muted hover:text-app-text'"
          @click="mode = 'graph'"
        >
          Граф
        </button>
        <button
          type="button"
          class="rounded-full px-3 py-1 transition"
          :class="mode === 'tree' ? 'bg-app-primary/25 text-app-accent' : 'text-app-muted hover:text-app-text'"
          @click="mode = 'tree'"
        >
          Древо
        </button>
      </div>

      <button
        v-if="conflictRows.length"
        type="button"
        class="rounded-full border px-3 py-1 text-xs transition"
        :class="showConflicts ? 'border-app-accent/60 text-app-accent' : 'border-app-border/40 text-app-muted'"
        @click="showConflicts = !showConflicts"
      >
        Конфликты ({{ conflictRows.length }})
      </button>

      <button
        v-if="unplacedRows.length"
        type="button"
        class="rounded-full border px-3 py-1 text-xs transition"
        :class="showUnplaced ? 'border-app-accent/60 text-app-accent' : 'border-app-border/40 text-app-muted'"
        @click="showUnplaced = !showUnplaced"
      >
        Уровень не определён ({{ unplacedRows.length }})
      </button>

      <button
        type="button"
        class="rounded-full border border-app-border/40 px-3 py-1 text-xs text-app-muted transition hover:text-app-text"
        @click="emit('refresh')"
      >
        обновить
      </button>

      <span v-if="selectedName" class="text-xs text-app-muted">
        смотрим от: <span class="text-app-accent">{{ selectedName }}</span>
      </span>
      <span v-if="mode === 'tree'" class="text-xs text-app-muted">в древе {{ tree.positions.size }} из {{ nodes.length }}</span>
    </div>

    <p
      v-if="empty"
      class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted"
    >
      Связи пока не определены. Упоминайте людей в дневнике — карта связей построится сама.
    </p>

    <div v-else class="overflow-hidden rounded-2xl border border-app-border/40 bg-app-panel/30">
      <svg ref="svgRef" :viewBox="`0 0 ${WIDTH} ${HEIGHT}`" class="h-[420px] w-full touch-none sm:h-[520px]">
        <g ref="viewportRef">
          <line
            v-for="edge in visibleEdges"
            :key="`edge-${edge.id}`"
            :x1="edge.x1"
            :y1="edge.y1"
            :x2="edge.x2"
            :y2="edge.y2"
            :stroke="edge.canon_key === 'pair:spouse' ? 'currentColor' : 'currentColor'"
            :stroke-width="edge.canon_key === 'pair:spouse' ? 3 : 1.2"
            :stroke-dasharray="edge.bond === 'blood' ? undefined : '5 4'"
            :opacity="mode === 'tree' ? 0.5 : 0.35"
            class="text-app-accent"
          />
          <g
            v-for="item in drawnNodes"
            :key="`node-${item.node.id}`"
            :ref="(element) => attachDrag(element, item.node)"
            :transform="`translate(${item.point.x}, ${item.point.y})`"
            :opacity="item.point.opacity"
            class="cursor-grab"
            @click="emit('select', item.node.id)"
          >
            <circle
              :r="NODE_RADIUS"
              :class="[
                item.node.id === selectedId ? 'fill-app-primary/60' : 'fill-app-panel',
                isDimmed(item.node.id) ? 'opacity-40' : '',
              ]"
              stroke="currentColor"
              :stroke-width="item.node.id === selectedId ? 2.5 : 1"
              class="text-app-border"
            />
            <text
              text-anchor="middle"
              dominant-baseline="central"
              class="pointer-events-none select-none fill-current text-[11px] text-app-text"
            >
              {{ initials(item.node.name) }}
            </text>
            <text
              text-anchor="middle"
              :y="NODE_RADIUS + 14"
              class="pointer-events-none select-none fill-current text-[10px] text-app-muted"
            >
              {{ item.node.name.length > 18 ? `${item.node.name.slice(0, 17)}…` : item.node.name }}
            </text>
            <text
              v-if="item.node.is_anchor"
              text-anchor="middle"
              :y="NODE_RADIUS + 26"
              class="pointer-events-none select-none fill-current text-[9px] text-app-accent"
            >
              роль без имени
            </text>
          </g>
        </g>
      </svg>
    </div>

    <div v-if="showConflicts && conflictRows.length" class="rounded-2xl border border-app-border/40 bg-app-panel/30 p-4">
      <p class="mb-2 text-xs uppercase tracking-widest text-app-accent">Конфликты</p>
      <ul class="space-y-1 text-sm">
        <li v-for="row in conflictRows" :key="row.key" class="text-app-text">
          {{ row.title }}
          <span v-if="row.details" class="text-xs text-app-muted">— {{ row.details }}</span>
        </li>
      </ul>
    </div>

    <div v-if="showUnplaced && unplacedRows.length" class="rounded-2xl border border-app-border/40 bg-app-panel/30 p-4">
      <p class="mb-2 text-xs uppercase tracking-widest text-app-accent">Уровень не определён</p>
      <p class="mb-2 text-xs text-app-muted">
        Эти люди есть в знаниях, но поколение по ним не определено: в древо они не попадают, чтобы не встать на нулевой слой по умолчанию.
      </p>
      <ul class="flex flex-wrap gap-2 text-xs">
        <li
          v-for="row in unplacedRows"
          :key="row.id"
          class="rounded-full border border-app-border/40 px-2 py-0.5 text-app-muted"
        >
          {{ row.name }} — {{ row.reason }}
        </li>
      </ul>
    </div>
  </div>
</template>
