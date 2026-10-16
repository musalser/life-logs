import dagre from 'dagre'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
} from 'd3-force'

/*
 * Раскладки графа связей — чистые функции: ни Vue, ни запросов.
 *
 * Уровни приходят с бэкенда уже согласованными (внутри компоненты родства, с
 * постоянным сдвигом), поэтому клиент не считает поколения сам: в древе он лишь
 * вычитает уровень корня. Раскладку слоёв делает dagre — руками порядок узлов и
 * минимизацию пересечений не пишем; наша часть — сказать, кто родитель, кто
 * ребёнок и на каком слое каждый стоит.
 *
 * Супруги — одна единица раскладки: иначе муж и жена, связанные ребром (0, 0),
 * вставали бы в разные слои или наезжали друг на друга.
 */

/** Ребро родства: поколение определено (у ритуальных и «не родни» — нет). */
export const isKinLink = (edge) => edge.delta !== null && edge.delta !== undefined

/** Ребро супругов: пара (0, 0) через брак. Сват и свояк сюда не попадают. */
export const isSpouseLink = (edge) => edge.canon_key === 'pair:spouse'

/**
 * Родитель и ребёнок ребра: старший считается родителем.
 *
 * Направление ребра — как у источника («B — это вид A»), поэтому знак дельты и
 * решает, кто выше; при нулевой дельте иерархии нет.
 */
export const orientLink = (edge) => {
  if (!isKinLink(edge) || edge.delta === 0) return null
  return edge.delta < 0
    ? { parent: edge.to_id, child: edge.from_id }
    : { parent: edge.from_id, child: edge.to_id }
}

/** Супружеские единицы: union-find по рёбрам pair:spouse. */
export const buildUnits = (nodeIds, links) => {
  const parent = new Map(nodeIds.map((id) => [id, id]))
  const find = (id) => {
    let root = id
    while (parent.get(root) !== root) root = parent.get(root)
    let node = id
    while (parent.get(node) !== root) {
      const next = parent.get(node)
      parent.set(node, root)
      node = next
    }
    return root
  }
  const union = (left, right) => {
    const a = find(left)
    const b = find(right)
    if (a !== b) parent.set(a, b)
  }
  for (const link of links) {
    if (!isSpouseLink(link)) continue
    if (!parent.has(link.from_id) || !parent.has(link.to_id)) continue
    union(link.from_id, link.to_id)
  }
  const groups = new Map()
  for (const id of nodeIds) {
    const root = find(id)
    if (!groups.has(root)) groups.set(root, [])
    groups.get(root).push(id)
  }
  const unitOf = new Map()
  const units = []
  for (const [key, members] of groups) {
    members.sort((a, b) => a - b)
    units.push({ key, members, anchor: members[0] })
    for (const id of members) unitOf.set(id, key)
  }
  return { unitOf, units }
}

/**
 * Раскладка «граф»: пружины d3-force.
 *
 * Начальные позиции детерминированы (окружность по порядку узлов) и число шагов
 * фиксировано, поэтому одинаковый вход даёт одинаковую картинку — иначе граф
 * дрожал бы при каждом обновлении, а тесты были бы невозможны.
 */
export const simulateGraph = ({ nodes, links, width = 960, height = 660, ticks = 320 }) => {
  const count = Math.max(1, nodes.length)
  const radius = Math.min(width, height) / 2.6
  const points = nodes.map((node, index) => {
    const angle = (index / count) * Math.PI * 2
    return {
      id: node.id,
      x: width / 2 + Math.cos(angle) * radius,
      y: height / 2 + Math.sin(angle) * radius,
    }
  })
  const springs = links.map((link) => ({
    source: link.from_id,
    target: link.to_id,
    strength: isSpouseLink(link) ? 0.9 : 0.5,
  }))
  const simulation = forceSimulation(points)
    .force('charge', forceManyBody().strength(-560))
    .force(
      'link',
      forceLink(springs)
        .id((point) => point.id)
        .distance(120)
        .strength((spring) => spring.strength),
    )
    .force('center', forceCenter(width / 2, height / 2))
    .force('collide', forceCollide(48))
    .force('x', forceX(width / 2).strength(0.04))
    .force('y', forceY(height / 2).strength(0.04))
    .stop()
  for (let step = 0; step < ticks; step += 1) simulation.tick()
  return new Map(points.map((point) => [point.id, { x: point.x, y: point.y }]))
}

/**
 * Раскладка «древо»: слои по уровням родства, порядок внутри слоя — dagre.
 *
 * Берём только компоненту выбранного человека и только узлы с определённым
 * уровнем: ритуальные и одинокие честно уходят в «вне древа», а не встают на
 * нулевой слой. Остовное дерево строим обходом от корня, чтобы у dagre не было
 * циклов даже при испорченном ребре (его ловит диагностика на бэкенде).
 */
export const buildTree = ({
  nodes,
  links,
  rootId,
  nodeWidth = 148,
  nodeHeight = 46,
  gapX = 28,
  gapY = 104,
  margin = 40,
}) => {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const root = byId.get(rootId)
  const empty = { positions: new Map(), links: [], levels: new Map(), unplaced: [], width: 0, height: 0 }
  if (!root || root.level === null || root.level === undefined) {
    return { ...empty, unplaced: nodes.map((node) => ({ id: node.id, reason: 'no-root' })) }
  }

  const treeIds = nodes
    .filter((node) => node.level !== null && node.level !== undefined && node.component === root.component)
    .map((node) => node.id)
  const inTree = new Set(treeIds)
  const unplaced = nodes
    .filter((node) => !inTree.has(node.id))
    .map((node) => ({ id: node.id, reason: node.level === null || node.level === undefined ? 'no-level' : 'other-component' }))

  const kinLinks = links.filter((link) => inTree.has(link.from_id) && inTree.has(link.to_id) && isKinLink(link))
  const levels = new Map(treeIds.map((id) => [id, byId.get(id).level - root.level]))

  const adjacency = new Map()
  for (const link of kinLinks) {
    if (!adjacency.has(link.from_id)) adjacency.set(link.from_id, [])
    if (!adjacency.has(link.to_id)) adjacency.set(link.to_id, [])
    adjacency.get(link.from_id).push(link)
    adjacency.get(link.to_id).push(link)
  }

  const visited = new Set([root.id])
  const queue = [root.id]
  const treeEdges = []
  while (queue.length) {
    const current = queue.shift()
    for (const link of adjacency.get(current) ?? []) {
      const oriented = orientLink(link)
      if (!oriented) continue
      const next = oriented.parent === current ? oriented.child : oriented.parent
      if (visited.has(next)) continue
      visited.add(next)
      treeEdges.push(oriented)
      queue.push(next)
    }
  }

  const { unitOf, units } = buildUnits([...visited], kinLinks)
  const graph = new dagre.graphlib.Graph({ multigraph: false })
  graph.setGraph({ rankdir: 'TB', nodesep: gapX, ranksep: gapY, marginx: margin, marginy: margin })
  graph.setDefaultEdgeLabel(() => ({}))
  for (const unit of units) {
    graph.setNode(unit.key, {
      width: unit.members.length * nodeWidth + (unit.members.length - 1) * 14,
      height: nodeHeight,
    })
  }
  for (const oriented of treeEdges) {
    const from = unitOf.get(oriented.parent)
    const to = unitOf.get(oriented.child)
    if (from === undefined || to === undefined || from === to) continue
    if (graph.hasEdge(from, to)) continue
    const minlen = Math.max(1, Math.abs(levels.get(oriented.child) - levels.get(oriented.parent)))
    graph.setEdge(from, to, { minlen, weight: 3 })
  }
  dagre.layout(graph)

  // Слой прибиваем к уровню родства: dagre решает порядок внутри слоя, а высоту
  // задаёт поколение, иначе дед и отец могли бы поменяться местами.
  const values = [...levels.values()]
  const lowest = values.length ? Math.min(...values) : 0
  const positions = new Map()
  for (const unit of units) {
    const laid = graph.node(unit.key)
    if (!laid) continue
    const y = margin + (levels.get(unit.anchor) - lowest) * gapY + nodeHeight / 2
    const span = unit.members.length * nodeWidth + (unit.members.length - 1) * 14
    unit.members.forEach((id, index) => {
      positions.set(id, {
        x: laid.x - span / 2 + nodeWidth / 2 + index * (nodeWidth + 14),
        y,
      })
    })
  }

  const drawn = kinLinks.map((link) => ({
    ...link,
    from: positions.get(link.from_id) ?? null,
    to: positions.get(link.to_id) ?? null,
  }))
  const xs = [...positions.values()].map((point) => point.x)
  const ys = [...positions.values()].map((point) => point.y)
  return {
    positions,
    links: drawn,
    levels,
    unplaced,
    width: xs.length ? Math.max(...xs) + nodeWidth : 0,
    height: ys.length ? Math.max(...ys) + nodeHeight : 0,
  }
}
