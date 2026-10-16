import { describe, expect, it } from 'vitest'
import { buildTree, buildUnits, isSpouseLink, orientLink, simulateGraph } from '../utils/relation-layout'

/*
 * Раскладка — чистые функции, поэтому проверяется без DOM и без бэкенда:
 * ориентация рёбер по знаку дельты, супружеские единицы, привязка слоёв к
 * уровням родства и детерминированность пружин.
 */

const node = (id, level, component = 1) => ({ id, name: `P${id}`, level, component })

const edge = (id, from_id, to_id, delta, extra = {}) => ({
  id,
  from_id,
  to_id,
  delta,
  canon_key: extra.canon_key ?? 'shape:0:1:0:blood',
  via_marriage: extra.via_marriage ?? false,
  bond: extra.bond ?? 'blood',
})

describe('ориентация ребра', () => {
  it('старший считается родителем независимо от направления записи', () => {
    // «20 — отец 10»: от 10 к 20, дельта отрицательная
    expect(orientLink(edge(1, 10, 20, -1))).toEqual({ parent: 20, child: 10 })
    // «10 — ребёнок 20»: та же пара с другой стороны
    expect(orientLink(edge(2, 20, 10, 1))).toEqual({ parent: 20, child: 10 })
  })

  it('при нулевой дельте иерархии нет', () => {
    expect(orientLink(edge(3, 10, 20, 0, { canon_key: 'pair:spouse', via_marriage: true }))).toBeNull()
  })

  it('неродственное ребро в иерархию не берётся', () => {
    expect(orientLink({ id: 4, from_id: 10, to_id: 20, delta: null, canon_key: 'kind:friend' })).toBeNull()
  })
})

describe('супружеские единицы', () => {
  it('муж и жена — одна единица, сват и свояк — нет', () => {
    const links = [
      edge(1, 10, 20, 0, { canon_key: 'pair:spouse', via_marriage: true }),
      edge(2, 20, 30, 0, { canon_key: 'pair:svojak', via_marriage: true }),
    ]
    expect(isSpouseLink(links[0])).toBe(true)
    expect(isSpouseLink(links[1])).toBe(false)
    const { units, unitOf } = buildUnits([10, 20, 30], links)
    expect(unitOf.get(10)).toBe(unitOf.get(20))
    expect(unitOf.get(30)).not.toBe(unitOf.get(20))
    expect(units).toHaveLength(2)
  })
})

describe('древо', () => {
  const nodes = [node(1, 0), node(2, 0), node(3, 1), node(4, 2)]
  const links = [
    edge(1, 3, 1, -1), // 1 — отец 3
    edge(2, 3, 2, -1), // 2 — мать 3
    edge(3, 3, 4, 1), //  4 — ребёнок 3
  ]

  it('слои прибиты к уровням родства', () => {
    const layout = buildTree({ nodes, links, rootId: 3 })
    const y = (id) => layout.positions.get(id).y
    expect(y(1)).toBeLessThan(y(3))
    expect(y(2)).toBe(y(1))
    expect(y(4)).toBeGreaterThan(y(3))
    // слой — это ровно уровень родства, а не «сколько шагов от корня»
    expect(Math.round((y(3) - y(1)) / 104)).toBe(1)
    expect(Math.round((y(4) - y(3)) / 104)).toBe(1)
  })

  it('в древо попадают только родня компоненты корня', () => {
    const mixed = [...nodes, node(9, null), node(10, 5, 99)]
    const layout = buildTree({ nodes: mixed, links, rootId: 3 })
    expect(layout.positions.has(9)).toBe(false)
    expect(layout.positions.has(10)).toBe(false)
    expect(layout.unplaced.map((item) => [item.id, item.reason])).toEqual([
      [9, 'no-level'],
      [10, 'other-component'],
    ])
  })

  it('без корня с определённым уровнем древо пустое и все вне древа', () => {
    const layout = buildTree({ nodes: [node(9, null)], links: [], rootId: 9 })
    expect(layout.positions.size).toBe(0)
    expect(layout.unplaced).toEqual([{ id: 9, reason: 'no-root' }])
  })

  it('порядок внутри слоя решает dagre, а не порядок входа', () => {
    const many = [node(1, 0), node(2, 0), node(3, 0), node(4, 0), node(5, 1)]
    const fan = [1, 2, 3, 4].map((parent) => edge(parent, 5, parent, -1))
    const layout = buildTree({ nodes: many, links: fan, rootId: 5 })
    const xs = [1, 2, 3, 4].map((id) => layout.positions.get(id).x)
    expect(new Set(xs).size).toBe(4)
    expect(xs).toEqual([...xs].sort((a, b) => a - b))
  })
})

describe('пружины', () => {
  it('одинаковый вход даёт одинаковую картинку', () => {
    const nodes = [node(1, 0), node(2, 1), node(3, 1)]
    const links = [edge(1, 2, 1, -1), edge(2, 3, 1, -1)]
    const first = simulateGraph({ nodes, links, ticks: 60 })
    const second = simulateGraph({ nodes, links, ticks: 60 })
    for (const id of [1, 2, 3]) {
      expect(first.get(id).x).toBeCloseTo(second.get(id).x, 6)
      expect(first.get(id).y).toBeCloseTo(second.get(id).y, 6)
    }
  })

  it('все узлы получают координаты', () => {
    const nodes = [node(1, 0), node(2, 1)]
    const points = simulateGraph({ nodes, links: [edge(1, 2, 1, -1)], ticks: 30 })
    expect(points.size).toBe(2)
    for (const point of points.values()) {
      expect(Number.isFinite(point.x)).toBe(true)
      expect(Number.isFinite(point.y)).toBe(true)
    }
  })
})
