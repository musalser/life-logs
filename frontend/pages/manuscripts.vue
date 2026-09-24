<template>
  <div class="flex flex-col gap-4">
    <!-- header -->
    <header class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-4">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 class="text-2xl font-semibold">Рукописи</h1>
          <p class="text-sm text-app-muted">
            Загрузите страницу, распознайте её, проверьте текст и подтвердите — подтверждённые
            страницы становятся обучающим набором персональной модели.
          </p>
        </div>

        <div class="flex flex-wrap items-center gap-2">
          <div v-if="authors.length" class="flex items-center gap-2">
            <span class="text-xs uppercase tracking-widest text-app-accent">Автор</span>
            <select
              v-model.number="authorId"
              class="app-select min-w-[9rem]"
              style="min-width: 9rem"
              @change="onAuthorChange"
            >
              <option v-for="a in authors" :key="a.id" :value="a.id">{{ a.name }}</option>
            </select>
          </div>

          <button type="button" class="app-ghost" @click="showAuthorForm = !showAuthorForm">
            {{ showAuthorForm ? 'Отмена' : '+ автор' }}
          </button>

          <button
            type="button"
            class="app-primary"
            :disabled="!authorId || uploading"
            @click="fileInput?.click()"
          >
            {{ uploading ? 'Загрузка…' : 'Загрузить страницу' }}
          </button>
          <input
            ref="fileInput"
            type="file"
            accept="image/*"
            class="hidden"
            @change="onUpload"
          />
        </div>
      </div>

      <form
        v-if="showAuthorForm"
        class="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-app-border/40 bg-app-input/40 p-3"
        @submit.prevent="createAuthor"
      >
        <input v-model="newAuthorName" class="app-input" placeholder="Имя автора (например, «Дед»)" />
        <button type="submit" class="app-primary" :disabled="!newAuthorName.trim() || creatingAuthor">
          {{ creatingAuthor ? 'Создание…' : 'Создать автора' }}
        </button>
      </form>
    </header>

    <p v-if="error" class="rounded-xl border border-app-border/40 bg-app-error/10 px-4 py-3 text-sm text-app-error">
      {{ error }}
    </p>
    <p v-else-if="notice" class="rounded-xl border border-app-primary/40 bg-app-primary/10 px-4 py-3 text-sm text-app-accent">
      {{ notice }}
    </p>

    <!-- training outcome after confirmation -->
    <section
      v-if="training"
      class="rounded-2xl border p-4"
      :class="trainingBannerClass"
    >
      <div class="flex flex-wrap items-center justify-between gap-2">
        <p class="text-sm font-semibold">{{ trainingTitle }}</p>
        <span class="rounded-full bg-black/20 px-2.5 py-0.5 text-[11px] font-semibold">
          {{ training.outcome }}
        </span>
      </div>
      <p v-if="training.message" class="mt-1 text-sm opacity-90">{{ training.message }}</p>
      <p v-if="trainingMetrics" class="mt-2 text-sm">
        <template v-if="training.metrics?.validation">
          CER <span class="font-semibold">{{ formatPercent(training.metrics.validation.cer) }}</span>
          · WER <span class="font-semibold">{{ formatPercent(training.metrics.validation.wer) }}</span>
        </template>
        <template v-if="training.metrics?.baseline">
          <span class="opacity-80">
            (базовая модель: CER {{ formatPercent(training.metrics.baseline.cer) }}
            · WER {{ formatPercent(training.metrics.baseline.wer) }})
          </span>
        </template>
      </p>
      <p v-if="training.metrics?.training?.line_height" class="mt-1 text-xs opacity-80">
        Высота строки: {{ training.metrics.training.line_height }} px
        <template v-if="training.metrics.training.line_height_override_from">
          (в чекпойнте {{ training.metrics.training.line_height_override_from }} px)
        </template>
      </p>
      <p v-if="training.lines_required" class="mt-1 text-xs opacity-80">
        Подтверждённых строк: {{ training.lines_collected }} / {{ training.lines_required }}
      </p>
      <p v-if="training.model_version" class="mt-1 text-xs opacity-80">
        Версия v{{ training.model_version.version }} — {{ training.model_version.status }}
      </p>
    </section>

    <div v-if="!authors.length && !showAuthorForm" class="rounded-2xl border border-dashed border-app-border/40 px-4 py-8 text-center text-sm text-app-muted">
      Сначала создайте автора рукописи.
    </div>

    <div
      v-else-if="authorId"
      class="grid grid-cols-1 gap-4 xl:grid-cols-[240px_minmax(0,1fr)_420px]"
    >
      <!-- pages -->
      <aside class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-3">
        <div class="mb-2 flex items-center justify-between gap-2">
          <p class="text-xs uppercase tracking-widest text-app-accent">Страницы</p>
          <div class="flex items-center gap-2">
            <button
              type="button"
              class="page-sort"
              :class="sortByOov ? 'page-sort--on' : ''"
              :disabled="!lexiconAvailableInList"
              title="Сначала страницы с наибольшим числом слов, которых нет в словаре"
              @click="sortByOov = !sortByOov"
            >
              по словарю
            </button>
            <span class="text-xs text-app-muted">{{ pages.length }} шт.</span>
          </div>
        </div>

        <p v-if="pagesLoading" class="text-xs text-app-muted">Загрузка…</p>
        <p v-else-if="!pages.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-xs text-app-muted">
          Страниц пока нет.
        </p>

        <div v-else class="no-scrollbar max-h-[70vh] space-y-2 overflow-y-auto pr-1">
          <div
            v-for="item in sortedPages"
            :key="item.page_id"
            role="button"
            tabindex="0"
            class="w-full cursor-pointer rounded-xl border px-3 py-2 text-left transition"
            :class="
              page?.page_id === item.page_id
                ? 'border-app-primary/70 bg-app-primary/10'
                : 'border-app-border/40 bg-app-panel/30 hover:border-app-primary/40 hover:bg-app-panel/50'
            "
            @click="selectPage(item.page_id)"
            @keydown.enter.prevent="selectPage(item.page_id)"
          >
            <div class="flex items-center justify-between gap-2">
              <p class="text-sm font-medium">Стр. #{{ item.page_id }}</p>
              <div class="flex items-center gap-1">
                <span class="rounded-full px-2 py-0.5 text-[10px] font-semibold" :class="statusOf(item.status).class">
                  {{ statusOf(item.status).label }}
                </span>
                <button
                  type="button"
                  class="page-delete"
                  title="Удалить страницу"
                  :disabled="deletingPageId === item.page_id"
                  @click.stop="deletePage(item)"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5">
                    <path d="M6 2h12l2 2v2H2V4l2-2zm2 6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H8zM10 10h4v10h-4V10z"></path>
                  </svg>
                </button>
              </div>
            </div>
            <p class="mt-1 text-[11px] text-app-muted">
              {{ item.line_count }} строк · {{ formatDate(item.created_at) }}
            </p>
            <p v-if="item.oov_count" class="text-[11px] text-violet-300">
              не в словаре: {{ item.oov_count }}
            </p>
            <p v-if="item.prediction_cer != null" class="text-[11px] text-app-muted">
              расхождение: CER {{ formatPercent(item.prediction_cer) }}
            </p>
          </div>
        </div>
      </aside>

      <!-- image + overlay -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-3">
        <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <button type="button" class="app-icon" title="Уменьшить" @click="zoomBy(1 / 1.25)">−</button>
            <span class="w-12 text-center text-xs text-app-muted">{{ Math.round(zoom * 100) }}%</span>
            <button type="button" class="app-icon" title="Увеличить" @click="zoomBy(1.25)">+</button>
            <button type="button" class="app-ghost" @click="fitToWidth">По ширине</button>
            <button type="button" class="app-ghost" @click="zoomTo(1)">100%</button>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            <button
              type="button"
              class="app-primary"
              :disabled="!canRecognize || recognizing"
              @click="recognize"
            >
              {{ recognizing ? 'Распознавание…' : canRecognize && page?.status !== 'UPLOADED' ? 'Распознать заново' : 'Распознать' }}
            </button>
            <button
              type="button"
              class="app-ghost"
              :disabled="!page?.lines.length || correcting"
              title="Спросить языковую модель и показать её правки отдельно: распознанный текст не меняется, пока вы не примете предложение"
              @click="suggestWithLlm"
            >
              {{ correcting ? 'Модель думает…' : 'Предложить правки' }}
            </button>
            <button
              v-if="page?.lines.some(hasSuggestion)"
              type="button"
              class="app-ghost app-ghost--verify"
              :disabled="!verifiedSuggestionCount || acceptingSuggestions"
              :title="verifiedSuggestionCount
                ? 'Принять только те предложения, где каждое новое слово есть в словаре'
                : 'Среди предложений нет ни одного полностью проверенного словарём'"
              @click="acceptVerifiedSuggestions"
            >
              {{ acceptingSuggestions ? 'Принятие…' : `Принять проверенные (${verifiedSuggestionCount})` }}
            </button>
            <button
              type="button"
              class="app-success"
              :disabled="!canConfirm || confirming"
              @click="confirmPage"
            >
              {{ confirming ? 'Подтверждение…' : 'Подтвердить страницу' }}
            </button>
          </div>
        </div>

        <div class="mb-2 flex flex-wrap items-center gap-3 text-[11px] text-app-muted">
          <span class="flex items-center gap-1">
            <span class="legend-box" :style="legendStyle('normal')"></span> уверенно
          </span>
          <span class="flex items-center gap-1">
            <span class="legend-box" :style="legendStyle('warning')"></span>
            сомнение &lt; {{ thresholds.warning }}
          </span>
          <span class="flex items-center gap-1">
            <span class="legend-box" :style="legendStyle('critical')"></span>
            плохо &lt; {{ thresholds.critical }}
          </span>
          <span v-if="lexiconAvailable" class="flex items-center gap-1">
            <span class="legend-box legend-box--oov"></span>
            нет в словаре
          </span>
          <span v-else-if="page?.lines.length" class="text-violet-300/80">
            словарь не установлен — слова не проверяются
          </span>
          <span v-if="hasStaleLines" class="flex items-center gap-1 text-amber-300">
            пунктир — разметка устарела после правки
          </span>
        </div>

        <div
          ref="viewport"
          class="relative h-[calc(100vh_-_320px)] min-h-[420px] overflow-hidden rounded-xl border border-app-border/40 bg-app-input/60"
          :class="panning ? 'cursor-grabbing' : 'cursor-grab'"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
          @pointerleave="onPointerUp"
          @wheel="onWheel"
        >
          <p v-if="imageError" class="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-app-error">
            {{ imageError }}
          </p>
          <p v-else-if="pageLoading || imageLoading" class="absolute inset-0 flex items-center justify-center text-sm text-app-muted">
            Загрузка изображения…
          </p>
          <p v-else-if="!page" class="absolute inset-0 flex items-center justify-center text-sm text-app-muted">
            Выберите страницу слева.
          </p>

          <div
            v-else
            class="absolute left-0 top-0 origin-top-left"
            :style="contentStyle"
          >
            <img
              v-if="imageUrl"
              :src="imageUrl"
              alt="Страница рукописи"
              class="pointer-events-none absolute left-0 top-0 select-none"
              :style="{ width: `${pageWidth}px`, height: `${pageHeight}px` }"
              draggable="false"
            />

            <!-- polygons follow the (curved) text lines instead of cutting them -->
            <svg
              v-if="imageUrl"
              class="absolute left-0 top-0 overflow-visible"
              :width="pageWidth"
              :height="pageHeight"
              :viewBox="`0 0 ${pageWidth} ${pageHeight}`"
            >
              <polygon
                v-for="line in page.lines"
                :key="`line-${line.id}`"
                :points="shapePoints(line)"
                :class="[
                  'line-poly',
                  activeLineId === line.id ? 'line-poly--active' : '',
                  line.words_stale ? 'line-poly--stale' : '',
                ]"
                :style="{ strokeWidth: lineStrokeWidth }"
              />

              <polygon
                v-for="word in allWords"
                :key="`word-${word.id}`"
                :points="shapePoints(word)"
                data-word
                class="word-poly"
                :class="[
                  `word-poly--${word.confidence_level || 'critical'}`,
                  word.in_lexicon === false ? 'word-poly--oov' : '',
                  activeWordId === word.id ? 'word-poly--active' : '',
                  parentLineIsStale(word) ? 'word-poly--stale' : '',
                ]"
                :style="{ strokeWidth: wordStrokeWidth }"
                @click.stop="onWordClick(word)"
              >
                <title>{{ wordTitle(word) }}</title>
              </polygon>
            </svg>
          </div>
        </div>

        <p v-if="page" class="mt-2 text-[11px] text-app-muted">
          {{ pageWidth }}×{{ pageHeight }} px ·
          строк {{ page.lines.length }} ·
          слов {{ totalWords }}
          <template v-if="lexiconAvailable && page.oov_count">
            · <span class="text-violet-300">не в словаре: {{ page.oov_count }}</span>
          </template>
          <template v-if="page.recognition_model_version_id">
            · модель #{{ page.recognition_model_version_id }}
          </template>
        </p>
      </section>

      <!-- transcription -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-3">
        <div class="mb-2 flex items-center justify-between">
          <p class="text-xs uppercase tracking-widest text-app-accent">Транскрипция</p>
          <span v-if="dirtyCount" class="text-[11px] text-amber-300">не сохранено: {{ dirtyCount }}</span>
          <span v-else-if="page" class="text-[11px] text-app-muted">все правки сохранены</span>
        </div>

        <p v-if="!page" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Выберите страницу, чтобы увидеть текст.
        </p>
        <p v-else-if="!page.lines.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-sm text-app-muted">
          Текст ещё не распознан.
        </p>

        <div v-else class="no-scrollbar max-h-[calc(100vh_-_320px)] space-y-2 overflow-y-auto pr-1">
          <article
            v-for="line in page.lines"
            :key="`text-${line.id}`"
            class="rounded-xl border px-3 py-2 transition"
            :class="
              activeLineId === line.id
                ? 'border-app-primary/70 bg-app-primary/5'
                : 'border-app-border/40 bg-app-panel/30'
            "
            @click="activeLineId = line.id"
          >
            <div class="mb-1 flex flex-wrap items-center gap-2">
              <span class="text-[11px] text-app-muted">Строка {{ line.order + 1 }}</span>
              <span
                v-if="line.words.length"
                class="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                :class="confidenceBadgeClass(worstConfidence(line))"
              >
                min conf {{ formatConfidence(worstConfidence(line)) }}
              </span>
              <span
                v-if="hasSuggestion(line)"
                class="rounded-full bg-app-primary/20 px-2 py-0.5 text-[10px] font-semibold text-app-accent"
                title="У модели есть предложение по этой строке"
              >
                предложение модели
              </span>
              <span
                v-if="line.oov_count"
                class="rounded-full bg-violet-500/20 px-2 py-0.5 text-[10px] font-semibold text-violet-300"
                title="Столько слов этой строки отсутствует в словаре — проверьте их"
              >
                не в словаре: {{ line.oov_count }}
              </span>
              <span
                v-if="line.words_stale"
                class="rounded-full bg-amber-400/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300"
                title="Текст изменён, поэтому рамки слов больше не соответствуют разбиению на слова"
              >
                разметка устарела
              </span>
              <span v-if="savingLines.has(line.id)" class="text-[10px] text-app-muted">сохранение…</span>
            </div>

            <textarea
              :ref="(el) => setTextareaRef(line.id, el)"
              :value="drafts[line.id]"
              :disabled="readOnly"
              rows="2"
              class="app-textarea"
              :placeholder="line.predicted_text ? '' : 'пустая строка'"
              @input="onDraftInput(line.id, $event.target.value)"
              @focus="activeLineId = line.id"
              @blur="saveLine(line)"
            ></textarea>

            <!--
              the model's proposal: shown as a diff, never written into the
              transcription until the user accepts it
            -->
            <div
              v-if="hasSuggestion(line)"
              class="mt-1 rounded-lg border border-app-primary/30 bg-app-primary/5 px-2 py-1.5"
            >
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-[10px] uppercase tracking-wider text-app-accent">
                  Предложение модели{{ line.suggested_by ? ` (${line.suggested_by})` : '' }}
                </span>
                <span
                  class="rounded-full px-2 py-0.5 text-[10px] font-semibold"
                  :class="suggestionBadge(line).class"
                  :title="suggestionBadge(line).title"
                >
                  {{ suggestionBadge(line).label }}
                </span>
              </div>

              <div class="mt-1 flex flex-wrap items-center gap-1">
                <!-- one click applies one word: the user rarely likes them all -->
                <button
                  v-for="(change, index) in line.suggestion_changes"
                  :key="`chg-${line.id}-${index}`"
                  type="button"
                  class="change-chip"
                  :class="[changeClass(change), readOnly ? 'change-chip--locked' : '']"
                  :disabled="readOnly || busySuggestionLine === line.id"
                  :title="readOnly
                    ? 'Страница подтверждена — правки больше не принимаются'
                    : `Применить только это: «${change.before || 'добавить'} → ${change.after || 'удалить'}»`"
                  @click.stop="acceptChange(line, index)"
                >
                  <span v-if="change.before" class="line-through opacity-70">{{ change.before }}</span>
                  <span v-if="change.before && change.after" class="mx-0.5 opacity-60">→</span>
                  <span v-if="change.after">{{ change.after }}</span>
                  <span v-if="!change.before" class="ml-0.5 text-[9px] uppercase opacity-70">добавить</span>
                  <span v-if="!change.after" class="ml-0.5 text-[9px] uppercase opacity-70">удалить</span>
                </button>
                <span v-if="!line.suggestion_changes.length" class="text-[11px] text-app-muted">
                  только пунктуация — примите строку целиком
                </span>
              </div>

              <p class="mt-1 text-[11px] text-app-muted">{{ line.suggested_text }}</p>

              <div v-if="!readOnly" class="mt-1 flex items-center gap-2">
                <button
                  type="button"
                  class="app-mini app-mini--ok"
                  :disabled="busySuggestionLine === line.id"
                  title="Применить всё предложение целиком"
                  @click.stop="acceptSuggestion(line)"
                >
                  Принять всё
                </button>
                <button
                  type="button"
                  class="app-mini"
                  :disabled="busySuggestionLine === line.id"
                  @click.stop="dismissSuggestion(line)"
                >
                  Скрыть
                </button>
              </div>
            </div>

            <!-- dictionary misses of this line: click jumps to the word on the page -->
            <div v-if="line.oov_words?.length" class="mt-1 flex flex-wrap items-center gap-1">
              <span class="text-[10px] text-app-muted">не в словаре:</span>
              <button
                v-for="(miss, index) in line.oov_words"
                :key="`oov-${line.id}-${index}`"
                type="button"
                class="oov-chip"
                title="Слова нет в словаре — проверьте его"
                @click.stop="onOovChipClick(line, miss)"
              >
                {{ miss }}
              </button>
            </div>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { authFetch } from '~/utils/auth-fetch'

const { public: { apiBaseUrl } } = useRuntimeConfig()
const route = useRoute()
const router = useRouter()

const STATUS = {
  UPLOADED: { label: 'Загружено', class: 'bg-app-input/60 text-app-muted' },
  RECOGNIZED: { label: 'Распознано', class: 'bg-app-primary/20 text-app-accent' },
  EDITING: { label: 'Правка', class: 'bg-amber-400/20 text-amber-300' },
  CONFIRMED: { label: 'Подтверждено', class: 'bg-emerald-500/20 text-emerald-300' },
}

const CONFIDENCE = {
  normal: { border: 'rgba(100, 116, 139, 0.65)', background: 'rgba(148, 163, 184, 0.08)' },
  warning: { border: '#f59e0b', background: 'rgba(245, 158, 11, 0.22)' },
  critical: { border: '#ef4444', background: 'rgba(239, 68, 68, 0.28)' },
}

const authors = ref([])
const authorId = ref(null)
const pages = ref([])
const deletingPageId = ref(null)
const sortByOov = ref(false)
const page = ref(null)
const drafts = reactive({})
const savingLines = reactive(new Set())

const fileInput = ref(null)
const viewport = ref(null)
const textareaRefs = new Map()

const zoom = ref(1)
const panX = ref(0)
const panY = ref(0)
const panning = ref(false)
let panStart = null

const imageUrl = ref('')
const imageError = ref('')
const naturalSize = ref({ width: 0, height: 0 })

const error = ref('')
const notice = ref('')
const training = ref(null)

const showAuthorForm = ref(false)
const newAuthorName = ref('')
const creatingAuthor = ref(false)

const pagesLoading = ref(false)
const pageLoading = ref(false)
const imageLoading = ref(false)
const uploading = ref(false)
const recognizing = ref(false)
const confirming = ref(false)
const correcting = ref(false)

const activeLineId = ref(null)
const activeWordId = ref(null)
const busySuggestionLine = ref(null)
const acceptingSuggestions = ref(false)
// programmatic query updates must not trigger the route watcher twice
let skipQueryWatch = false

// ---------------------------------------------------------------- derived

const allWords = computed(() => (page.value?.lines || []).flatMap((l) => l.words))
const wordLineMap = computed(() => {
  const map = new Map()
  for (const line of page.value?.lines || []) {
    for (const word of line.words) map.set(word.id, line)
  }
  return map
})

/** The model proposed something different from what the line currently holds. */
const hasSuggestion = (line) =>
  Boolean(line.suggested_text) &&
  line.suggested_text.trim() !== effectiveText(line).trim()
const pendingSuggestionCount = computed(
  () => (page.value?.lines || []).filter(hasSuggestion).length
)
const verifiedSuggestionCount = computed(
  () => (page.value?.lines || []).filter((line) => hasSuggestion(line) && line.suggestion_verified).length
)
/** Why a proposal can or cannot be accepted in bulk. */
const suggestionBadge = (line) => {
  if (line.suggestion_verified) {
    return {
      label: 'проверено словарём',
      class: 'bg-emerald-500/15 text-emerald-300',
      title: 'Каждое новое слово есть в словаре — можно принять',
    }
  }
  if (!line.suggestion_changes?.length) {
    return {
      label: 'только пунктуация',
      class: 'bg-amber-400/20 text-amber-300',
      title: 'Модель меняет только пунктуацию: проверять словарём нечего, решайте сами',
    }
  }
  if (line.suggestion_changes.some((change) => !change.after)) {
    return {
      label: 'удаление или склейка',
      class: 'bg-amber-400/20 text-amber-300',
      title: 'Модель убирает или склеивает слова — такое словарём не проверить, прочитайте строку',
    }
  }
  if (line.suggestion_changes.some((change) => change.in_lexicon === false)) {
    return {
      label: 'есть слова не из словаря',
      class: 'bg-violet-500/20 text-violet-300',
      title: 'Модель предлагает слова, которых нет ни в словаре, ни в ваших подтверждённых страницах',
    }
  }
  return {
    label: 'нет словаря',
    class: 'bg-app-input/60 text-app-muted',
    title: 'Словарь не установлен, поэтому проверить правки нечем',
  }
}

/** Violet again: the same colour as the OOV marks, because it is the same fact. */
const changeClass = (change) => {
  if (change.in_lexicon === true) return 'change-chip--ok'
  if (change.in_lexicon === false) return 'change-chip--oov'
  return 'change-chip--unknown'
}

const readOnly = computed(() => page.value?.status === 'CONFIRMED')
const canRecognize = computed(() =>
  ['UPLOADED', 'RECOGNIZED', 'EDITING', 'CONFIRMED'].includes(page.value?.status)
)
const canConfirm = computed(() => ['RECOGNIZED', 'EDITING'].includes(page.value?.status))
const thresholds = computed(() => page.value?.confidence_thresholds || { warning: 0.9, critical: 0.7 })
const lexiconAvailable = computed(() => page.value?.lexicon_available === true)
const lexiconAvailableInList = computed(() => pages.value.some((p) => p.lexicon_available))
// worst pages first: that is where the dictionary has something to show
const sortedPages = computed(() => {
  if (!sortByOov.value) return pages.value
  return [...pages.value].sort(
    (a, b) => (b.oov_count ?? -1) - (a.oov_count ?? -1) || b.page_id - a.page_id
  )
})
const pageWidth = computed(() => page.value?.width || naturalSize.value.width || 1)
const pageHeight = computed(() => page.value?.height || naturalSize.value.height || 1)
const totalWords = computed(() => (page.value?.lines || []).reduce((sum, l) => sum + l.words.length, 0))
const hasStaleLines = computed(() => (page.value?.lines || []).some((l) => l.words_stale))
const contentStyle = computed(() => ({
  width: `${pageWidth.value}px`,
  height: `${pageHeight.value}px`,
  transform: `translate(${panX.value}px, ${panY.value}px) scale(${zoom.value})`,
}))
const dirtyCount = computed(() => {
  if (!page.value) return 0
  return page.value.lines.filter((l) => (drafts[l.id] ?? '') !== effectiveText(l)).length
})
const trainingMetrics = computed(() => {
  const m = training.value?.metrics
  return m && (m.validation || m.baseline)
})
const trainingTitle = computed(() => ({
  SUCCESS: 'Модель обучена и активирована',
  INSUFFICIENT_DATA: 'Данных пока недостаточно для обучения',
  NO_IMPROVEMENT: 'Модель не стала лучше — оставлена предыдущая',
  BUSY: 'Обучение уже идёт',
  FAILED: 'Обучение не удалось',
}[training.value?.outcome] || 'Результат обучения'))
const trainingBannerClass = computed(() => ({
  SUCCESS: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
  INSUFFICIENT_DATA: 'border-amber-400/40 bg-amber-400/10 text-amber-200',
  NO_IMPROVEMENT: 'border-amber-400/40 bg-amber-400/10 text-amber-200',
  BUSY: 'border-app-border/60 bg-app-panel/60 text-app-text',
  FAILED: 'border-app-error/40 bg-app-error/10 text-app-error',
}[training.value?.outcome] || 'border-app-border/60 bg-app-panel/60 text-app-text'))

const statusOf = (status) => STATUS[status] || { label: status, class: 'bg-app-input/60 text-app-muted' }
const effectiveText = (line) => line.corrected_text ?? line.predicted_text ?? ''

/** SVG points for a line/word: the stored outline, or its bbox as a fallback. */
const shapePoints = (item) => {
  const polygon = item.polygon
  if (Array.isArray(polygon) && polygon.length >= 3) {
    return polygon.map((point) => `${point[0]},${point[1]}`).join(' ')
  }
  const box = item.bbox
  return `${box.x1},${box.y1} ${box.x2},${box.y1} ${box.x2},${box.y2} ${box.x1},${box.y2}`
}

// the SVG lives inside the scaled wrapper, so keep strokes constant on screen
const lineStrokeWidth = computed(() => Math.min(1.1 / zoom.value, 20))
const wordStrokeWidth = computed(() => Math.min(1.5 / zoom.value, 28))

const legendStyle = (level) => ({
  background: CONFIDENCE[level].background,
  borderColor: CONFIDENCE[level].border,
})
const parentLineIsStale = (word) => Boolean(wordLineMap.value.get(word.id)?.words_stale)
const wordTitle = (word) =>
  `${word.effective_text ?? ''}${word.confidence != null ? ` — ${(word.confidence * 100).toFixed(0)}%` : ''}`

const formatConfidence = (value) => (value == null ? '—' : `${Math.round(value * 100)}%`)
const formatPercent = (value) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`)
const formatDate = (value) => {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString('ru-RU', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
  } catch {
    return '—'
  }
}
const confidenceBadgeClass = (value) => {
  if (value == null) return 'bg-app-input/60 text-app-muted'
  if (value >= thresholds.value.warning) return 'bg-emerald-500/15 text-emerald-300'
  if (value >= thresholds.value.critical) return 'bg-amber-400/20 text-amber-300'
  return 'bg-app-error/20 text-app-error'
}
const worstConfidence = (line) => {
  const values = line.words.map((w) => w.confidence).filter((v) => v != null)
  return values.length ? Math.min(...values) : null
}

/** The text of a miss, not a word box: stale boxes may hold an older reading. */
const onOovChipClick = async (line, miss) => {
  const word = line.words_stale
    ? null
    : line.words.find((w) => (w.effective_text || '').toLowerCase() === miss.toLowerCase())
  if (word) {
    await onWordClick(word)
    return
  }
  activeLineId.value = line.id
  await nextTick()
  const textarea = textareaRefs.get(line.id)
  textarea?.closest('article')?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  textarea?.focus({ preventScroll: true })
}

const setTextareaRef = (lineId, el) => {
  if (el) textareaRefs.set(lineId, el)
  else textareaRefs.delete(lineId)
}

// ---------------------------------------------------------------- data

const api = (path, init) => authFetch(apiBaseUrl, `${apiBaseUrl}${path}`, init)

const loadAuthors = async () => {
  try {
    const response = await api('/htr/authors')
    if (response.status === 401) {
      error.value = 'Войдите, чтобы работать с рукописями.'
      return
    }
    if (!response.ok) throw new Error('Не удалось загрузить авторов')
    authors.value = await response.json()
    const requested = Number(route.query.author)
    const preferred = authors.value.find((a) => a.id === requested) || authors.value[0]
    if (preferred) {
      authorId.value = preferred.id
      await loadPages()
    }
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки авторов'
  } finally {
  }
}

const createAuthor = async () => {
  creatingAuthor.value = true
  error.value = ''
  try {
    const response = await api('/htr/authors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newAuthorName.value.trim() }),
    })
    if (!response.ok) throw new Error('Не удалось создать автора')
    const created = await response.json()
    authors.value = [...authors.value, created]
    authorId.value = created.id
    newAuthorName.value = ''
    showAuthorForm.value = false
    pages.value = []
    page.value = null
  } catch (err) {
    error.value = err.message || 'Ошибка создания автора'
  } finally {
    creatingAuthor.value = false
  }
}

const loadPages = async (autoOpen = true) => {
  if (!authorId.value) return
  pagesLoading.value = true
  try {
    const response = await api(`/htr/authors/${authorId.value}/pages`)
    if (!response.ok) throw new Error('Не удалось загрузить список страниц')
    pages.value = await response.json()
    if (!autoOpen) return

    const requested = Number(route.query.page)
    const target =
      pages.value.find((p) => p.page_id === requested) ||
      pages.value.find((p) => p.page_id === page.value?.page_id) ||
      pages.value[0]
    if (target) await openPage(target.page_id)
    else {
      page.value = null
      releaseImage()
    }
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки страниц'
  } finally {
    pagesLoading.value = false
  }
}

const openPage = async (pageId) => {
  pageLoading.value = true
  clearMessages()
  imageError.value = ''
  try {
    const response = await api(`/htr/pages/${pageId}`)
    if (!response.ok) throw new Error('Не удалось загрузить страницу')
    page.value = await response.json()
    resetDrafts()
    activeLineId.value = null
    activeWordId.value = null
    await loadImage(pageId)
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки страницы'
  } finally {
    pageLoading.value = false
  }
}

const resetDrafts = () => {
  for (const key of Object.keys(drafts)) delete drafts[key]
  for (const line of page.value?.lines || []) drafts[line.id] = effectiveText(line)
}

const releaseImage = () => {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  imageUrl.value = ''
}

const loadImage = async (pageId) => {
  imageLoading.value = true
  releaseImage()
  try {
    const response = await api(`/htr/pages/${pageId}/image`)
    if (response.status === 404) {
      imageError.value = 'Файл изображения недоступен на сервере (страница могла быть перенесена без картинки).'
      return
    }
    if (!response.ok) throw new Error('Не удалось загрузить изображение')
    const blob = await response.blob()
    imageUrl.value = URL.createObjectURL(blob)
    const size = await readImageSize(imageUrl.value)
    naturalSize.value = size
    await nextTick()
    fitToWidth()
  } catch (err) {
    imageError.value = err.message || 'Ошибка загрузки изображения'
  } finally {
    imageLoading.value = false
  }
}

const readImageSize = (url) =>
  new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = () => resolve({ width: 0, height: 0 })
    img.src = url
  })

// ---------------------------------------------------------------- editing

const onDraftInput = (lineId, value) => {
  drafts[lineId] = value
}

const saveLine = async (line) => {
  const value = drafts[line.id] ?? ''
  if (value === effectiveText(line)) return
  savingLines.add(line.id)
  try {
    const response = await api(`/htr/pages/${page.value.page_id}/lines/${line.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corrected_text: value }),
    })
    if (!response.ok) throw new Error('Не удалось сохранить строку')
    const updated = await response.json()
    applyPageUpdate(updated)
  } catch (err) {
    error.value = err.message || 'Ошибка сохранения строки'
  } finally {
    savingLines.delete(line.id)
  }
}

const applyPageUpdate = (updated) => {
  if (!page.value || updated.page_id !== page.value.page_id) return
  const byId = new Map(updated.lines.map((l) => [l.id, l]))
  for (const line of page.value.lines) {
    const fresh = byId.get(line.id)
    if (!fresh) continue
    line.corrected_text = fresh.corrected_text
    line.corrected_by = fresh.corrected_by
    line.words_stale = fresh.words_stale
    line.oov_count = fresh.oov_count
    line.oov_words = fresh.oov_words
    line.suggested_text = fresh.suggested_text
    line.suggested_by = fresh.suggested_by
    line.suggestion_changes = fresh.suggestion_changes
    line.suggestion_verified = fresh.suggestion_verified
    // the dictionary check is recomputed server-side on every write
    const wordsById = new Map(fresh.words.map((w) => [w.id, w]))
    for (const word of line.words) {
      const freshWord = wordsById.get(word.id)
      if (freshWord) word.in_lexicon = freshWord.in_lexicon
    }
  }
  page.value.status = updated.status
  page.value.prediction_cer = updated.prediction_cer
  page.value.prediction_wer = updated.prediction_wer
  page.value.oov_count = updated.oov_count
  page.value.lexicon_available = updated.lexicon_available
}

const flushPendingSaves = async () => {
  if (!page.value) return
  for (const line of page.value.lines) {
    if ((drafts[line.id] ?? '') !== effectiveText(line)) await saveLine(line)
  }
}

// ---------------------------------------------------------------- actions

const deletePage = async (item) => {
  const message =
    item.status === 'CONFIRMED'
      ? `Удалить страницу #${item.page_id}? Она подтверждена и входит в обучающий набор — в следующих обучениях использоваться не будет. Уже обученные версии модели останутся.`
      : `Удалить страницу #${item.page_id}? Изображение и разметка будут удалены.`
  if (!confirm(message)) return

  deletingPageId.value = item.page_id
  error.value = ''
  try {
    const response = await api(`/htr/pages/${item.page_id}`, { method: 'DELETE' })
    // 404 means it is already gone, which is the state we wanted anyway
    if (!response.ok && response.status !== 404) {
      throw new Error('Не удалось удалить страницу')
    }
    const wasActive = page.value?.page_id === item.page_id
    if (wasActive) {
      page.value = null
      releaseImage()
      training.value = null
    }
    await loadPages(false)
    if (wasActive) {
      const next = pages.value[0]
      if (next) await selectPage(next.page_id)
      else {
        skipQueryWatch = true
        await router.replace({ query: { author: String(authorId.value) } })
        skipQueryWatch = false
      }
    }
  } catch (err) {
    error.value = err.message || 'Ошибка удаления страницы'
  } finally {
    deletingPageId.value = null
  }
}

const onAuthorChange = async () => {
  page.value = null
  releaseImage()
  training.value = null
  skipQueryWatch = true
  await router.replace({ query: { author: String(authorId.value) } })
  skipQueryWatch = false
  await loadPages()
}

const selectPage = async (pageId) => {
  training.value = null
  await flushPendingSaves()
  skipQueryWatch = true
  await router.replace({
    query: { author: String(authorId.value), page: String(pageId) },
  })
  skipQueryWatch = false
  await openPage(pageId)
}

const onUpload = async (event) => {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || !authorId.value) return
  uploading.value = true
  error.value = ''
  try {
    const form = new FormData()
    form.append('file', file)
    const response = await api(`/htr/authors/${authorId.value}/pages`, { method: 'POST', body: form })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось загрузить страницу')
    }
    const created = await response.json()
    await loadPages(false)
    await selectPage(created.page_id)
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки страницы'
  } finally {
    uploading.value = false
  }
}

const recognize = async () => {
  if (!page.value) return
  // re-segmenting an already fixed page drops its corrections / confirmation
  const force = ['EDITING', 'CONFIRMED'].includes(page.value.status)
  if (force) {
    const warned =
      page.value.status === 'CONFIRMED'
        ? 'Страница подтверждена и входит в обучающий набор.\n\n' +
          'Перераспознать? Разметка, правки и подтверждение будут потеряны — ' +
          'страница вернётся в состояние «Распознано», её нужно будет проверить ' +
          'и подтвердить заново.'
        : page.value.lines.some((line) => line.corrected_text !== null)
          ? 'Перераспознать страницу? Разметка и все правки этой страницы будут потеряны.'
          : 'Перераспознать страницу? Текущая разметка будет заменена.'
    if (!confirm(warned)) return
    await flushPendingSaves()
  }
  recognizing.value = true
  clearMessages()
  training.value = null
  try {
    const url = `/htr/pages/${page.value.page_id}/recognize${force ? '?force=true' : ''}`
    const response = await api(url, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Распознавание не удалось')
    }
    page.value = await response.json()
    resetDrafts()
    activeLineId.value = null
    activeWordId.value = null
    await loadImage(page.value.page_id)
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка распознавания'
  } finally {
    recognizing.value = false
  }
}

const clearMessages = () => {
  error.value = ''
  notice.value = ''
}

const suggestWithLlm = async () => {
  if (!page.value) return
  correcting.value = true
  clearMessages()
  training.value = null
  try {
    await flushPendingSaves()
    const response = await api(`/htr/pages/${page.value.page_id}/suggestions`, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось получить предложения')
    }
    page.value = await response.json()
    resetDrafts()
    await loadPages(false)
    const count = pendingSuggestionCount.value
    notice.value = count
      ? `Модель предложила правки в ${count} строках — распознанный текст не изменён, примите нужные.`
      : 'Модель не предложила правок (или недоступна) — текст оставлен как есть.'
  } catch (err) {
    error.value = err.message || 'Ошибка обращения к модели'
  } finally {
    correcting.value = false
  }
}

/** Apply a single proposed word; the rest of the proposal stays for review. */
const acceptChange = async (line, index) => {
  if (!page.value || readOnly.value) return
  busySuggestionLine.value = line.id
  clearMessages()
  try {
    const response = await api(
      `/htr/pages/${page.value.page_id}/lines/${line.id}/suggestion/changes/${index}`,
      { method: 'PUT' }
    )
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось применить слово')
    }
    const updated = await response.json()
    applyPageUpdate(updated)
    resetDrafts()
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка применения слова'
  } finally {
    busySuggestionLine.value = null
  }
}

const acceptSuggestion = async (line) => {
  if (!page.value) return
  busySuggestionLine.value = line.id
  clearMessages()
  try {
    const response = await api(`/htr/pages/${page.value.page_id}/lines/${line.id}/suggestion`, {
      method: 'PUT',
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось принять предложение')
    }
    const updated = await response.json()
    applyPageUpdate(updated)
    resetDrafts()
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка принятия предложения'
  } finally {
    busySuggestionLine.value = null
  }
}

const dismissSuggestion = async (line) => {
  if (!page.value) return
  busySuggestionLine.value = line.id
  clearMessages()
  try {
    const response = await api(`/htr/pages/${page.value.page_id}/lines/${line.id}/suggestion`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось скрыть предложение')
    }
    const updated = await response.json()
    applyPageUpdate(updated)
    resetDrafts()
  } catch (err) {
    error.value = err.message || 'Ошибка скрытия предложения'
  } finally {
    busySuggestionLine.value = null
  }
}

const acceptVerifiedSuggestions = async () => {
  if (!page.value) return
  const expected = verifiedSuggestionCount.value
  if (!expected) return
  if (
    !confirm(
      `Принять предложения в ${expected} строках?\n\n` +
        'Берутся только те строки, где каждое новое слово есть в словаре. ' +
        'В остальных строках останутся непроверенные правки — их нужно прочитать.'
    )
  ) {
    return
  }
  acceptingSuggestions.value = true
  clearMessages()
  try {
    const response = await api(`/htr/pages/${page.value.page_id}/suggestions/accept`, {
      method: 'POST',
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось принять предложения')
    }
    const payload = await response.json()
    if (payload.page) {
      page.value = payload.page
      resetDrafts()
    }
    await loadPages(false)
    notice.value = payload.accepted
      ? `Принято проверенных предложений: ${payload.accepted}.`
      : 'Полностью проверенных предложений нет — каждую строку нужно принять вручную.'
  } catch (err) {
    error.value = err.message || 'Ошибка принятия предложений'
  } finally {
    acceptingSuggestions.value = false
  }
}

const confirmPage = async () => {
  if (!page.value) return
  confirming.value = true
  clearMessages()
  training.value = null
  try {
    await flushPendingSaves()
    const response = await api(`/htr/pages/${page.value.page_id}/confirm`, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось подтвердить страницу')
    }
    const payload = await response.json()
    if (payload.page) {
      page.value = payload.page
      resetDrafts()
    }
    training.value = payload.training || null
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка подтверждения'
  } finally {
    confirming.value = false
  }
}

// ---------------------------------------------------------------- image navigation

const fitToWidth = () => {
  if (!viewport.value) return
  const width = viewport.value.clientWidth
  if (!width || !pageWidth.value) return
  zoom.value = width / pageWidth.value
  panX.value = 0
  panY.value = 0
}

const zoomTo = (value) => {
  if (!viewport.value) return
  const rect = viewport.value.getBoundingClientRect()
  applyZoom(value, rect.width / 2, rect.height / 2)
}

const zoomBy = (factor) => {
  if (!viewport.value) return
  const rect = viewport.value.getBoundingClientRect()
  applyZoom(zoom.value * factor, rect.width / 2, rect.height / 2)
}

const applyZoom = (value, cx, cy) => {
  const next = Math.min(Math.max(value, 0.05), 8)
  const contentX = (cx - panX.value) / zoom.value
  const contentY = (cy - panY.value) / zoom.value
  zoom.value = next
  panX.value = cx - contentX * next
  panY.value = cy - contentY * next
}

const onWheel = (event) => {
  if (!event.ctrlKey && !event.metaKey) return
  event.preventDefault()
  const rect = viewport.value.getBoundingClientRect()
  const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15
  applyZoom(zoom.value * factor, event.clientX - rect.left, event.clientY - rect.top)
}

const onPointerDown = (event) => {
  if (event.button !== 0) return
  if (event.target.closest('[data-word]')) return
  panning.value = true
  panStart = { x: event.clientX, y: event.clientY, panX: panX.value, panY: panY.value }
}

const onPointerMove = (event) => {
  if (!panning.value || !panStart) return
  panX.value = panStart.panX + (event.clientX - panStart.x)
  panY.value = panStart.panY + (event.clientY - panStart.y)
}

const onPointerUp = () => {
  panning.value = false
  panStart = null
}

// ---------------------------------------------------------------- word <-> text

const onWordClick = async (word) => {
  const line = wordLineMap.value.get(word.id)
  if (!line) return
  activeLineId.value = line.id
  activeWordId.value = word.id
  await nextTick()
  const textarea = textareaRefs.get(line.id)
  if (!textarea) return
  textarea.closest('article')?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  textarea.focus({ preventScroll: true })
  if (!line.words_stale && !readOnly.value) {
    const span = tokenSpan(drafts[line.id] ?? '', word.order)
    if (span) textarea.setSelectionRange(span[0], span[1])
  }
}

const tokenSpan = (text, index) => {
  const spans = []
  const re = /\S+/g
  let match
  while ((match = re.exec(text)) !== null) spans.push([match.index, match.index + match[0].length])
  return spans[index] || null
}

// ---------------------------------------------------------------- lifecycle

onMounted(async () => {
  await loadAuthors()
  if (!authors.value.length) showAuthorForm.value = true
})

onBeforeUnmount(() => {
  releaseImage()
})

watch(
  () => route.query.page,
  async (value) => {
    if (skipQueryWatch) return
    const pageId = Number(value)
    if (!pageId || !authorId.value || page.value?.page_id === pageId) return
    if (!pages.value.some((p) => p.page_id === pageId)) return
    await openPage(pageId)
  }
)
</script>

<style scoped>
.app-select,
.app-input {
  border-radius: 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: rgba(2, 6, 23, 0.6);
  color: #f8fafc;
  padding: 0.45rem 0.75rem;
  font-size: 0.875rem;
}

.app-primary,
.app-success,
.app-ghost,
.app-icon {
  border-radius: 9999px;
  border: 1px solid transparent;
  font-weight: 600;
  transition: background 0.15s ease, opacity 0.15s ease;
}

.app-primary {
  background: rgba(99, 102, 241, 0.95);
  color: #fff;
  padding: 0.45rem 1.1rem;
  font-size: 0.875rem;
}

.app-primary:hover:not(:disabled) {
  background: rgba(99, 102, 241, 0.8);
}

.app-success {
  background: rgba(16, 185, 129, 0.9);
  color: #052e22;
  padding: 0.45rem 1.1rem;
  font-size: 0.875rem;
}

.app-success:hover:not(:disabled) {
  background: rgba(16, 185, 129, 0.75);
}

.app-ghost {
  background: rgba(15, 23, 42, 0.5);
  border-color: rgba(148, 163, 184, 0.4);
  color: #e2e8f0;
  padding: 0.45rem 0.9rem;
  font-size: 0.8rem;
}

.app-ghost:hover:not(:disabled) {
  background: rgba(30, 41, 59, 0.75);
}

.app-icon {
  width: 2rem;
  height: 2rem;
  background: rgba(15, 23, 42, 0.5);
  border-color: rgba(148, 163, 184, 0.4);
  color: #e2e8f0;
  line-height: 1;
}

.app-icon:hover {
  background: rgba(30, 41, 59, 0.75);
}

.page-delete {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.5rem;
  padding: 0.15rem 0.25rem;
  color: #94a3b8;
  background: transparent;
  transition: background 0.15s ease, color 0.15s ease;
}

.page-delete:hover:not(:disabled) {
  background: rgba(248, 113, 113, 0.2);
  color: #f87171;
}

.page-delete:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.app-primary:disabled,
.app-success:disabled,
.app-ghost:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.app-textarea {
  width: 100%;
  resize: vertical;
  border-radius: 0.65rem;
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: rgba(2, 6, 23, 0.6);
  color: #f8fafc;
  padding: 0.5rem 0.65rem;
  font-size: 0.9rem;
  line-height: 1.45;
}

.app-textarea:focus {
  outline: none;
  border-color: rgba(99, 102, 241, 0.8);
}

.app-textarea:disabled {
  opacity: 0.85;
  border-style: dashed;
}

.legend-box {
  display: inline-block;
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 0.2rem;
  border: 1px solid;
}

.legend-box--oov {
  border-color: #a78bfa;
  background: rgba(167, 139, 250, 0.32);
}

.app-mini {
  border-radius: 9999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  padding: 0.1rem 0.6rem;
  font-size: 0.7rem;
  transition: background 0.12s ease;
}

.app-mini:hover:not(:disabled) {
  background: rgba(30, 41, 59, 0.85);
}

.app-mini--ok {
  border-color: rgba(16, 185, 129, 0.6);
  background: rgba(16, 185, 129, 0.18);
  color: #a7f3d0;
}

.app-mini--ok:hover:not(:disabled) {
  background: rgba(16, 185, 129, 0.32);
}

.app-mini:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.app-ghost--verify {
  border-color: rgba(16, 185, 129, 0.45);
}

.change-chip {
  display: inline-flex;
  align-items: center;
  border-radius: 0.4rem;
  border: 1px solid;
  padding: 0.05rem 0.4rem;
  font-size: 0.72rem;
  line-height: 1.35;
  cursor: pointer;
  transition: filter 0.12s ease, background 0.12s ease;
}

.change-chip:hover:not(:disabled) {
  filter: brightness(1.35);
}

.change-chip:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.change-chip--locked {
  opacity: 0.5;
}

.change-chip--ok {
  border-color: rgba(16, 185, 129, 0.5);
  background: rgba(16, 185, 129, 0.12);
  color: #a7f3d0;
}

.change-chip--oov {
  border-color: rgba(167, 139, 250, 0.6);
  background: rgba(167, 139, 250, 0.16);
  color: #ddd6fe;
}

.change-chip--unknown {
  border-color: rgba(148, 163, 184, 0.45);
  background: rgba(148, 163, 184, 0.12);
  color: #cbd5e1;
}

.oov-chip {
  border-radius: 9999px;
  border: 1px solid rgba(167, 139, 250, 0.5);
  background: rgba(167, 139, 250, 0.16);
  color: #ddd6fe;
  padding: 0.05rem 0.5rem;
  font-size: 0.7rem;
  line-height: 1.3;
  transition: background 0.12s ease;
}

.oov-chip:hover {
  background: rgba(167, 139, 250, 0.35);
}

.page-sort {
  border-radius: 9999px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  background: rgba(15, 23, 42, 0.5);
  color: #cbd5e1;
  padding: 0.1rem 0.5rem;
  font-size: 0.65rem;
  transition: background 0.12s ease, color 0.12s ease;
}

.page-sort:hover:not(:disabled) {
  background: rgba(30, 41, 59, 0.8);
}

.page-sort--on {
  border-color: rgba(167, 139, 250, 0.7);
  background: rgba(167, 139, 250, 0.2);
  color: #ddd6fe;
}

.page-sort:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* --- SVG overlay: polygons follow the text, not the page axes ------------- */

.line-poly {
  fill: none;
  stroke: rgba(148, 163, 184, 0.45);
  pointer-events: none;
}

.line-poly--active {
  stroke: rgba(165, 180, 252, 0.95);
}

.line-poly--stale {
  stroke-dasharray: 8 6;
  opacity: 0.55;
}

.word-poly {
  cursor: pointer;
  transition: fill 0.12s ease;
  /* the scan is light paper: a dark halo keeps the outline readable on it */
  filter: drop-shadow(0 0 1px rgba(2, 6, 23, 0.8));
}

/* confidence keeps amber/red (fill + outline) ... */
.word-poly--normal {
  fill: rgba(148, 163, 184, 0.08);
  stroke: rgba(71, 85, 105, 0.75);
}

.word-poly--normal:hover {
  fill: rgba(226, 232, 240, 0.25);
  stroke: #e2e8f0;
}

.word-poly--warning {
  fill: rgba(245, 158, 11, 0.22);
  stroke: #f59e0b;
}

.word-poly--warning:hover {
  fill: rgba(245, 158, 11, 0.42);
}

.word-poly--critical {
  fill: rgba(239, 68, 68, 0.28);
  stroke: #ef4444;
}

.word-poly--critical:hover {
  fill: rgba(239, 68, 68, 0.5);
}

/*
 * ... while "not in the dictionary" gets a colour channel of its own: a violet
 * fill, so an amber or red outline stays readable on the same word. The
 * two-class selectors win over the confidence fills without !important.
 */
.word-poly.word-poly--oov {
  fill: rgba(167, 139, 250, 0.32);
}

.word-poly.word-poly--oov:hover {
  fill: rgba(167, 139, 250, 0.5);
}

/* a confident-but-unknown word needs a violet outline as well, otherwise a
   faintly violet box between grey ones is easy to miss */
.word-poly--oov.word-poly--normal {
  stroke: #a78bfa;
}

/* the selected word is marked on top of everything, in a colour of its own */
.word-poly.word-poly--active {
  stroke: #f8fafc;
  stroke-width: 2px;
  filter: drop-shadow(0 0 3px rgba(248, 250, 252, 0.95));
}

.word-poly.word-poly--active:not(.word-poly--oov) {
  fill: rgba(99, 102, 241, 0.4);
}

.word-poly--stale {
  opacity: 0.35;
  stroke-dasharray: 6 5;
}
</style>
