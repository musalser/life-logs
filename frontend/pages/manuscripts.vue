<template>
  <div class="flex flex-col gap-4">
    <!-- header -->
    <!--
      One row when the window allows it: the title, the author everything below
      works in, and the actions. The long explanation moved into the title's
      tooltip and the "who am I working as" controls lost their shouting label,
      because the header is a toolbar, not a landing page.
    -->
    <header class="rounded-2xl border border-app-border/40 bg-app-panel/50 px-3 py-2">
      <div class="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1
          class="flex-none text-lg font-semibold"
          title="Загрузите страницу, распознайте её, проверьте текст и подтвердите — подтверждённые страницы становятся обучающим набором персональной модели"
        >
          Рукописи
        </h1>

        <div v-if="authors.length" class="flex min-w-0 items-center gap-1.5">
          <span class="flex-none text-[11px] text-app-muted">автор</span>
          <select
            v-model.number="authorId"
            class="app-select app-select--compact"
            @change="onAuthorChange"
          >
            <option v-for="a in authors" :key="a.id" :value="a.id">{{ a.name }}</option>
          </select>
          <button
            type="button"
            class="app-ghost app-ghost--compact"
            :title="showAuthorForm ? 'Скрыть форму нового автора' : 'Добавить автора'"
            @click="showAuthorForm = !showAuthorForm"
          >
            {{ showAuthorForm ? 'Отмена' : '+ автор' }}
          </button>
          <button
            type="button"
            class="app-ghost app-ghost--compact"
            :disabled="!authorId"
            title="Слова, которые ваши подтверждённые страницы добавили в словарь: их можно убрать"
            @click="openLexicon"
          >
            Словарь автора
          </button>
        </div>

        <div class="ml-auto flex flex-wrap items-center justify-end gap-1.5">
          <button
            type="button"
            class="app-ghost app-ghost--train"
            :disabled="!authorId || trainingStarting"
            title="Дообучить персональную модель на всех подтверждённых страницах автора"
            @click="trainAuthor"
          >
            {{ trainingStarting ? 'Обучение…' : 'Обучить модель' }}
          </button>
          <button
            type="button"
            class="app-ghost"
            :disabled="!authorId || uploading"
            title="Загрузить все изображения из папки — имена файлов станут названиями страниц"
            @click="folderInput?.click()"
          >
            Загрузить папку
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
          <input
            ref="folderInput"
            type="file"
            webkitdirectory
            directory
            multiple
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

    <!-- outcome of the explicit «Обучить модель» run -->
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
          <span class="text-xs text-app-muted">{{ pages.length }} шт.</span>
        </div>

        <p v-if="pagesLoading" class="text-xs text-app-muted">Загрузка…</p>
        <p v-else-if="!pages.length" class="rounded-xl border border-dashed border-app-border/40 px-3 py-3 text-xs text-app-muted">
          Страниц пока нет.
        </p>

        <div v-else class="no-scrollbar max-h-[70vh] space-y-2 overflow-y-auto pr-1">
          <div
            v-for="item in pages"
            :key="item.page_id"
            role="button"
            tabindex="0"
            :draggable="canReorder"
            class="page-item w-full cursor-pointer rounded-xl border px-3 py-2 text-left transition"
            :class="[
              page?.page_id === item.page_id
                ? 'border-app-primary/70 bg-app-primary/10'
                : 'border-app-border/40 bg-app-panel/30 hover:border-app-primary/40 hover:bg-app-panel/50',
              dragPageId === item.page_id ? 'page-item--dragging' : '',
              dropTarget && dropTarget.id === item.page_id
                ? (dropTarget.after ? 'page-item--drop-after' : 'page-item--drop-before')
                : '',
            ]"
            @click="selectPage(item.page_id)"
            @keydown.enter.prevent="selectPage(item.page_id)"
            @dragstart="onPageDragStart(item, $event)"
            @dragover.prevent="onPageDragOver(item, $event)"
            @drop.prevent="onPageDrop(item)"
            @dragend="onPageDragEnd"
          >
            <div class="flex items-start gap-1.5">
              <span
                v-if="canReorder"
                class="page-grip mt-0.5"
                title="Перетащите, чтобы изменить порядок"
                aria-hidden="true"
              >
                <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5">
                  <path d="M8 6h2v2H8V6zm6 0h2v2h-2V6zM8 11h2v2H8v-2zm6 0h2v2h-2v-2zM8 16h2v2H8v-2zm6 0h2v2h-2v-2z" />
                </svg>
              </span>
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-1">
                  <p class="min-w-0 flex-1 truncate text-sm font-medium" :title="pageOrigin(item)">
                    {{ pageLabel(item) }}
                  </p>
                  <!-- closed padlock on a confirmed page: clicking it reopens -->
                  <button
                    v-if="item.status === 'CONFIRMED'"
                    type="button"
                    class="page-icon text-emerald-300"
                    title="Страница подтверждена (эталон обучения). Вернуть в редактирование"
                    :disabled="reopeningPageId === item.page_id"
                    @click.stop="reopenPageFromList(item)"
                  >
                    <LockIcon :closed="true" class="h-3.5 w-3.5" />
                  </button>
                  <span
                    v-else
                    class="page-icon text-app-muted opacity-60"
                    title="Страница не подтверждена: правки доступны"
                  >
                    <LockIcon :closed="false" class="h-3.5 w-3.5" />
                  </span>
                  <button
                    type="button"
                    class="page-icon"
                    title="Переименовать страницу"
                    :disabled="renamingPageId === item.page_id"
                    @click.stop="renamePage(item)"
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5">
                      <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"></path>
                    </svg>
                  </button>
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
            </div>
          </div>
        </div>
      </aside>

      <!-- image + overlay -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-3">
        <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <p class="text-xs uppercase tracking-widest text-app-accent">Изображение</p>
            <span
              v-if="readOnly"
              class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300"
              title="Страница подтверждена: разметка показана приглушённо, текст больше не редактируется"
            >
              подтверждена · разметка приглушена
            </span>
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
            <!--
              One control for both directions: an open padlock confirms the page,
              a closed one returns it to editing (with the consequences spelled
              out in the dialog). Confirming is not a one-way door any more.
            -->
            <button
              type="button"
              :class="readOnly ? 'app-ghost' : 'app-success'"
              :disabled="(readOnly ? reopening : !canConfirm) || confirming || reopening"
              :title="readOnly
                ? 'Страница подтверждена (эталон обучения). Нажмите, чтобы вернуть её в редактирование'
                : 'Подтвердить страницу как эталон: разметка станет приглушённой, страница войдёт в обучающую выборку'"
              @click="readOnly ? reopenPage() : confirmPage()"
            >
              <LockIcon :closed="readOnly" class="mr-1 inline-block h-3.5 w-3.5 align-[-2px]" />
              {{
                confirming
                  ? 'Подтверждение…'
                  : reopening
                    ? 'Возврат…'
                    : readOnly
                      ? 'Вернуть в редактирование'
                      : 'Подтвердить страницу'
              }}
            </button>
          </div>
        </div>

        <div
          class="mb-2 flex flex-wrap items-center gap-3 text-[11px] text-app-muted transition-opacity"
          :class="readOnly ? 'opacity-40' : ''"
        >
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
          <span v-if="lexiconAvailable" class="flex items-center gap-1">
            <span class="legend-box legend-box--author"></span>
            слово автора
          </span>
          <span v-else-if="page?.lines.length" class="text-violet-300/80">
            словарь не установлен — слова не проверяются
          </span>
          <span v-if="hasStaleLines" class="flex items-center gap-1 text-amber-300">
            пунктир — правка изменила число слов, боксы больше не совпадают
          </span>
        </div>

        <div
          ref="viewport"
          class="group relative h-[calc(100vh_-_320px)] min-h-[420px] select-none overflow-hidden rounded-xl border border-app-border/40 bg-app-input/60"
          :class="panning ? 'cursor-grabbing' : 'cursor-grab'"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
          @pointerleave="onPointerUp"
          @wheel.prevent="onWheel"
          @dblclick="fitToWindow(true)"
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
            class="absolute left-0 top-0 origin-top-left overflow-hidden rounded-xl"
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
              :class="{ 'overlay--muted': readOnly }"
              :style="{ '--line-stroke': lineStrokeWidth, '--word-stroke': wordStrokeWidth }"
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
                  word.author_only ? 'word-poly--author' : '',
                  activeWordId === word.id ? 'word-poly--active' : '',
                  parentLineIsStale(word) ? 'word-poly--stale' : '',
                ]"
                @click.stop="onWordClick(word)"
              >
                <title>{{ wordTitle(word) }}</title>
              </polygon>
            </svg>
          </div>

          <!-- viewer chrome: zoom level, and a hint that only appears on hover -->
          <span
            v-if="page && imageUrl && !imageLoading"
            class="pointer-events-none absolute right-2 top-2 rounded-full bg-black/55 px-2 py-0.5 text-[11px] font-semibold text-app-muted backdrop-blur transition-opacity duration-300"
            :class="zoomBadge ? 'opacity-100' : 'opacity-0'"
          >
            {{ zoomPercent }}%
          </span>
          <p
            v-if="page && imageUrl && !imageLoading"
            class="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-black/50 px-2.5 py-0.5 text-[10px] text-app-muted opacity-0 backdrop-blur transition-opacity duration-200 group-hover:opacity-100"
          >
            колесо — масштаб · перетаскивание — сдвиг · двойной клик — вписать по ширине
          </p>
        </div>

        <!-- page metadata moved out of the sidebar: it describes the open page -->
        <div
          v-if="page"
          class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-app-muted"
        >
          <span class="font-semibold text-app-text">#{{ page.page_id }}</span>
          <span>{{ page.lines.length }} строк</span>
          <span>загружено {{ formatDate(page.created_at) }}</span>
          <span v-if="page.prediction_cer != null">
            расхождение: CER {{ formatPercent(page.prediction_cer) }}
          </span>
          <span>{{ pageWidth }}×{{ pageHeight }} px</span>
          <span>слов {{ totalWords }}</span>
          <span v-if="page.recognition_model_version_id">
            модель #{{ page.recognition_model_version_id }}
          </span>
        </div>
      </section>

      <!-- transcription -->
      <section class="rounded-2xl border border-app-border/40 bg-app-panel/50 p-3">
        <div class="mb-2 flex items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <p class="text-xs uppercase tracking-widest text-app-accent">Транскрипция</p>
            <span
              v-if="page?.lines.length"
              class="rounded-full bg-app-input/60 px-2 py-0.5 text-[10px] font-semibold text-app-muted"
              title="Строк на странице"
            >
              {{ page.lines.length }}
            </span>
          </div>
          <div class="flex items-center gap-1.5">
            <span v-if="dirtyCount" class="text-[11px] text-amber-300">не сохранено: {{ dirtyCount }}</span>
            <span v-else-if="page" class="text-[11px] text-app-muted">сохранено</span>
            <button
              v-if="page?.lines.length"
              type="button"
              class="page-icon"
              :disabled="copyingAll"
              :title="copyingAll ? 'Копирование…' : 'Скопировать весь текст в буфер обмена'"
              @click.stop="copyAllText"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="h-4 w-4"
              >
                <rect x="9" y="9" width="11" height="11" rx="2"></rect>
                <path d="M5 15V6a2 2 0 0 1 2-2h9"></path>
              </svg>
            </button>
          </div>
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
            class="line-card rounded-xl border px-2 py-1.5 transition"
            :class="[
              activeLineId === line.id
                ? 'line-card--active border-app-primary/70 bg-app-primary/5'
                : 'border-app-border/40 bg-app-panel/30',
              line.words.length ? confidenceEdgeClass(worstConfidence(line)) : '',
            ]"
            @click="activeLineId = line.id"
          >
            <!--
              No header row any more: the confidence is the coloured left edge of
              the card, the actions sit right of the box, and the rest of the row
              is given to the transcription. Only states that ask for an action
              (a pending proposal, a stale markup, a save in flight) ever add a
              second line to a card.
            -->
            <div class="flex items-start gap-1">
              <div
                class="line-editor min-w-0 flex-1"
                :class="readOnly ? 'line-editor--locked' : ''"
                @mousemove="onEditorMousemove($event)"
                @mouseleave="hoverTooltip = null"
              >
                <div class="line-editor__layer" v-html="lineHighlightHtml(line)"></div>
                <textarea
                  :ref="(el) => setTextareaRef(line.id, el)"
                  :value="drafts[line.id]"
                  :disabled="readOnly"
                  rows="1"
                  spellcheck="false"
                  class="line-editor__input"
                  :placeholder="line.predicted_text ? '' : 'пустая строка'"
                  @input="onDraftInput(line.id, $event.target.value); autoGrow($event.target)"
                  @focus="activeLineId = line.id"
                  @blur="saveLine(line)"
                  @click="onTextareaClick(line, $event)"
                ></textarea>
              </div>

              <div class="line-card__tools flex flex-none flex-col items-center gap-0.5 pt-1">
                <button
                  type="button"
                  class="line-action"
                  :disabled="copyingLineId === line.id"
                  title="Скопировать строку"
                  @click.stop="copyLine(line)"
                >
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.8"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class="h-3.5 w-3.5"
                  >
                    <rect x="9" y="9" width="11" height="11" rx="2"></rect>
                    <path d="M5 15V6a2 2 0 0 1 2-2h9"></path>
                  </svg>
                </button>
                <button
                  v-if="!readOnly"
                  type="button"
                  class="line-action line-action--danger"
                  :disabled="deletingLineId === line.id"
                  title="Удалить строку"
                  @click.stop="deleteLine(line)"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor" class="h-3.5 w-3.5">
                    <path d="M6 2h12l2 2v2H2V4l2-2zm2 6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H8zM10 10h4v10h-4V10z"></path>
                  </svg>
                </button>
              </div>
            </div>

            <div
              v-if="hasSuggestion(line) || line.words_stale || savingLines.has(line.id)"
              class="mt-1 flex flex-wrap items-center gap-1.5"
            >
              <span
                v-if="hasSuggestion(line)"
                class="line-chip bg-app-primary/20 text-app-accent"
                title="У модели есть предложение по этой строке"
              >
                <span class="line-chip__dot"></span>
                предложение
              </span>
              <span
                v-if="line.words_stale"
                class="line-chip bg-amber-400/20 text-amber-300"
                title="Текст изменён, поэтому рамки слов больше не соответствуют разбиению на слова"
              >
                <span class="line-chip__dot"></span>
                разметка устарела
              </span>
              <span v-if="savingLines.has(line.id)" class="text-[10px] text-app-muted">сохранение…</span>
            </div>

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

            <!--
              Other readings the decoder considered for the clicked word. No
              caption: the word is selected in the text and highlighted on the
              scan, so the chips can only be the variants themselves.
            -->
            <div
              v-if="activeWordAlternatives(line).length"
              class="mt-1 flex flex-wrap items-center gap-1 rounded-lg border border-app-primary/25 bg-app-primary/5 px-1.5 py-1"
              title="Варианты, которые рассматривал декодер — нажмите, чтобы подставить"
            >
              <button
                v-for="alternative in activeWordAlternatives(line)"
                :key="`alt-${line.id}-${alternative.text}`"
                type="button"
                class="alt-chip"
                :disabled="readOnly"
                :title="`Оценка строки: ${Number(alternative.score ?? 0).toFixed(1)}`"
                @click.stop="applyAlternative(line, alternative)"
              >
                {{ alternative.text }}
              </button>
              <button
                type="button"
                class="line-action ml-auto"
                title="Скрыть варианты"
                @click.stop="activeWordId = null"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" class="h-3 w-3">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
          </article>
        </div>
      </section>
    </div>

    <!--
      Confirming is what puts a page's words into the author's dictionary, so the
      words are shown *before* the write: a word that would get in by mistake is
      much easier to fix here than to meet later as an unexpected "known". Each
      word opens the line it stands in — the dialog closes then, because it would
      cover that very line.
    -->
    <div
      v-if="confirmPreview"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      @click.self="confirmPreview = null"
    >
      <div class="flex max-h-[80vh] w-full max-w-xl flex-col rounded-2xl border border-app-border/60 bg-app-panel p-4 shadow-2xl">
        <div class="mb-2 flex items-start justify-between gap-3">
          <div>
            <p class="text-sm font-semibold">Подтвердить страницу?</p>
            <p class="text-[11px] text-app-muted">
              Страница станет эталоном обучения, а её слова войдут в словарь автора.
            </p>
          </div>
          <button type="button" class="page-icon" title="Закрыть" @click="confirmPreview = null">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" class="h-4 w-4">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        <p v-if="confirmPreviewLoading" class="py-6 text-center text-sm text-app-muted">
          Проверяем слова…
        </p>
        <template v-else-if="page && page.lexicon_available === false">
          <p class="rounded-xl border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-[11px] text-amber-200">
            Общий словарь не установлен, поэтому заранее показать, какие слова станут новыми,
            нельзя. При подтверждении слова страницы всё равно войдут в словарь автора.
          </p>
        </template>
        <template v-else>
          <p class="text-[11px] text-app-muted">
            В словарь автора будет добавлено
            <b>{{ confirmPreview.added.length }}</b>
            {{ confirmPreview.added.length === 1 ? 'слово' : 'слов' }}<template
              v-if="confirmPreview.learned.length"
            >, из них новых для общего словаря: <b>{{ confirmPreview.learned.length }}</b></template>.
          </p>

          <template v-if="confirmPreview.learned.length">
            <p class="mt-2 text-[11px] text-app-muted">
              Нажмите на слово, чтобы открыть строку, где оно написано, и исправить
              (окно закроется):
            </p>
            <div class="no-scrollbar mt-1 flex flex-wrap gap-1 overflow-y-auto">
              <button
                v-for="word in confirmPreview.learned.slice(0, 40)"
                :key="word"
                type="button"
                class="rounded-full border border-emerald-300/40 bg-emerald-400/10 px-2 py-0.5 text-[11px] hover:bg-emerald-400/20"
                :title="`Показать «${word}» в тексте`"
                @click="showPreviewWord(word)"
              >
                {{ word }}
              </button>
            </div>
            <p v-if="confirmPreview.learned.length > 40" class="mt-1 text-[11px] text-app-muted">
              …и ещё {{ confirmPreview.learned.length - 40 }} — весь список появится в «Словаре автора».
            </p>
          </template>
          <p v-else class="mt-2 text-[11px] text-app-muted">
            Все слова этой страницы общий словарь уже знает — подсветка не изменится.
          </p>
        </template>

        <div class="mt-4 flex items-center justify-end gap-2">
          <button type="button" class="app-ghost" @click="confirmPreview = null">Отмена</button>
          <button
            type="button"
            class="app-success"
            :disabled="confirming || confirmPreviewLoading"
            @click="applyConfirmation"
          >
            <LockIcon :closed="false" class="mr-1 inline-block h-3.5 w-3.5 align-[-2px]" />
            {{ confirming ? 'Подтверждение…' : 'Подтвердить страницу' }}
          </button>
        </div>
      </div>
    </div>

    <!--
      Returning a confirmed page to editing has consequences, so they are listed
      in a dialog rather than in a native confirm() with a wall of text.
    -->
    <div
      v-if="reopenPrompt"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      @click.self="reopenPrompt = null"
    >
      <div class="w-full max-w-md rounded-2xl border border-app-border/60 bg-app-panel p-4 shadow-2xl">
        <div class="mb-2 flex items-start justify-between gap-3">
          <p class="text-sm font-semibold">Вернуть страницу в редактирование?</p>
          <button type="button" class="page-icon" title="Закрыть" @click="reopenPrompt = null">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" class="h-4 w-4">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
        <ul class="list-disc space-y-1 pl-5 text-[11px] text-app-muted">
          <li v-for="consequence in REOPEN_CONSEQUENCES" :key="consequence">{{ consequence }}</li>
        </ul>
        <div class="mt-4 flex items-center justify-end gap-2">
          <button type="button" class="app-ghost" @click="reopenPrompt = null">Отмена</button>
          <button type="button" class="app-primary" :disabled="reopening" @click="applyReopen">
            {{ reopening ? 'Возврат…' : 'Вернуть в редактирование' }}
          </button>
        </div>
      </div>
    </div>

    <!-- the author's own vocabulary: what their confirmed pages taught the check -->
    <div
      v-if="lexiconOpen"
      class="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-4 backdrop-blur-sm"
      @click.self="lexiconOpen = false"
    >
      <div class="mt-10 flex max-h-[80vh] w-full max-w-xl flex-col rounded-2xl border border-app-border/60 bg-app-panel p-4 shadow-2xl">
        <div class="mb-2 flex items-start justify-between gap-3">
          <div>
            <p class="text-sm font-semibold">Словарь автора</p>
            <p class="text-[11px] text-app-muted">
              Слова, которых нет в общем словаре русского языка, но которые
              встречаются в ваших подтверждённых страницах. Нажмите на слово,
              чтобы найти его в тексте.
            </p>
          </div>
          <button type="button" class="page-icon" title="Закрыть" @click="lexiconOpen = false">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" class="h-4 w-4">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        <input v-model="lexiconQuery" class="app-input mb-2" placeholder="Поиск слова" />

        <p v-if="lexiconLoading" class="py-6 text-center text-sm text-app-muted">Загрузка…</p>
        <p
          v-else-if="!lexiconWords.length"
          class="rounded-xl border border-dashed border-app-border/40 px-3 py-6 text-center text-sm text-app-muted"
        >
          Таких слов нет: всё, что добавили подтверждённые страницы, есть в общем словаре.
        </p>
        <template v-else>
          <p
            v-if="!filteredLexiconWords.length"
            class="rounded-xl border border-dashed border-app-border/40 px-3 py-4 text-center text-xs text-app-muted"
          >
            Ничего не найдено.
          </p>
          <div v-else class="no-scrollbar -mr-1 flex-1 space-y-1 overflow-y-auto pr-1">
            <div
              v-for="item in filteredLexiconWords"
              :key="item.word"
              class="flex items-center justify-between gap-2 rounded-lg border border-app-border/30 bg-app-panel/40 px-2 py-1"
            >
              <button
                type="button"
                class="min-w-0 flex-1 truncate text-left text-sm hover:text-app-accent disabled:hover:text-inherit"
                :disabled="!item.occurrences || !item.occurrences.length"
                :title="item.occurrences && item.occurrences.length
                  ? `Показать «${item.word}» в тексте (мест: ${item.occurrences.length})`
                  : 'Место не найдено: слово могло остаться от удалённой правки'"
                @click="showLexiconWordInText(item)"
              >
                {{ item.word }}
              </button>
              <span
                v-if="item.count"
                class="flex-none text-[11px] text-app-muted"
                :title="`Встречается на подтверждённых страницах: ${item.count}`"
              >
                {{ item.count }}×
              </span>
            </div>
          </div>
        </template>

      </div>
    </div>

    <!--
      The coloured words of the transcription used to rely on the native `title`
      tooltip, which forced them to take pointer events -- and that swallowed the
      click that should have placed the caret. They are click-through now, so a
      single tooltip element paints the same explanation under the pointer.
    -->
    <div
      v-if="hoverTooltip"
      class="word-tooltip"
      :style="{ left: `${hoverTooltip.x}px`, top: `${hoverTooltip.y}px` }"
    >
      {{ hoverTooltip.text }}
    </div>
  </div>
</template>

<script setup>
import { authFetch } from '~/utils/auth-fetch'

const { public: { apiBaseUrl } } = useRuntimeConfig()
const route = useRoute()
const router = useRouter()

const CONFIDENCE = {
  normal: { border: 'rgba(100, 116, 139, 0.65)', background: 'rgba(148, 163, 184, 0.08)' },
  warning: { border: '#f59e0b', background: 'rgba(245, 158, 11, 0.22)' },
  critical: { border: '#ef4444', background: 'rgba(239, 68, 68, 0.28)' },
}

const authors = ref([])
const authorId = ref(null)
const pages = ref([])
const deletingPageId = ref(null)
const renamingPageId = ref(null)
const page = ref(null)
const drafts = reactive({})
const savingLines = reactive(new Set())
const deletingLineId = ref(null)
const copyingLineId = ref(null)
const copyingAll = ref(false)

// the author dictionary panel
const lexiconOpen = ref(false)
const lexiconLoading = ref(false)
const lexiconWords = ref([])
const lexiconQuery = ref('')

// drag-and-drop ordering of the sidebar list
const dragPageId = ref(null)
const dropTarget = ref(null) // { id, after }

const fileInput = ref(null)
const folderInput = ref(null)
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
const trainingStarting = ref(false)

const showAuthorForm = ref(false)
const newAuthorName = ref('')
const creatingAuthor = ref(false)

const pagesLoading = ref(false)
const pageLoading = ref(false)
const imageLoading = ref(false)
const uploading = ref(false)
const recognizing = ref(false)
const confirming = ref(false)
//: the confirmation dialog: what the page would teach the dictionary before the
//: write. Null while it is closed.
const confirmPreview = ref(null)
const confirmPreviewLoading = ref(false)
//: the reopen dialog: { pageId } while it is open, null otherwise
const reopenPrompt = ref(null)
const reopening = ref(false)
const reopeningPageId = ref(null)

const activeLineId = ref(null)
const activeWordId = ref(null)
// explanation of a coloured word, shown next to the pointer while hovering it
const hoverTooltip = ref(null)
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

// ------------------------------------------------------- sidebar name & order

const canReorder = computed(() => pages.value.length > 1)

/** File names used by more than one page: those rows must show their full path. */
const duplicateFileNames = computed(() => {
  const counts = new Map()
  for (const item of pages.value) {
    const name = (item.file_name || '').trim().toLowerCase()
    if (!name) continue
    counts.set(name, (counts.get(name) || 0) + 1)
  }
  return new Set([...counts].filter(([, count]) => count > 1).map(([name]) => name))
})

const pageNameOf = (item) => (item.file_name || '').trim()
/**
 * Where the page was uploaded from. Only a folder upload knows the full path;
 * the server-side storage path is never shown — it says nothing to the user.
 */
const pageOrigin = (item) => item.source_path || pageNameOf(item)
/** What the sidebar shows: the file name, or the upload path when names collide. */
const pageLabel = (item) => {
  const name = pageNameOf(item)
  if (!name) return `Стр. #${item.page_id}`
  return duplicateFileNames.value.has(name.toLowerCase()) ? pageOrigin(item) : name
}

// the dictionary panel is read whole and filtered in the browser: the list is
// the author's own words only, which is small enough for that

const filteredLexiconWords = computed(() => {
  const query = lexiconQuery.value.trim().toLowerCase()
  if (!query) return lexiconWords.value
  return lexiconWords.value.filter((item) => item.word.includes(query))
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
  SUCCESS: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  INSUFFICIENT_DATA: 'border-amber-400/40 bg-amber-400/10 text-amber-300',
  NO_IMPROVEMENT: 'border-amber-400/40 bg-amber-400/10 text-amber-300',
  BUSY: 'border-app-border/60 bg-app-panel/60 text-app-text',
  FAILED: 'border-app-error/40 bg-app-error/10 text-app-error',
}[training.value?.outcome] || 'border-app-border/60 bg-app-panel/60 text-app-text'))

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
const wordTitle = (word) => {
  const parts = [word.effective_text ?? '']
  if (word.confidence != null) parts.push(`${(word.confidence * 100).toFixed(0)}%`)
  if (word.author_only) {
    parts.push('нет в общем словаре, но встречается в ваших подтверждённых страницах')
  }
  return parts.filter(Boolean).join(' — ')
}

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
/** The confidence of a line as the colour of its left edge (the chip is gone). */
const confidenceEdgeClass = (value) => {
  if (value == null) return 'line-card--none'
  if (value >= thresholds.value.warning) return 'line-card--ok'
  if (value >= thresholds.value.critical) return 'line-card--warn'
  return 'line-card--bad'
}

const worstConfidence = (line) => {
  const values = line.words.map((w) => w.confidence).filter((v) => v != null)
  return values.length ? Math.min(...values) : null
}

/** The text of a miss, not a word box: stale boxes may hold an older reading. */
const setTextareaRef = (lineId, el) => {
  if (el) {
    textareaRefs.set(lineId, el)
    const wrapper = el.closest('.line-editor')
    if (wrapper) widthObserver?.observe(wrapper)
  } else {
    const previous = textareaRefs.get(lineId)
    const wrapper = previous?.closest?.('.line-editor')
    if (wrapper) widthObserver?.unobserve(wrapper)
    textareaRefs.delete(lineId)
  }
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
    // the boxes are sized from their content, which only exists after paint
    await nextTick()
    growTextareas()
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
    resetView()
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

// ------------------------------------------------- per-line text actions

/** The text the user currently sees in a line (unsaved draft wins). */
const lineText = (line) => drafts[line.id] ?? effectiveText(line) ?? ''

/**
 * Copy helper with a fallback for non-secure contexts, where the async
 * clipboard API is unavailable.
 */
const copyToClipboard = async (text) => {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }
  const area = document.createElement('textarea')
  area.value = text
  area.style.position = 'fixed'
  area.style.top = '-1000px'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  try {
    document.execCommand('copy')
  } finally {
    document.body.removeChild(area)
  }
}

const copyAllText = async () => {
  if (!page.value?.lines.length) return
  copyingAll.value = true
  clearMessages()
  try {
    await copyToClipboard(page.value.lines.map(lineText).join('\n'))
    notice.value = `Скопировано строк: ${page.value.lines.length}.`
  } catch (err) {
    error.value = 'Не удалось скопировать текст в буфер обмена'
  } finally {
    copyingAll.value = false
  }
}

const copyLine = async (line) => {
  copyingLineId.value = line.id
  clearMessages()
  try {
    await copyToClipboard(lineText(line))
    notice.value = `Строка ${line.order + 1} скопирована.`
  } catch (err) {
    error.value = 'Не удалось скопировать строку в буфер обмена'
  } finally {
    copyingLineId.value = null
  }
}

/** Drop a line the segmenter invented; the server renumbers the rest. */
const deleteLine = async (line) => {
  if (!page.value || readOnly.value) return
  const text = lineText(line).trim()
  const preview = text
    ? `строку «${text.length > 60 ? `${text.slice(0, 60)}…` : text}»`
    : 'пустую строку'
  if (!confirm(`Удалить ${preview} (#${line.order + 1})? Разметка этой строки будет потеряна.`)) {
    return
  }
  deletingLineId.value = line.id
  clearMessages()
  try {
    // a reload follows, so unsaved edits of the other lines must land first
    await flushPendingSaves()
    const response = await api(`/htr/pages/${page.value.page_id}/lines/${line.id}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось удалить строку')
    }
    page.value = await response.json()
    resetDrafts()
    activeLineId.value = null
    activeWordId.value = null
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка удаления строки'
  } finally {
    deletingLineId.value = null
  }
}

// ------------------------------------------------- the author dictionary panel

const applyLexiconPayload = (payload) => {
  lexiconWords.value = payload?.words || []
}

const loadLexicon = async (silent = false) => {
  if (!authorId.value) return
  lexiconLoading.value = true
  try {
    const response = await api(`/htr/authors/${authorId.value}/lexicon`)
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось загрузить словарь автора')
    }
    applyLexiconPayload(await response.json())
  } catch (err) {
    // silent: called after a confirmation just to make its report clickable,
    // and a dictionary failure must not look like a failed confirmation
    if (!silent) error.value = err.message || 'Ошибка загрузки словаря автора'
  } finally {
    lexiconLoading.value = false
  }
}

const openLexicon = async () => {
  if (!authorId.value) return
  clearMessages()
  lexiconQuery.value = ''
  lexiconOpen.value = true
  await loadLexicon()
}

/** Re-read the annotations of the open page (no image reload, no draft reset). */
const refreshPageAnnotations = async () => {
  if (!page.value) return
  const response = await api(`/htr/pages/${page.value.page_id}`)
  if (!response.ok) return
  page.value = await response.json()
}



// ------------------------------------------------- sidebar order & page names

/** Store the order the user dragged the sidebar into. */
const savePageOrder = async (pageIds) => {
  const response = await api(`/htr/authors/${authorId.value}/pages/order`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ page_ids: pageIds }),
  })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail || 'Не удалось сохранить порядок страниц')
  }
  // the server is the source of truth: it renumbers the whole list
  pages.value = await response.json()
}

const onPageDragStart = (item, event) => {
  if (!canReorder.value) {
    event.preventDefault()
    return
  }
  dragPageId.value = item.page_id
  dropTarget.value = null
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', String(item.page_id))
  }
}

/** Half of the row under the pointer decides before/after, as file managers do. */
const onPageDragOver = (item, event) => {
  if (!dragPageId.value || dragPageId.value === item.page_id) {
    dropTarget.value = null
    return
  }
  const rect = event.currentTarget.getBoundingClientRect()
  dropTarget.value = {
    id: item.page_id,
    after: event.clientY > rect.top + rect.height / 2,
  }
}

const onPageDragEnd = () => {
  dragPageId.value = null
  dropTarget.value = null
}

const onPageDrop = async (item) => {
  const sourceId = dragPageId.value
  const target = dropTarget.value
  onPageDragEnd()
  if (!sourceId || !target || sourceId === item.page_id) return

  const next = [...pages.value]
  const from = next.findIndex((p) => p.page_id === sourceId)
  if (from < 0) return
  const [moved] = next.splice(from, 1)
  let to = next.findIndex((p) => p.page_id === target.id)
  if (to < 0) return
  if (target.after) to += 1
  next.splice(to, 0, moved)

  const previous = pages.value
  pages.value = next
  clearMessages()
  try {
    await savePageOrder(next.map((p) => p.page_id))
  } catch (err) {
    pages.value = previous
    error.value = err.message || 'Ошибка изменения порядка страниц'
  }
}

const renamePage = async (item) => {
  const current = pageNameOf(item)
  const value = prompt('Имя страницы (имя файла)', current || `Стр. #${item.page_id}`)
  if (value === null) return
  const name = value.trim()
  if (!name || name === current) return

  renamingPageId.value = item.page_id
  clearMessages()
  try {
    const response = await api(`/htr/pages/${item.page_id}/name`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_name: name }),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось переименовать страницу')
    }
    const updated = await response.json()
    item.file_name = updated.file_name
    // the upload path is dropped server-side; mirror that so a duplicate name
    // is not shown as a stale full path
    item.source_path = updated.source_path
    if (page.value?.page_id === item.page_id) {
      page.value.file_name = updated.file_name
      page.value.source_path = updated.source_path
    }
  } catch (err) {
    error.value = err.message || 'Ошибка переименования страницы'
  } finally {
    renamingPageId.value = null
  }
}

// ---------------------------------------------------------------- actions

const deletePage = async (item) => {
  const label = pageLabel(item)
  const message =
    item.status === 'CONFIRMED'
      ? `Удалить страницу «${label}» (#${item.page_id})? Она подтверждена и входит в обучающий набор — в следующих обучениях использоваться не будет. Уже обученные версии модели останутся.`
      : `Удалить страницу «${label}» (#${item.page_id})? Изображение и разметка будут удалены.`
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

const IMAGE_FILE = /\.(png|jpe?g|tiff?|bmp|webp|gif|heic|heif)$/i

/** A folder upload may pick up non-images; only images become pages. */
const isImageFile = (file) => file.type.startsWith('image/') || IMAGE_FILE.test(file.name)

const onUpload = async (event) => {
  const files = Array.from(event.target.files || [])
  event.target.value = ''
  if (!files.length || !authorId.value) return
  uploading.value = true
  error.value = ''
  notice.value = ''
  let lastPageId = null
  let uploadedCount = 0
  let skippedCount = 0
  try {
    for (const file of files) {
      if (!isImageFile(file)) {
        skippedCount += 1
        continue
      }
      const form = new FormData()
      form.append('file', file)
      // a folder upload knows the full client-side path; keeping it is what
      // lets two equal file names from different folders be told apart
      const relative = file.webkitRelativePath || ''
      if (relative) form.append('source_path', relative)
      const response = await api(`/htr/authors/${authorId.value}/pages`, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail || `Не удалось загрузить «${file.name}»`)
      }
      const created = await response.json()
      lastPageId = created.page_id
      uploadedCount += 1
    }
    await loadPages(false)
    if (lastPageId != null) await selectPage(lastPageId)
    if (skippedCount) {
      notice.value = `Загружено страниц: ${uploadedCount}. Пропущено не-изображений: ${skippedCount}.`
    }
  } catch (err) {
    error.value = err.message || 'Ошибка загрузки страницы'
    // part of a folder may already be stored: show what made it
    await loadPages(false)
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

/**
 * What leaving the confirmed state actually does — listed in the reopen modal.
 */
const REOPEN_CONSEQUENCES = [
  'текст и ваши правки сохранятся, разметка снова станет активной;',
  'страница перестанет быть эталоном — в следующем обучении она участвовать не будет;',
  'её слова перестанут считаться словами автора (лексикон и подсветка «не в словаре»);',
  'метрики подтверждения (CER/WER) будут очищены;',
  'уже обученная модель не изменится: «разучить» её нельзя, нужен новый прогон обучения.',
]

/**
 * Ask before leaving the confirmed state. The dialog is the only way in: the
 * consequences are a list, not a wall of text in a native `confirm()`.
 */
const reopenPage = () => {
  if (!page.value || !readOnly.value || reopening.value) return
  clearMessages()
  reopenPrompt.value = { pageId: page.value.page_id }
}

/** The same, started from the padlock in the page list. */
const reopenPageFromList = (item) => {
  if (reopening.value || reopeningPageId.value) return
  clearMessages()
  reopenPrompt.value = { pageId: item.page_id }
}

/**
 * Return a confirmed page to editing. The launch button for LLM proposals lives
 * on the backend only now (``POST /pages/{id}/suggestions``); proposals that are
 * already stored keep being shown and can still be accepted or dismissed.
 */
const applyReopen = async () => {
  const target = reopenPrompt.value
  if (!target) return
  reopenPrompt.value = null
  const onScreen = page.value?.page_id === target.pageId
  reopening.value = true
  reopeningPageId.value = target.pageId
  clearMessages()
  try {
    // unsaved drafts belong to the page on screen; saving them keeps the text
    // the user sees when the same page comes back editable
    if (onScreen) await flushPendingSaves()
    const response = await api(`/htr/pages/${target.pageId}/reopen`, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось вернуть страницу в редактирование')
    }
    const updated = await response.json()
    // if that page is the one on screen, show its fresh state right away
    if (onScreen) {
      page.value = updated
      resetDrafts()
    }
    await loadPages(false)
    notice.value = onScreen
      ? 'Страница снова редактируется и исключена из будущего обучения. ' +
        'Если из неё ушли слова, лексикон и языковые модели обновятся при следующем обучении.'
      : 'Страница возвращена в редактирование и исключена из будущего обучения.'
  } catch (err) {
    error.value = err.message || 'Ошибка возврата в редактирование'
  } finally {
    reopening.value = false
    reopeningPageId.value = null
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

/**
 * Open the confirmation dialog: it shows the words the page is about to teach
 * the author's dictionary, so a misread word can be fixed *before* it becomes
 * "known". Unsaved drafts are flushed first, so the dialog describes the text
 * the confirmation would actually write.
 */
const confirmPage = async () => {
  if (!page.value || !canConfirm.value) return
  clearMessages()
  confirmPreview.value = { added: [], learned: [] }
  confirmPreviewLoading.value = true
  try {
    await flushPendingSaves()
    const response = await api(`/htr/pages/${page.value.page_id}/confirmation-preview`)
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось проверить, что добавит страница')
    }
    const payload = await response.json()
    confirmPreview.value = {
      added: payload.added_author_words ?? [],
      learned: payload.author_words_learned ?? [],
    }
  } catch (err) {
    confirmPreview.value = null
    error.value = err.message || 'Ошибка подготовки подтверждения'
  } finally {
    confirmPreviewLoading.value = false
  }
}

/** Confirm the page for real — the dialog has shown what it is about to teach. */
const applyConfirmation = async () => {
  if (!page.value || confirming.value) return
  confirming.value = true
  clearMessages()
  try {
    const response = await api(`/htr/pages/${page.value.page_id}/confirm`, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось подтвердить страницу')
    }
    // the response is the confirmed page itself: confirming never trains. The
    // word report belongs to the decision, so it is gone once the decision is made
    page.value = await response.json()
    resetDrafts()
    await loadPages(false)
    confirmPreview.value = null
    notice.value = 'Страница подтверждена и вошла в обучающую выборку.'
  } catch (err) {
    error.value = err.message || 'Ошибка подтверждения'
  } finally {
    confirming.value = false
  }
}

/**
 * Where a word of the confirmation dialog stands on the open page. The dialog
 * lists the words in the form that would enter the dictionary, so each is
 * matched back to a token of the page through the same normalization.
 */
const previewWordPlace = (word) => {
  const wanted = dictionaryForm(word)
  if (!wanted || !page.value) return null
  const lines = [...(page.value.lines || [])].sort((a, b) => a.order - b.order)
  for (const line of lines) {
    const surface = lineWordTokens(effectiveText(line)).find(
      (token) => dictionaryForm(token) === wanted
    )
    if (surface) {
      return { page_id: page.value.page_id, line_id: line.id, line_order: line.order, surface }
    }
  }
  return null
}

/** Open a word of the confirmation dialog in the line where it stands. */
const showPreviewWord = async (word) => {
  const place = previewWordPlace(word)
  if (!place) {
    error.value = `Слово «${word}» не найдено на странице`
    return
  }
  // the dialog would cover the very line it opens: close it and let the user
  // fix the word; asking to confirm again shows a list of the current text
  confirmPreview.value = null
  await showLexiconWordInText({ occurrences: [place] })
}

/**
 * Fine-tune the author's model on every confirmed page.
 *
 * Deliberately separate from «Подтвердить страницу»: a page becomes ground
 * truth at confirmation time, while training costs minutes and is started by
 * the user when they want a new model version.
 */
const trainAuthor = async () => {
  if (!authorId.value) return
  trainingStarting.value = true
  clearMessages()
  training.value = null
  try {
    const response = await api(`/htr/authors/${authorId.value}/train`, { method: 'POST' })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new Error(detail?.detail || 'Не удалось запустить обучение')
    }
    training.value = await response.json()
    await loadPages(false)
  } catch (err) {
    error.value = err.message || 'Ошибка обучения'
  } finally {
    trainingStarting.value = false
  }
}

// ---------------------------------------------------------------- image navigation

/**
 * Zooming glides instead of jumping. The wheel only moves a *target*; one
 * animation loop eases the displayed zoom towards it, time-based so a 60 Hz and
 * a 144 Hz screen feel the same. The point under the cursor is the anchor: the
 * pan is recomputed from it on every frame, so the handwriting grows around the
 * pointer instead of sliding away from it.
 */
/**
 * The percentage on screen is *relative*: 100 % means "the scan is as wide as
 * the viewer". A diary scan is ~3000px wide against a ~700px column, so the same
 * view is 24 % of its natural size — a number that says nothing to the reader
 * about whether they are seeing too much or too little. The absolute scale is
 * still what the transform and the stroke widths use; only the label is
 * rebased, and the base is recomputed when the image or the window changes.
 */
const ZOOM_MIN = 0.1
const ZOOM_MAX = 8
const ZOOM_EASE_MS = 90
/** How long the zoom badge stays after the last change. */
const ZOOM_BADGE_MS = 1100

let zoomTarget = 1
let zoomAnchor = null
let panTarget = null
let zoomFrame = 0
let zoomFrameTime = 0

/** The scale at which the page fills the window by width: this is 100 %. */
const fitScale = ref(1)

/**
 * 100 % is "the scan is as wide as the window", not "the whole scan is inside
 * the window". A diary page is taller than the viewport, so fitting both ways
 * left bands on the left and right and drew the handwriting ~30 % smaller than
 * the window could show. Fitting by width uses the space there is; the rest of
 * the page is one drag (or one wheel notch) away.
 *
 * The page box is the viewer's client box exactly, so the scan's own corners sit
 * under the rounded ones and fill them.
 */
const computeFitScale = () => {
  if (!viewport.value) return fitScale.value
  const scale = viewport.value.clientWidth / pageWidth.value
  return scale > 0 ? scale : 1
}

/** How much bigger (or smaller) the view is than "fills the window by width". */
const relativeZoom = computed(() => (fitScale.value > 0 ? zoom.value / fitScale.value : 1))
const zoomPercent = computed(() => Math.round(relativeZoom.value * 100))

const clampZoom = (value) =>
  Math.min(Math.max(value, fitScale.value * ZOOM_MIN), fitScale.value * ZOOM_MAX)

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches)

/** Where the anchor point is, in image coordinates. */
const anchorAt = (viewX, viewY) => ({
  contentX: (viewX - panX.value) / zoom.value,
  contentY: (viewY - panY.value) / zoom.value,
  viewX,
  viewY,
})

/** Put the anchor back under its viewport point at the current zoom. */
const applyAnchor = () => {
  if (!zoomAnchor) return
  panX.value = zoomAnchor.viewX - zoomAnchor.contentX * zoom.value
  panY.value = zoomAnchor.viewY - zoomAnchor.contentY * zoom.value
}

/**
 * Stop the glide and make the target follow the zoom that is on screen. Call it
 * *after* any direct change of ``zoom`` — a stale target would make the next
 * wheel notch jump from wherever the target was left (measured: a fitted page at
 * 25 % jumped to 116 % on the first notch).
 */
const cancelZoomGlide = () => {
  if (zoomFrame) cancelAnimationFrame(zoomFrame)
  zoomFrame = 0
  zoomFrameTime = 0
  zoomAnchor = null
  panTarget = null
  zoomTarget = zoom.value
}

/** Jump straight to the target (page load, or when the user asked for no motion). */
const finishZoomGlide = () => {
  zoom.value = zoomTarget
  applyAnchor()
  if (panTarget) {
    panX.value = panTarget.x
    panY.value = panTarget.y
  }
  cancelZoomGlide()
}

const stepZoomGlide = (time) => {
  const elapsed = zoomFrameTime ? Math.max(1, Math.min(time - zoomFrameTime, 64)) : 16
  zoomFrameTime = time
  const ease = 1 - Math.exp(-elapsed / ZOOM_EASE_MS)
  zoom.value += (zoomTarget - zoom.value) * ease
  if (zoomAnchor) {
    applyAnchor()
  } else if (panTarget) {
    panX.value += (panTarget.x - panX.value) * ease
    panY.value += (panTarget.y - panY.value) * ease
  }
  const zoomSettled = Math.abs(zoomTarget - zoom.value) < 0.0005
  const panSettled =
    !panTarget ||
    (Math.abs(panTarget.x - panX.value) < 0.5 && Math.abs(panTarget.y - panY.value) < 0.5)
  if (zoomSettled && panSettled) {
    finishZoomGlide()
    return
  }
  zoomFrame = requestAnimationFrame(stepZoomGlide)
}

const startZoomGlide = () => {
  if (prefersReducedMotion()) {
    finishZoomGlide()
    return
  }
  if (!zoomFrame) {
    zoomFrameTime = 0
    zoomFrame = requestAnimationFrame(stepZoomGlide)
  }
}

/** Zoom towards ``value`` keeping the image point under (viewX, viewY) still. */
const glideZoomAround = (value, viewX, viewY) => {
  zoomTarget = clampZoom(value)
  panTarget = null
  zoomAnchor = anchorAt(viewX, viewY)
  showZoomBadge()
  startZoomGlide()
}

/**
 * Back to 100 %: the page exactly as wide as the viewer, flush with the top and
 * centred horizontally. The page is taller than the window, so the vertical
 * position starts at the beginning — centring it would cut off the first lines.
 * Double click and the first view of a page are the same thing, so there is one
 * definition of the baseline instead of two.
 */
const fitToWindow = (animate = false) => {
  if (!viewport.value) return
  fitScale.value = computeFitScale()
  const target = fitScale.value
  const centred = {
    x: (viewport.value.clientWidth - pageWidth.value * target) / 2,
    y: 0,
  }
  if (!animate) {
    zoom.value = target
    panX.value = centred.x
    panY.value = centred.y
    cancelZoomGlide() // snapshot the new values, so the next notch starts here
    return
  }
  zoomTarget = target
  zoomAnchor = null
  panTarget = centred
  showZoomBadge()
  startZoomGlide()
}

/** Centre the page horizontally at the given scale, starting at its top edge. */
const centreView = (scale) => {
  if (!viewport.value) {
    panX.value = 0
    panY.value = 0
    return
  }
  panX.value = (viewport.value.clientWidth - pageWidth.value * scale) / 2
  panY.value = 0
}

/** A freshly loaded page: fitted by width from the top, nothing to announce. */
const resetView = () => {
  fitScale.value = computeFitScale()
  zoom.value = fitScale.value
  centreView(zoom.value)
  cancelZoomGlide()
}

/** Keep a width-fitted page fitted when the window changes size. */
const onWindowResize = () => {
  const wasFitted = Math.abs(relativeZoom.value - 1) < 0.02
  fitScale.value = computeFitScale()
  if (wasFitted) {
    zoom.value = fitScale.value
    centreView(zoom.value)
    cancelZoomGlide()
  }
}

const zoomBadge = ref(false)
let zoomBadgeTimer = 0

/** The percentage is worth a glance while it changes, and noise afterwards. */
const showZoomBadge = () => {
  zoomBadge.value = true
  if (zoomBadgeTimer) clearTimeout(zoomBadgeTimer)
  zoomBadgeTimer = setTimeout(() => {
    zoomBadge.value = false
    zoomBadgeTimer = 0
  }, ZOOM_BADGE_MS)
}

/**
 * Plain wheel zooms around the pointer, so the pixel under the cursor stays put.
 * The step is exponential in deltaY, which keeps one mouse notch perceptible
 * without making a trackpad jump.
 */
const onWheel = (event) => {
  if (!viewport.value) return
  const raw = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaMode === 2 ? event.deltaY * 400 : event.deltaY
  const step = Math.min(Math.max(Math.exp(-raw * 0.0015), 0.8), 1.25)
  const rect = viewport.value.getBoundingClientRect()
  // accumulate on the target: a fast wheel must not wait for the animation
  glideZoomAround(zoomTarget * step, event.clientX - rect.left, event.clientY - rect.top)
}

const onPointerDown = (event) => {
  if (event.button !== 0) return
  if (event.target.closest('[data-word]')) return
  cancelZoomGlide() // dragging must be 1:1 with the pointer, not racing a glide
  panning.value = true
  panStart = { x: event.clientX, y: event.clientY, panX: panX.value, panY: panY.value }
}

const onPointerMove = (event) => {
  if (!panning.value || !panStart) return
  // free drag on purpose: the page may be pulled past its own edges
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

/** The word the user clicked last, wherever it lives on the page. */
const activeWord = computed(() => {
  if (!activeWordId.value) return null
  return allWords.value.find((word) => word.id === activeWordId.value) || null
})

/** Alternative readings of the active word, if it belongs to this line. */
const activeWordAlternatives = (line) => {
  const word = activeWord.value
  if (!word || !(line.words || []).some((item) => item.id === word.id)) return []
  return word.alternatives || []
}

/**
 * Replace the active word with one of the readings the recognizer considered.
 * The word keeps its position, so the line's word geometry stays valid, and the
 * new text is saved through the same path a manual edit takes.
 */
const applyAlternative = async (line, alternative) => {
  if (readOnly.value || !alternative) return
  const word = activeWord.value
  if (!word) return
  const text = drafts[line.id] ?? effectiveText(line)
  const span = tokenSpan(text, word.order)
  if (!span) return
  const updated = text.slice(0, span[0]) + alternative.text + text.slice(span[1])
  if (updated === text) return
  drafts[line.id] = updated
  clearMessages()
  await saveLine(line)
  // the line is re-read from the server, so its words hold the fresh state
  activeWordId.value = null
}

const tokenSpan = (text, index) => {
  const spans = []
  const re = /\S+/g
  let match
  while ((match = re.exec(text)) !== null) spans.push([match.index, match.index + match[0].length])
  return spans[index] || null
}

const escapeHtml = (value) =>
  String(value ?? '').replace(
    /[&<>"']/g,
    (character) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]
  )

/**
 * Everything the app knows about the words of a line, painted in place: the
 * dictionary verdict (fuchsia = not in the dictionary, violet + dotted = the
 * author's own word) and the confidence (amber = below the warning threshold,
 * red = below the critical one). Colour carries the strongest signal; when a word
 * has two, the second one stays as an underline, so nothing is silently dropped.
 * The marks explain themselves in a tooltip.
 */
const wordTooltip = (value, { isMiss, isAuthor, level, confidence }) => {
  const parts = []
  if (isMiss) parts.push('нет в словаре — проверьте слово')
  else if (isAuthor) {
    parts.push('слово автора: в общем словаре его нет, но оно есть на подтверждённых страницах')
  }
  if (level === 'warning' || level === 'critical') {
    const percent = Math.round((confidence ?? 0) * 100)
    const limit = level === 'critical' ? thresholds.value.critical : thresholds.value.warning
    parts.push(`уверенность ${percent} % — ниже порога ${limit}`)
  }
  parts.push('кликните в слово, чтобы увидеть варианты декодера')
  return parts.join(' · ')
}

const lineHighlightHtml = (line) => {
  const text = drafts[line.id] ?? line.corrected_text ?? line.predicted_text ?? ''
  if (!text) return escapeHtml(text)
  const dictionary = lexiconAvailable.value
  const misses = dictionary ? new Set((line.oov_words || []).map(comparableWord)) : new Set()
  const authored = dictionary ? new Set((line.author_only_words || []).map(comparableWord)) : new Set()
  const byOrder = new Map((line.words || []).map((word) => [word.order, word]))
  const fresh = !line.words_stale
  let index = -1
  return text.replace(/\s+|\S+/g, (chunk) => {
    if (!chunk.trim()) return chunk
    index += 1
    const word = fresh ? byOrder.get(index) : null
    const level = word?.confidence_level
    const doubtful = level === 'warning' || level === 'critical'
    const key = comparableWord(chunk)
    const isMiss = misses.has(key)
    const isAuthor = authored.has(key)
    if (!doubtful && !isMiss && !isAuthor) return escapeHtml(chunk)
    // critical beats warning, doubt beats the dictionary verdict
    const colour =
      level === 'critical'
        ? 'word-critical'
        : level === 'warning'
          ? 'word-warning'
          : isMiss
            ? 'word-oov'
            : 'word-author'
    const underline = doubtful && isMiss ? 'word-under-oov' : doubtful && isAuthor ? 'word-under-author' : ''
    const tooltip = wordTooltip(chunk, {
      isMiss,
      isAuthor,
      level,
      confidence: word?.confidence,
    })
    return `<mark class="${colour} ${underline}" data-word-index="${index}" title="${escapeHtml(tooltip)}">${escapeHtml(chunk)}</mark>`
  })
}

/** Index of the word the caret sits in (or just after). */
const tokenIndexAt = (text, position) => {
  const re = /\S+/g
  let match
  let index = 0
  while ((match = re.exec(text)) !== null) {
    if (position <= match.index + match[0].length) return index
    index += 1
  }
  return Math.max(0, index - 1)
}

/** Keep a textarea as tall as its content: no inner scrollbar, no clipped text. */
const autoGrow = (element) => {
  if (!element) return
  element.style.height = 'auto'
  // ceil + 1: a fractional content height rounded down by the browser clips the
  // last pixel of the descenders (and used to summon a scrollbar on Windows)
  element.style.height = `${Math.ceil(element.scrollHeight) + 1}px`
}

const growTextareas = () => {
  textareaRefs.forEach((element) => autoGrow(element))
}

/**
 * Re-grow a box when its **width** changes — a window resize, a panel, or a
 * scrollbar appearing (the Windows trap). Height changes are ignored on purpose:
 * reacting to them would loop, since the observer watches the box we resize.
 */
const widthObserver =
  typeof ResizeObserver === 'undefined'
    ? null
    : new ResizeObserver((entries) => {
        for (const entry of entries) {
          const area = entry.target.querySelector('.line-editor__input')
          if (!area) continue
          const width = entry.contentRect.width
          if (area.dataset.grownWidth && Math.abs(Number(area.dataset.grownWidth) - width) < 0.5) {
            continue
          }
          area.dataset.grownWidth = String(width)
          autoGrow(area)
        }
      })

/**
 * Clicking a word *in the transcription* should behave like clicking in any
 * text field: the browser places the caret where the pointer is, and the user
 * edits right there. The variants panel and the white outline on the scan still
 * follow the word under the caret, but they never steal the selection.
 */
const selectWordContext = (line, index) => {
  const word = (line.words || []).find((entry) => entry.order === index)
  activeLineId.value = line.id
  if (line.words_stale) {
    activeWordId.value = null
    notice.value = 'Разметка этой строки устарела — варианты декодера к ней не относятся'
    return
  }
  activeWordId.value = word ? word.id : null
  // the scan and the transcription are two different scroll areas: a word picked
  // in the text has to be found on the image as well (no-op when it already is)
  if (word) bringWordIntoView(word)
}

/**
 * Is the word properly inside the viewer? The scan and the transcription are two
 * different scroll areas, so a word picked in the text is often off screen on the
 * image. Only a mostly visible word counts: a word whose outline pokes out from
 * under an edge still has to be brought out, but one that is merely clipped by a
 * few pixels should not cost the reader their place.
 */
const isWordVisible = (word) => {
  if (!viewport.value || !word?.bbox) return true
  const { x1, y1, x2, y2 } = word.bbox
  const left = x1 * zoom.value + panX.value
  const top = y1 * zoom.value + panY.value
  const right = x2 * zoom.value + panX.value
  const bottom = y2 * zoom.value + panY.value
  const width = right - left
  const height = bottom - top
  const insideX = Math.min(right, viewport.value.clientWidth) - Math.max(left, 0)
  const insideY = Math.min(bottom, viewport.value.clientHeight) - Math.max(top, 0)
  return insideX >= width * 0.8 && insideY >= height * 0.8
}

/**
 * Centre the viewer on the word. Skipped when it is already visible: moving a page
 * the reader is looking at just to answer a click costs them their place. Reuses
 * the glide the zoom already has, so the move eases instead of jumping and
 * respects ``prefers-reduced-motion``.
 *
 * The pan itself waits one tick: the click that got here may be about to expand
 * the alternatives panel under the image, and the viewer's geometry is measured
 * *after* that layout settles, not before.
 */
const bringWordIntoView = async (word) => {
  if (!viewport.value || !word?.bbox) return
  await nextTick()
  if (!viewport.value || isWordVisible(word)) return
  const { x1, y1, x2, y2 } = word.bbox
  const target = {
    x: viewport.value.clientWidth / 2 - ((x1 + x2) / 2) * zoom.value,
    y: viewport.value.clientHeight / 2 - ((y1 + y2) / 2) * zoom.value,
  }
  zoomTarget = zoom.value
  zoomAnchor = null
  panTarget = target
  showZoomBadge()
  startZoomGlide()
}

const onTextareaClick = (line, event) => {
  if (readOnly.value) return
  const element = event.target
  // a drag or a double click already selected what the user wanted: keep it
  if (element.selectionStart !== element.selectionEnd) {
    activeLineId.value = line.id
    return
  }
  // A plain click already left the caret where it was clicked -- don't move it
  // by selecting the whole word; just show which word the caret is in.
  selectWordContext(line, tokenIndexAt(element.value ?? '', element.selectionStart ?? 0))
}

/**
 * The coloured words are click-through now (see the CSS), so the native `title`
 * tooltip is gone. Find the mark under the pointer by its rectangle and paint
 * the same explanation in the shared ``.word-tooltip`` element.
 */
const markAtPoint = (editor, x, y) => {
  for (const mark of editor.querySelectorAll('mark[data-word-index]')) {
    const rect = mark.getBoundingClientRect()
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      return mark
    }
  }
  return null
}

const onEditorMousemove = (event) => {
  const editor = event.currentTarget
  const mark = markAtPoint(editor, event.clientX, event.clientY)
  const text = mark?.getAttribute('title')
  if (!text) {
    hoverTooltip.value = null
    return
  }
  if (
    hoverTooltip.value &&
    hoverTooltip.value.text === text &&
    hoverTooltip.value.x === event.clientX &&
    hoverTooltip.value.y === event.clientY
  ) {
    return
  }
  hoverTooltip.value = { text, x: event.clientX, y: event.clientY }
}

/** Only letters/digits/hyphens, case-insensitive: enough to match a word box. */
const comparableWord = (value) =>
  (value || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}-]+/gu, '')
    .replace(/ъ$/, '')

/**
 * The dictionary form of a token — the frontend twin of the backend
 * ``normalize_word``: casefold, pre-1918 letters mapped to modern ones, no
 * word-final hard sign. The confirmation dialog names words in this form, so it
 * is what matches one back to the spelling actually written on the page.
 */
const dictionaryForm = (value) =>
  (value || '')
    .toLowerCase()
    .replace(/[ѣіѳѵ]/g, (letter) => ({ ѣ: 'е', і: 'и', ѳ: 'ф', ѵ: 'и' }[letter]))
    .replace(/ъ$/, '')

/** The word tokens of a line, tokenized the way the backend reads them. */
const lineWordTokens = (text) => (text || '').match(/[\w'’\-]+/gu) || []

/** Span of the first token of ``text`` that reads like ``word``. */
const tokenSpanOf = (text, word) => {
  const wanted = comparableWord(word)
  if (!wanted) return null
  const re = /\S+/g
  let match
  while ((match = re.exec(text)) !== null) {
    if (comparableWord(match[0]) === wanted) {
      return [match.index, match.index + match[0].length]
    }
  }
  return null
}

/**
 * Jump to a word of the author dictionary: open its page, scroll to the line
 * that holds it and put the cursor on the word, so it can be corrected right
 * away. Falls back to selecting the token in the textarea when the line has no
 * geometry for it (a stale markup, or a word added after recognition).
 */
const showLexiconWordInText = async (item) => {
  const place = item?.occurrences?.[0]
  if (!place) return
  lexiconOpen.value = false
  try {
    if (page.value?.page_id !== place.page_id) {
      await selectPage(place.page_id)
    }
    await nextTick()
    const line = page.value?.lines.find((entry) => entry.id === place.line_id)
    if (!line) {
      error.value = 'Строка со словом не найдена — страница изменилась'
      return
    }
    const geometryWord = (line.words || []).find(
      (word) => comparableWord(word.effective_text) === comparableWord(place.surface)
    )
    if (geometryWord) {
      await onWordClick(geometryWord)
      return
    }
    activeLineId.value = line.id
    activeWordId.value = null
    await nextTick()
    const textarea = textareaRefs.get(line.id)
    if (!textarea) return
    textarea.closest('article')?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    textarea.focus({ preventScroll: true })
    const text =
      drafts[line.id] ?? line.corrected_text ?? line.predicted_text ?? ''
    const span = tokenSpanOf(text, place.surface)
    if (span) textarea.setSelectionRange(span[0], span[1])
    notice.value = `Слово «${place.surface}» — строка ${line.order + 1}, можно исправить`
  } catch (err) {
    error.value = err.message || 'Не удалось показать слово в тексте'
  }
}

// ---------------------------------------------------------------- lifecycle

onBeforeUnmount(() => {
  cancelZoomGlide()
  if (zoomBadgeTimer) clearTimeout(zoomBadgeTimer)
  window.removeEventListener('resize', onWindowResize)
})

onMounted(async () => {
  window.addEventListener('resize', onWindowResize)
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
  border: 1px solid rgb(var(--c-line) / 0.45);
  background: rgb(var(--c-input) / 0.6);
  color: rgb(var(--c-text));
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

/* ---------------------------------------------------------------------------
 * the page toolbar: one row of controls instead of a landing-page header
 * ------------------------------------------------------------------------ */

/* the author picker is a control, not a headline: smaller than the block inputs */
.app-select--compact {
  border-radius: 0.55rem;
  padding: 0.28rem 0.6rem;
  font-size: 0.8rem;
  min-width: 7rem;
}

.app-ghost--compact {
  padding: 0.3rem 0.7rem;
  font-size: 0.75rem;
}

/* training is a rare, expensive action: visible, but it must not compete with
   "upload a page", which is what the page is for */
.app-ghost--train {
  border-color: rgb(var(--c-ok) / 0.45);
  color: rgb(var(--c-ok-text-soft));
}

.app-ghost--train:hover:not(:disabled) {
  background: rgb(var(--c-ok) / 0.16);
  border-color: rgb(var(--c-ok) / 0.7);
}

.app-primary {
  background: rgb(var(--c-primary) / 0.95);
  color: rgb(var(--c-on-primary));
  padding: 0.45rem 1.1rem;
  font-size: 0.875rem;
}

.app-primary:hover:not(:disabled) {
  background: rgb(var(--c-primary) / 0.8);
}

.app-success {
  background: rgb(var(--c-ok) / 0.9);
  color: rgb(var(--c-ok-on));
  padding: 0.45rem 1.1rem;
  font-size: 0.875rem;
}

.app-success:hover:not(:disabled) {
  background: rgb(var(--c-ok) / 0.75);
}

.app-ghost {
  background: rgb(var(--c-surface) / 0.5);
  border-color: rgb(var(--c-line) / 0.4);
  color: rgb(var(--c-on-surface));
  padding: 0.45rem 0.9rem;
  font-size: 0.8rem;
}

.app-ghost:hover:not(:disabled) {
  background: rgb(var(--c-soft) / 0.75);
}

.app-icon {
  width: 2rem;
  height: 2rem;
  background: rgb(var(--c-surface) / 0.5);
  border-color: rgb(var(--c-line) / 0.4);
  color: rgb(var(--c-on-surface));
  line-height: 1;
}

.app-icon:hover {
  background: rgb(var(--c-soft) / 0.75);
}

.page-delete {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.5rem;
  padding: 0.15rem 0.25rem;
  color: rgb(var(--c-muted));
  background: transparent;
  transition: background 0.15s ease, color 0.15s ease;
}

.page-delete:hover:not(:disabled) {
  background: rgb(var(--c-error) / 0.2);
  color: rgb(var(--c-error));
}

.page-delete:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* --- sidebar ordering ----------------------------------------------------- */

.page-item[draggable='true'] {
  cursor: grab;
}

.page-item--dragging {
  opacity: 0.45;
  cursor: grabbing;
}

/* a line on the edge the page would land on */
.page-item--drop-before {
  box-shadow: inset 0 2px 0 0 rgb(var(--c-primary) / 0.95);
}

.page-item--drop-after {
  box-shadow: inset 0 -2px 0 0 rgb(var(--c-primary) / 0.95);
}

.page-grip {
  display: inline-flex;
  flex: none;
  align-items: center;
  color: rgb(var(--c-dim));
  cursor: grab;
}

.page-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 0.5rem;
  padding: 0.15rem 0.25rem;
  color: rgb(var(--c-muted));
  background: transparent;
  transition: background 0.15s ease, color 0.15s ease;
}

.page-icon:hover:not(:disabled) {
  background: rgb(var(--c-primary) / 0.2);
  color: rgb(var(--c-accent));
}

.page-icon:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* --- transcription lines -------------------------------------------------- */

.line-card:hover {
  border-color: rgb(var(--c-line) / 0.65);
}

/* the active line also gets a quiet accent bar on the left */
.line-card--active {
  box-shadow: inset 2px 0 0 0 rgb(var(--c-primary) / 0.9);
}

.line-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  border-radius: 9999px;
  padding: 0.05rem 0.5rem;
  font-size: 0.65rem;
  font-weight: 600;
  line-height: 1.5;
  white-space: nowrap;
}

/* the dot inherits the badge colour, so one look says "how sure" */
.line-chip__dot {
  width: 0.35rem;
  height: 0.35rem;
  border-radius: 9999px;
  background: currentColor;
}

.line-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.6rem;
  height: 1.6rem;
  border-radius: 0.5rem;
  color: rgb(var(--c-dim));
  background: transparent;
  transition: background 0.12s ease, color 0.12s ease;
}

.line-action:hover:not(:disabled) {
  background: rgb(var(--c-primary) / 0.18);
  color: rgb(var(--c-accent-strong));
}

.line-action--danger:hover:not(:disabled) {
  background: rgb(var(--c-error) / 0.2);
  color: rgb(var(--c-error));
}

.line-action:disabled {
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
  border: 1px solid rgb(var(--c-line) / 0.3);
  background: rgb(var(--c-input) / 0.6);
  color: rgb(var(--c-text));
  padding: 0.5rem 0.65rem;
  font-size: 0.9rem;
  line-height: 1.45;
}

.app-textarea:focus {
  outline: none;
  border-color: rgb(var(--c-primary) / 0.8);
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
  border-color: rgb(var(--c-magenta));
  background: rgb(var(--c-magenta) / 0.35);
}

/* the author's own words: known only because their pages confirmed them */
.legend-box--author {
  border-color: rgb(var(--c-oov-text));
  border-style: dotted;
  background: transparent;
}

/* ---------------------------------------------------------------------------
 * the transcription box: a coloured layer behind a transparent textarea, so the
 * dictionary verdict is painted on the word itself ("не в словаре" used to be a
 * counter chip in the line header, which said how many but never which)
 * ------------------------------------------------------------------------ */
.line-editor {
  position: relative;
  border: 1px solid rgb(var(--c-line) / 0.28);
  border-radius: 0.65rem;
  background: rgb(var(--c-input) / 0.55);
  transition: border-color 0.12s ease;
}

.line-editor:hover {
  border-color: rgb(var(--c-line) / 0.5);
}

/* the transcription is frozen: a dashed box says so without another badge */
.line-editor--locked {
  border-style: dashed;
}

.line-editor:focus-within {
  border-color: rgb(var(--c-primary) / 0.75);
}

/*
 * Both layers must lay the text out identically, and a proportional font cannot
 * do that: the bold word of the layer is wider than the same word in the
 * textarea, the layer wraps earlier, the mark slides to a second line and the
 * caret (which the browser maps through the textarea's own layout) lands on the
 * wrong word. Monospace keeps the advance width of bold and regular identical,
 * so the two layouts agree character for character.
 */
.line-editor__layer,
.line-editor__input {
  margin: 0;
  padding: 0.5rem 0.65rem;
  font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", "DejaVu Sans Mono",
    Menlo, Consolas, monospace;
  font-size: 0.85rem;
  line-height: 1.5;
  letter-spacing: normal;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  word-break: break-word;
  tab-size: 4;
}

.line-editor__layer {
  position: absolute;
  inset: 0;
  /* the textarea owns the selection; the layer only draws */
  user-select: none;
  /* above the textarea: the browser paints the selection rectangle with the
     textarea, so a layer underneath would be hidden exactly when the user
     clicks a word to see what the dictionary thinks of it */
  z-index: 1;
  overflow: hidden;
  color: rgb(var(--c-text));
  pointer-events: none;
}

/*
 * The layer draws the text, the textarea only carries the caret and the
 * selection: a textarea cannot colour one word, and painting the marks under
 * visible text left the white glyphs showing through the coloured ones. So the
 * textarea's own text is transparent (the caret keeps a colour of its own) and
 * the layer on top of it is what the user reads.
 */
.line-editor__input {
  position: relative;
  display: block;
  width: 100%;
  border: none;
  border-radius: inherit;
  background: transparent;
  color: transparent;
  caret-color: rgb(var(--c-text));
  outline: none;
  resize: none;
  /*
   * The box grows with its content, so it must never scroll. A classic Windows
   * scrollbar is the trap here: it appears on a 1px overflow, takes ~17px of
   * width, the text then wraps one line further and the scrollbar stays — a
   * feedback loop that both clipped the text and left the box too tall
   * (measured on Windows Chrome, invisible with Linux overlay scrollbars).
   */
  overflow: hidden;
  scrollbar-width: none;
}

.line-editor__input::-webkit-scrollbar {
  display: none;
}

.line-editor__input:disabled {
  caret-color: transparent;
}

.line-editor__input::placeholder {
  color: rgb(var(--c-dim) / 0.7);
}

/*
 * The browser paints the selected text with its own colour (highlighttext,
 * white) -- over the coloured word the layer draws, which doubled every glyph.
 * Selected or not, the letters come from the layer; only the highlight block
 * belongs to the textarea.
 */
.line-editor__input::selection {
  background: rgb(var(--c-primary) / 0.4);
  color: transparent;
}

/*
 * The verdict on the word itself: colour and weight, no background. ``:deep``
 * is required because these elements come from ``v-html`` and therefore never
 * carry the scoped-style attribute -- without it the browser paints its own
 * yellow ``mark`` (measured: that is exactly what happened).
 */
:deep(mark.word-oov) {
  background: transparent;
  color: rgb(var(--c-magenta));
  font-weight: 700;
}

:deep(mark.word-author) {
  background: transparent;
  color: rgb(var(--c-oov-text));
  font-weight: 700;
  border-bottom: 1px dotted rgb(var(--c-oov-text) / 0.85);
}

/* low confidence, in the same amber and red the scan uses */
:deep(mark.word-warning) {
  background: transparent;
  color: rgb(var(--c-warn-text));
  font-weight: 700;
}

:deep(mark.word-critical) {
  background: transparent;
  color: rgb(var(--c-error));
  font-weight: 700;
}

/* doubtful *and* a dictionary miss: the colour tells the doubt, the underline
   keeps the dictionary verdict visible */
:deep(mark.word-under-oov) {
  border-bottom: 1px solid rgb(var(--c-magenta) / 0.9);
}

:deep(mark.word-under-author) {
  border-bottom: 1px dotted rgb(var(--c-oov-text) / 0.85);
}

/*
 * The marks stay click-through: the whole layer is ``pointer-events: none`` so
 * a click anywhere (including on a coloured word) reaches the textarea and the
 * browser places the caret exactly where the pointer is. The explanation of the
 * colours is drawn by the shared ``.word-tooltip`` on hover instead.
 */

/* explanation of a coloured word, pinned to the pointer */
.word-tooltip {
  position: fixed;
  z-index: 60;
  max-width: 22rem;
  padding: 0.4rem 0.65rem;
  border-radius: 0.55rem;
  border: 1px solid rgb(var(--c-line) / 0.4);
  background: rgb(2 6 23 / 0.94);
  color: rgb(var(--c-on-surface));
  font-size: 0.72rem;
  line-height: 1.4;
  white-space: normal;
  pointer-events: none;
  transform: translate(12px, 14px);
  box-shadow: 0 6px 18px rgb(0 0 0 / 0.45);
}

/*
 * The confidence the chip used to spell out is now the colour of the card's left
 * edge: readable at a glance, and it costs no vertical space.
 */
/* quiet for a confident line, louder as the confidence drops: the edge is read
   at a glance, so the doubtful lines must be the ones that stand out */
.line-card--ok {
  border-left: 3px solid rgb(var(--c-ok) / 0.45);
}

.line-card--warn {
  border-left: 3px solid rgb(var(--c-warn) / 0.9);
}

.line-card--bad {
  border-left: 3px solid rgb(var(--c-error) / 0.9);
}

.line-card--none {
  border-left: 3px solid rgb(var(--c-line) / 0.35);
}

/* copy and delete live right of the box, not on a row of their own; the column
   is narrower than the box would like, so the two buttons are shrunk to keep the
   text from wrapping (measured: a row beside the box costs 16 wrapped lines) */
.line-card__tools {
  opacity: 0.55;
  transition: opacity 0.12s ease;
}

.line-card__tools .line-action {
  width: 1.35rem;
  height: 1.35rem;
}

.line-card:hover .line-card__tools,
.line-card:focus-within .line-card__tools {
  opacity: 1;
}

/* a variant the decoder considered, offered under the word it belongs to */
.alt-chip {
  border: 1px solid rgb(var(--c-line) / 0.5);
  border-radius: 9999px;
  background: rgb(var(--c-surface) / 0.65);
  padding: 0.05rem 0.5rem;
  font-size: 0.72rem;
  line-height: 1.35;
  transition: border-color 0.12s ease, background 0.12s ease;
}

.alt-chip:hover:not(:disabled) {
  border-color: rgb(var(--c-primary) / 0.85);
  background: rgb(var(--c-primary) / 0.2);
}

.alt-chip:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.app-mini {
  border-radius: 9999px;
  border: 1px solid rgb(var(--c-line) / 0.4);
  background: rgb(var(--c-surface) / 0.6);
  color: rgb(var(--c-on-surface));
  padding: 0.1rem 0.6rem;
  font-size: 0.7rem;
  transition: background 0.12s ease;
}

.app-mini:hover:not(:disabled) {
  background: rgb(var(--c-soft) / 0.85);
}

.app-mini--ok {
  border-color: rgb(var(--c-ok) / 0.6);
  background: rgb(var(--c-ok) / 0.18);
  color: rgb(var(--c-ok-text));
}

.app-mini--ok:hover:not(:disabled) {
  background: rgb(var(--c-ok) / 0.32);
}

.app-mini:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.app-ghost--verify {
  border-color: rgb(var(--c-ok) / 0.45);
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
  border-color: rgb(var(--c-ok) / 0.5);
  background: rgb(var(--c-ok) / 0.12);
  color: rgb(var(--c-ok-text));
}

.change-chip--oov {
  border-color: rgb(var(--c-oov) / 0.6);
  background: rgb(var(--c-oov) / 0.16);
  color: rgb(var(--c-oov-text));
}

.change-chip--unknown {
  border-color: rgb(var(--c-line) / 0.45);
  background: rgb(var(--c-overlay-soft) / 0.12);
  color: rgb(var(--c-on-surface));
}

.oov-chip {
  border-radius: 9999px;
  border: 1px solid rgb(var(--c-oov) / 0.5);
  background: rgb(var(--c-oov) / 0.16);
  color: rgb(var(--c-oov-text));
  padding: 0.05rem 0.5rem;
  font-size: 0.7rem;
  line-height: 1.3;
  transition: background 0.12s ease;
}

.oov-chip:hover {
  background: rgb(var(--c-oov) / 0.35);
}

/* --- SVG overlay: polygons follow the text, not the page axes ------------- */

.line-poly {
  fill: none;
  stroke: rgb(var(--c-overlay-line) / 0.45);
  pointer-events: none;
  /* the stroke is kept ~1px on screen against the zoom; it arrives as a variable
     so one zoom frame updates a single element instead of every polygon (there
     are hundreds of them on a full page) */
  stroke-width: var(--line-stroke, 1.1);
}

.line-poly--active {
  stroke: rgb(var(--c-accent) / 0.95);
}

.line-poly--stale {
  stroke-dasharray: 8 6;
  opacity: 0.55;
}

.word-poly {
  cursor: pointer;
  stroke-width: var(--word-stroke, 1.5);
  transition: fill 0.12s ease;
  /* the scan is light paper: a dark halo keeps the outline readable on it */
  filter: drop-shadow(0 0 1px rgb(2 6 23 / 0.8));
}

/* confidence keeps amber/red (fill + outline) ... */
.word-poly--normal {
  fill: rgb(var(--c-overlay-soft) / 0.08);
  stroke: rgb(var(--c-overlay-line) / 0.75);
}

.word-poly--normal:hover {
  fill: rgb(var(--c-overlay-strong) / 0.25);
  stroke: rgb(var(--c-overlay-strong));
}

.word-poly--warning {
  fill: rgb(var(--c-warn) / 0.22);
  stroke: #f59e0b;
}

.word-poly--warning:hover {
  fill: rgb(var(--c-warn) / 0.42);
}

.word-poly--critical {
  fill: rgb(var(--c-error) / 0.28);
  stroke: #ef4444;
}

.word-poly--critical:hover {
  fill: rgb(var(--c-error) / 0.5);
}

/*
 * ... while "not in the dictionary" gets a colour channel of its own: a violet
 * fill, so an amber or red outline stays readable on the same word. The
 * two-class selectors win over the confidence fills without !important.
 */
.word-poly.word-poly--oov {
  fill: rgb(var(--c-oov) / 0.32);
}

.word-poly.word-poly--oov:hover {
  fill: rgb(var(--c-oov) / 0.5);
}

/* a confident-but-unknown word needs a violet outline as well, otherwise a
   faintly violet box between grey ones is easy to miss */
.word-poly--oov.word-poly--normal {
  stroke: rgb(var(--c-oov));
}

/*
 * Known only from the author's own confirmed pages: a name, a dialect word —
 * or a typo that was confirmed once. It is not out-of-vocabulary, so it gets a
 * dotted outline instead of the solid violet fill, and it stays visible on a
 * confirmed page where the rest of the markup is muted.
 */
.word-poly.word-poly--author {
  fill: rgb(var(--c-oov) / 0.1);
  stroke: rgb(var(--c-oov));
  stroke-dasharray: 4 4;
}

.word-poly.word-poly--author:hover {
  fill: rgb(var(--c-oov) / 0.28);
}

/* the selected word is marked on top of everything, in a colour of its own */
.word-poly.word-poly--active {
  stroke: rgb(var(--c-overlay-strong));
  stroke-width: 2px;
  filter: drop-shadow(0 0 3px rgb(var(--c-ink) / 0.95));
}

.word-poly.word-poly--active:not(.word-poly--oov) {
  fill: rgb(var(--c-primary) / 0.4);
}

.word-poly--stale {
  opacity: 0.35;
  stroke-dasharray: 6 5;
}

/* A confirmed page is frozen ground truth: the markup stays as a faint memory
   on the scan instead of fighting the handwriting for attention. */
.overlay--muted .line-poly,
.overlay--muted .word-poly {
  opacity: 0.16;
  filter: none;
}

.overlay--muted .word-poly {
  cursor: default;
}

.overlay--muted .word-poly.word-poly--active {
  opacity: 0.5;
}

/* the weak "author only" hint must survive the muting: a confirmed page is
   exactly where a self-confirmed typo hides */
.overlay--muted .word-poly.word-poly--author {
  opacity: 0.55;
}
</style>
