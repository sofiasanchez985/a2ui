/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {LitElement, TemplateResult, css, html, nothing} from 'lit';
import {customElement, property, state} from 'lit/decorators.js';

import {
  BlockInputStage,
  CandidateTrace,
  DecisionInputStage,
  DecisionOutputStage,
  LlmStage2Trace,
  PlaygroundTurnTrace,
} from '../types/inspector.js';

@customElement('playground-inspector')
export class PlaygroundInspector extends LitElement {
  @property({attribute: false})
  accessor trace: PlaygroundTurnTrace | null = null;

  @property({type: Boolean})
  accessor loading = false;

  @state()
  accessor _stageOpen: Record<number, boolean> = {
    1: true,
    2: false,
    3: true,
    4: true,
  };

  @state()
  accessor _stage2View: 'structured' | 'raw' = 'structured';

  @state()
  accessor _stage3View: 'structured' | 'raw' = 'structured';

  @state()
  accessor _stage4View: 'json' | 'express' = 'json';

  @state()
  accessor _showPruned = false;

  @state()
  accessor _selectedMode: string | null = null;

  @state()
  private accessor _copiedKey: string | null = null;

  static override styles = css`
    :host {
      display: block;
      width: 100%;
      color: light-dark(var(--n-10, #1b1b1b), var(--n-90, #e2e2e2));
      font-family: var(--font-family, sans-serif);
      font-size: 13px;
      line-height: 1.45;
    }

    * {
      box-sizing: border-box;
    }

    .material-symbols {
      font-family: 'Material Symbols Outlined', sans-serif;
      font-variation-settings: 'FILL' 1;
      font-weight: normal;
      font-style: normal;
      font-size: 16px;
      line-height: 1;
      display: inline-block;
      white-space: nowrap;
      vertical-align: middle;
    }

    .inspector-panel {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 12px;
      border-radius: 16px;
      background: light-dark(rgba(255, 255, 255, 0.82), rgba(20, 26, 36, 0.85));
      border: 1px solid light-dark(rgba(81, 84, 179, 0.2), rgba(160, 163, 255, 0.18));
      backdrop-filter: blur(12px);
    }

    .top-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      padding-bottom: 8px;
      border-bottom: 1px solid light-dark(rgba(0, 0, 0, 0.08), rgba(255, 255, 255, 0.1));
    }

    .top-bar-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 14px;
      color: light-dark(var(--p-30, #383b99), var(--n-90, #e2e2e2));
    }

    .top-bar-actions {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    button {
      font-family: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    .action-btn,
    .copy-btn,
    .pruned-toggle-btn {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid light-dark(rgba(81, 84, 179, 0.35), rgba(160, 163, 255, 0.3));
      background: light-dark(var(--n-100, #fff), var(--n-20, #303030));
      color: inherit;
      font-weight: 500;
      transition: background 0.15s ease;
    }

    .action-btn:hover,
    .copy-btn:hover,
    .pruned-toggle-btn:hover {
      background: light-dark(rgba(81, 84, 179, 0.1), rgba(160, 163, 255, 0.18));
    }

    .summary-header {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 12px;
      background: light-dark(rgba(81, 84, 179, 0.06), rgba(160, 163, 255, 0.08));
    }

    .summary-row,
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
    }

    .turn-badge {
      font-weight: 700;
      font-size: 13px;
      color: light-dark(var(--p-30, #383b99), var(--p-70, #a0a3ff));
    }

    .query-preview {
      font-family: var(--font-family-mono, monospace);
      font-size: 12px;
      opacity: 0.85;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 100%;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      background: light-dark(rgba(0, 0, 0, 0.06), rgba(255, 255, 255, 0.1));
    }

    .pill.accent {
      background: light-dark(rgba(81, 84, 179, 0.15), rgba(160, 163, 255, 0.22));
      color: light-dark(var(--p-30, #383b99), var(--p-70, #a0a3ff));
    }

    .pill.cache-hit {
      background: light-dark(rgba(16, 137, 62, 0.15), rgba(52, 211, 153, 0.2));
      color: light-dark(#0b6a2f, #6ee7b7);
    }

    .pill.danger {
      background: var(--e-95, #ffedea);
      color: var(--e-40, #ba1a1a);
    }

    .segmented-group {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px;
      background: light-dark(rgba(0, 0, 0, 0.07), rgba(255, 255, 255, 0.1));
      gap: 2px;
    }

    .segmented-btn {
      border: none;
      background: transparent;
      color: inherit;
      padding: 3px 10px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 11px;
    }

    .segmented-btn.active {
      background: var(--p-40, #5154b3);
      color: var(--n-100, #fff);
    }

    .stage-card {
      border-radius: 12px;
      border: 1px solid light-dark(rgba(0, 0, 0, 0.1), rgba(255, 255, 255, 0.12));
      background: light-dark(var(--n-100, #fff), rgba(16, 22, 32, 0.9));
      overflow: hidden;
    }

    .stage-card.has-error {
      border-color: var(--e-80, #ffb4ab);
    }

    .stage-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      padding: 10px 12px;
      border: none;
      background: light-dark(rgba(0, 0, 0, 0.025), rgba(255, 255, 255, 0.04));
      color: inherit;
      text-align: left;
      gap: 8px;
      cursor: pointer;
    }

    .stage-title-wrap {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 700;
      font-size: 13px;
    }

    .stage-chevron {
      transition: transform 0.15s ease;
    }

    .stage-card.collapsed .stage-chevron {
      transform: rotate(-90deg);
    }

    .stage-body {
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid light-dark(rgba(0, 0, 0, 0.06), rgba(255, 255, 255, 0.08));
    }

    .stage-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }

    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 6px 12px;
      font-size: 12px;
    }

    .kv-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    .kv-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      opacity: 0.65;
      font-weight: 600;
    }

    .kv-value {
      font-family: var(--font-family-mono, monospace);
      font-size: 12px;
      word-break: break-all;
    }

    .code-block {
      margin: 0;
      padding: 10px;
      border-radius: 8px;
      background: light-dark(#f4f6f9, #0d131c);
      border: 1px solid light-dark(rgba(0, 0, 0, 0.08), rgba(255, 255, 255, 0.08));
      font-family: var(--font-family-mono, monospace);
      font-size: 11.5px;
      line-height: 1.45;
      overflow-x: auto;
      max-height: 280px;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .error-callout {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 8px;
      background: var(--e-95, #ffedea);
      border: 1px solid var(--e-80, #ffb4ab);
      color: var(--e-40, #ba1a1a);
      font-weight: 500;
    }

    .question-list,
    .record-list,
    .block-tree {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .question-row,
    .record-row,
    .block-node {
      padding: 8px 10px;
      border-radius: 8px;
      background: light-dark(rgba(0, 0, 0, 0.025), rgba(255, 255, 255, 0.04));
      border: 1px solid light-dark(rgba(0, 0, 0, 0.06), rgba(255, 255, 255, 0.07));
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .question-row.pruned,
    .record-row.pruned {
      opacity: 0.62;
    }

    .record-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
    }

    .record-id {
      font-family: var(--font-family-mono, monospace);
      font-weight: 600;
    }

    .prob-bar-track {
      position: relative;
      height: 8px;
      border-radius: 999px;
      background: light-dark(rgba(0, 0, 0, 0.1), rgba(255, 255, 255, 0.12));
      overflow: visible;
    }

    .prob-bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--p-60, #8487ea), var(--p-40, #5154b3));
    }

    .threshold-tick {
      position: absolute;
      top: -3px;
      bottom: -3px;
      width: 2px;
      background: var(--e-40, #ba1a1a);
      border-radius: 1px;
    }

    .prob-options {
      font-family: var(--font-family-mono, monospace);
      font-size: 11px;
      opacity: 0.78;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 8px;
      padding: 32px 16px;
      opacity: 0.75;
    }
  `;

  #getActiveCandidate(trace: PlaygroundTurnTrace): CandidateTrace | null {
    if (trace.candidates.length === 0) {
      return null;
    }
    if (this._selectedMode) {
      const selected = trace.candidates.find(c => c.mode === this._selectedMode);
      if (selected) {
        return selected;
      }
    }
    const active = trace.candidates.find(c => c.mode === trace.activeMode);
    return active ?? trace.candidates[0];
  }

  #toggleStage(stageNum: number): void {
    this._stageOpen = {
      ...this._stageOpen,
      [stageNum]: !this._stageOpen[stageNum],
    };
  }

  #copyText(key: string, text: string): void {
    void navigator.clipboard?.writeText(text);
    this._copiedKey = key;
  }

  #dispatchToggleInspector(): void {
    this.dispatchEvent(
      new CustomEvent('toggle-inspector', {
        bubbles: true,
        composed: true,
      }),
    );
  }

  #dispatchResetSession(): void {
    this.dispatchEvent(
      new CustomEvent('reset-session', {
        bubbles: true,
        composed: true,
      }),
    );
  }

  override render(): TemplateResult {
    const trace = this.trace;
    const candidate = trace ? this.#getActiveCandidate(trace) : null;

    return html`
      <aside class="inspector-panel" aria-label="Playground pipeline inspector">
        ${this.#renderTopBar()}
        ${trace
          ? html`
              ${this.#renderSummaryHeader(trace, candidate)} ${this.#renderStage1(trace.stage1)}
              ${this.#renderStage2(candidate)} ${this.#renderStage3(candidate)}
              ${this.#renderStage4(candidate)}
            `
          : this.#renderEmptyState()}
      </aside>
    `;
  }

  #renderTopBar(): TemplateResult {
    return html`
      <div class="top-bar">
        <div class="top-bar-title">
          <span>Pipeline Inspector</span>
          ${this.loading ? html`<span class="pill accent">Running...</span>` : nothing}
        </div>
        <div class="top-bar-actions">
          <button
            type="button"
            class="action-btn reset-session-btn"
            data-testid="reset-session"
            @click=${this.#dispatchResetSession}
            title="Clear session contextId, surfaces, and trace"
          >
            <span class="material-symbols">refresh</span>
            <span>Reset session</span>
          </button>
          <button
            type="button"
            class="action-btn collapse-btn"
            data-testid="toggle-inspector"
            aria-label="Collapse inspector"
            @click=${this.#dispatchToggleInspector}
            title="Collapse inspector drawer"
          >
            <span class="material-symbols">visibility_off</span>
          </button>
        </div>
      </div>
    `;
  }

  #renderEmptyState(): TemplateResult {
    return html`
      <div class="empty-state" data-testid="inspector-empty">
        <span class="material-symbols">info</span>
        <div>
          Click a tool-mode quick action, paste a JSON payload, or enter a prompt to inspect the
          four-stage synthesis waterfall.
        </div>
      </div>
    `;
  }

  #renderSummaryHeader(
    trace: PlaygroundTurnTrace,
    candidate: CandidateTrace | null,
  ): TemplateResult {
    const mode = candidate?.mode ?? trace.activeMode;
    const backend = candidate?.metrics.backend ?? null;
    const modelId = candidate?.metrics.modelId ?? trace.stage1.model;
    const stage1Ms = Math.round(trace.stage1.latencyMs);
    const decisionMs =
      candidate?.metrics.decisionLatencyMs !== null &&
      candidate?.metrics.decisionLatencyMs !== undefined
        ? Math.round(candidate.metrics.decisionLatencyMs)
        : null;
    const totalMs = Math.round(trace.stage1.latencyMs + (candidate?.metrics.totalLatencyMs ?? 0));
    const promptTokens = trace.stage1.promptTokens + (candidate?.metrics.promptTokens ?? 0);
    const outputTokens = trace.stage1.outputTokens + (candidate?.metrics.outputTokens ?? 0);
    const isCacheHit = Boolean(candidate?.metrics.cacheHit || candidate?.decisionInput?.cacheHit);

    return html`
      <div class="summary-header" data-testid="turn-summary-header">
        <div class="summary-row">
          <span class="turn-badge">Turn #${trace.turn}</span>
          <span class="pill accent">mode: ${mode}</span>
          ${backend ? html`<span class="pill">backend: ${backend}</span>` : nothing}
          ${modelId ? html`<span class="pill">model: ${modelId}</span>` : nothing}
          ${isCacheHit
            ? html`<span class="pill cache-hit">PlanCache Hit · 0 HTTP calls</span>`
            : nothing}
        </div>
        <div class="query-preview" title=${trace.query}>${trace.query}</div>
        <div class="badge-row">
          <span class="pill">Stage 1: ${stage1Ms}ms</span>
          ${decisionMs !== null
            ? html`<span class="pill">Decision: ${decisionMs}ms</span>`
            : nothing}
          <span class="pill">Total: ${totalMs}ms</span>
          <span class="pill">${promptTokens} in / ${outputTokens} out</span>
        </div>
        ${this.#renderCompareSwitcher(trace, candidate)}
      </div>
    `;
  }

  #renderCompareSwitcher(
    trace: PlaygroundTurnTrace,
    activeCandidate: CandidateTrace | null,
  ): TemplateResult | typeof nothing {
    if (!trace.compare && trace.candidates.length <= 1) {
      return nothing;
    }
    const activeMode = activeCandidate?.mode ?? trace.activeMode;
    return html`
      <div class="summary-row" data-testid="compare-switcher">
        <span class="kv-label">Compare candidate:</span>
        <div class="segmented-group" role="tablist" aria-label="Compare mode switcher">
          ${trace.candidates.map(c => {
            const isActive = c.mode === activeMode;
            return html`
              <button
                type="button"
                role="tab"
                class="segmented-btn ${isActive ? 'active' : ''}"
                data-mode=${c.mode}
                aria-selected=${isActive ? 'true' : 'false'}
                @click=${() => {
                  this._selectedMode = c.mode;
                }}
              >
                ${c.mode}
              </button>
            `;
          })}
        </div>
      </div>
    `;
  }

  #renderStageCard(
    stageNum: number,
    title: string,
    badges: TemplateResult | typeof nothing,
    body: TemplateResult | typeof nothing,
    hasError = false,
  ): TemplateResult {
    const isOpen = Boolean(this._stageOpen[stageNum]);
    return html`
      <section
        class="stage-card ${isOpen ? 'open' : 'collapsed'} ${hasError ? 'has-error' : ''}"
        data-stage=${stageNum}
      >
        <button
          type="button"
          class="stage-header"
          data-stage-toggle=${stageNum}
          aria-expanded=${isOpen ? 'true' : 'false'}
          @click=${() => this.#toggleStage(stageNum)}
        >
          <span class="stage-title-wrap">
            <span class="material-symbols stage-chevron">arrow_drop_down</span>
            <span>${title}</span>
          </span>
          <span class="badge-row">${badges}</span>
        </button>
        ${isOpen ? html`<div class="stage-body">${body}</div>` : nothing}
      </section>
    `;
  }

  #renderCopyButton(key: string, text: string): TemplateResult {
    const copied = this._copiedKey === key;
    return html`
      <button
        type="button"
        class="copy-btn"
        data-copy-key=${key}
        @click=${() => this.#copyText(key, text)}
      >
        <span class="material-symbols">${copied ? 'check' : 'content_copy'}</span>
        <span>${copied ? 'Copied' : 'Copy'}</span>
      </button>
    `;
  }

  #renderErrorCallout(error: string | null): TemplateResult | typeof nothing {
    if (!error) {
      return nothing;
    }
    return html`
      <div class="error-callout" role="alert">
        <span class="material-symbols">error</span>
        <span>${error}</span>
      </div>
    `;
  }

  #renderStage1(stage1: BlockInputStage): TemplateResult {
    const badges = html`
      <span class="pill accent">${stage1.source}</span>
      ${stage1.model ? html`<span class="pill">${stage1.model}</span>` : nothing}
      <span class="pill">${Math.round(stage1.latencyMs)}ms</span>
      <span class="pill">${stage1.promptTokens} in / ${stage1.outputTokens} out</span>
    `;
    const copyPayload = JSON.stringify(
      stage1.block ?? stage1.toolArgs ?? stage1.rawText ?? '',
      null,
      2,
    );
    const body = html`
      <div class="stage-toolbar">
        <div class="badge-row">
          ${stage1.toolName ? html`<span class="pill">tool: ${stage1.toolName}</span>` : nothing}
        </div>
        ${this.#renderCopyButton('stage1', copyPayload)}
      </div>
      ${this.#renderErrorCallout(stage1.error)}
      ${stage1.toolArgs
        ? html`<pre class="code-block">${JSON.stringify(stage1.toolArgs, null, 2)}</pre>`
        : nothing}
      ${stage1.block ? this.#renderStage1Block(stage1.block) : nothing}
      ${!stage1.block && stage1.rawText
        ? html`<pre class="code-block">${stage1.rawText}</pre>`
        : nothing}
    `;
    return this.#renderStageCard(1, '1. Block Input', badges, body, Boolean(stage1.error));
  }

  #renderStage1Block(block: NonNullable<BlockInputStage['block']>): TemplateResult {
    return html`
      <div class="kv-grid">
        <div class="kv-item">
          <span class="kv-label">Intent</span>
          <span class="kv-value">${block.intent}</span>
        </div>
        ${block.surfaceId
          ? html`<div class="kv-item">
              <span class="kv-label">Surface ID</span>
              <span class="kv-value">${block.surfaceId}</span>
            </div>`
          : nothing}
        ${block.operation
          ? html`<div class="kv-item">
              <span class="kv-label">Operation</span>
              <span class="kv-value">${block.operation}</span>
            </div>`
          : nothing}
        ${block.layout
          ? html`<div class="kv-item">
              <span class="kv-label">Layout</span>
              <span class="kv-value">${block.layout}</span>
            </div>`
          : nothing}
      </div>
      <pre class="code-block">${JSON.stringify(block.payload, null, 2)}</pre>
    `;
  }

  #renderStage2(candidate: CandidateTrace | null): TemplateResult {
    const input = candidate?.decisionInput ?? null;
    const badges = input
      ? html`
          ${input.cacheHit
            ? html`<span class="pill cache-hit">PlanCache Hit · 0 HTTP calls</span>`
            : nothing}
          <span class="pill">${input.questionCount} questions</span>
          <span class="pill">${input.wireFormat}</span>
        `
      : html`<span class="pill">${candidate?.mode ?? 'n/a'}</span>`;

    if (!input) {
      const fallbackBody = html` <div>No decision batch input for this rendering mode.</div> `;
      return this.#renderStageCard(2, '2. Decision Input', badges, fallbackBody);
    }

    const copyContent =
      this._stage2View === 'raw' ? JSON.stringify(input.wireRequest, null, 2) : input.contextJson;

    const body = html`
      <div class="stage-toolbar">
        <div class="segmented-group" role="tablist" aria-label="Stage 2 view toggle">
          <button
            type="button"
            class="segmented-btn ${this._stage2View === 'structured' ? 'active' : ''}"
            data-stage2-view="structured"
            @click=${() => {
              this._stage2View = 'structured';
            }}
          >
            Structured
          </button>
          <button
            type="button"
            class="segmented-btn ${this._stage2View === 'raw' ? 'active' : ''}"
            data-stage2-view="raw"
            @click=${() => {
              this._stage2View = 'raw';
            }}
          >
            Raw Wire JSON
          </button>
        </div>
        ${this.#renderCopyButton('stage2', copyContent)}
      </div>
      ${input.cacheHit
        ? html`<div class="badge-row">
            <span class="pill cache-hit">PlanCache Hit · 0 HTTP calls</span>
          </div>`
        : nothing}
      ${this._stage2View === 'structured'
        ? this.#renderStage2Structured(input)
        : html`<pre class="code-block raw-wire-json" data-testid="stage2-raw-json">
${JSON.stringify(input.wireRequest, null, 2)}</pre
          >`}
    `;
    return this.#renderStageCard(2, '2. Decision Input', badges, body);
  }

  #renderStage2Structured(input: DecisionInputStage): TemplateResult {
    const totalCount = input.questions.length;
    const prunedCount = input.questions.filter(q => q.pruned).length;
    const visibleQuestions = this._showPruned
      ? input.questions
      : input.questions.filter(q => !q.pruned);

    return html`
      <div class="kv-grid">
        <div class="kv-item">
          <span class="kv-label">Shape Signature</span>
          <span class="kv-value">${input.shapeSignature ?? 'n/a'}</span>
        </div>
        <div class="kv-item">
          <span class="kv-label">Endpoint URL</span>
          <span class="kv-value">${input.endpointUrl ?? 'in-process'}</span>
        </div>
        <div class="kv-item">
          <span class="kv-label">Wire Format</span>
          <span class="kv-value">${input.wireFormat}</span>
        </div>
        <div class="kv-item">
          <span class="kv-label">Backend / Model</span>
          <span class="kv-value">${input.backendKind} (${input.modelId})</span>
        </div>
      </div>
      <div class="kv-item">
        <span class="kv-label">Context JSON</span>
        <pre class="code-block">${input.contextJson}</pre>
      </div>
      <div class="question-list" data-testid="stage2-questions">
        ${visibleQuestions.map(q => this.#renderQuestionRow(q))}
      </div>
      ${this.#renderPrunedToggle(totalCount, prunedCount)}
    `;
  }

  #renderQuestionRow(q: DecisionInputStage['questions'][number]): TemplateResult {
    const optionSummary =
      q.kind === 'boolean'
        ? `${q.trueDescription} / ${q.falseDescription}`
        : q.options.map(o => o.value).join(', ');
    return html`
      <div class="question-row ${q.pruned ? 'pruned' : ''}" data-question-id=${q.id}>
        <div class="record-header">
          <span class="record-id">${q.id}</span>
          <span class="badge-row">
            <span class="pill">${q.kind}</span>
            ${q.pruned ? html`<span class="pill">pruned</span>` : nothing}
          </span>
        </div>
        <div>${q.prompt}</div>
        <div class="prob-options">${optionSummary}</div>
      </div>
    `;
  }

  #renderPrunedToggle(totalCount: number, prunedCount: number): TemplateResult | typeof nothing {
    if (prunedCount === 0) {
      return nothing;
    }
    return html`
      <div>
        <button
          type="button"
          class="pruned-toggle-btn"
          data-testid="toggle-pruned"
          @click=${() => {
            this._showPruned = !this._showPruned;
          }}
        >
          ${this._showPruned
            ? `Hide ${prunedCount} pruned speculative questions`
            : `Show all ${totalCount} speculative questions`}
        </button>
      </div>
    `;
  }

  #renderStage3(candidate: CandidateTrace | null): TemplateResult {
    const output = candidate?.decisionOutput ?? null;
    const llmStage2 = candidate?.llmStage2 ?? null;
    const hasError = Boolean(candidate?.error);
    const cacheHit = Boolean(candidate?.metrics.cacheHit || candidate?.decisionInput?.cacheHit);

    const badges = output
      ? html`
          <span class="pill accent">${output.stage}</span>
          <span class="pill">confidence: ${output.confidence.toFixed(2)}</span>
          ${output.fitScore !== null
            ? html`<span class="pill">fit: ${output.fitScore.toFixed(2)}</span>`
            : nothing}
          <span class="pill">${output.operation}</span>
        `
      : html`<span class="pill">${candidate?.mode ?? 'n/a'}</span>`;

    const body = html`
      ${this.#renderErrorCallout(candidate?.error ?? null)}
      ${output
        ? this.#renderStage3DecisionOutput(output, cacheHit)
        : llmStage2
          ? this.#renderLlmStage2(llmStage2)
          : nothing}
    `;
    return this.#renderStageCard(3, '3. Decision Output & Plan', badges, body, hasError);
  }

  #renderStage3DecisionOutput(output: DecisionOutputStage, cacheHit: boolean): TemplateResult {
    const copyContent = JSON.stringify(
      this._stage3View === 'raw' ? output.rawAnswers : output,
      null,
      2,
    );
    return html`
      <div class="stage-toolbar">
        <div class="segmented-group" role="tablist" aria-label="Stage 3 view toggle">
          <button
            type="button"
            class="segmented-btn ${this._stage3View === 'structured' ? 'active' : ''}"
            data-stage3-view="structured"
            @click=${() => {
              this._stage3View = 'structured';
            }}
          >
            Structured
          </button>
          <button
            type="button"
            class="segmented-btn ${this._stage3View === 'raw' ? 'active' : ''}"
            data-stage3-view="raw"
            @click=${() => {
              this._stage3View = 'raw';
            }}
          >
            Raw Wire JSON
          </button>
        </div>
        ${this.#renderCopyButton('stage3', copyContent)}
      </div>
      ${this._stage3View === 'structured'
        ? this.#renderStage3Structured(output, cacheHit)
        : html`<pre class="code-block raw-wire-json" data-testid="stage3-raw-json">
${JSON.stringify(output.rawAnswers, null, 2)}</pre
          >`}
    `;
  }

  #renderStage3Structured(output: DecisionOutputStage, cacheHit: boolean): TemplateResult {
    const totalCount = output.records.length;
    const prunedCount = output.records.filter(r => r.pruned).length;
    const visibleRecords = this._showPruned
      ? output.records
      : output.records.filter(r => !r.pruned);

    return html`
      <div class="badge-row">
        <span class="pill accent">stage: ${output.stage}</span>
        <span class="pill">confidence: ${output.confidence.toFixed(2)}</span>
        <span class="pill">
          fitScore: ${output.fitScore !== null ? output.fitScore.toFixed(2) : 'n/a'}
        </span>
        <span class="pill">operation: ${output.operation}</span>
        ${output.selectedTemplateId
          ? html`<span class="pill">template: ${output.selectedTemplateId}</span>`
          : nothing}
        ${output.fallbackSource || output.fallbackReason
          ? html`<span class="pill danger">
              fallback: ${output.fallbackSource ?? ''} ${output.fallbackReason ?? ''}
            </span>`
          : nothing}
        ${cacheHit ? html`<span class="pill cache-hit">Reused cached plan</span>` : nothing}
      </div>
      <div class="record-list" data-testid="stage3-records">
        ${visibleRecords.map(r => this.#renderRecordRow(r))}
      </div>
      ${this.#renderPrunedToggle(totalCount, prunedCount)}
      ${output.layoutPlan ? this.#renderLayoutPlan(output.layoutPlan) : nothing}
    `;
  }

  #renderRecordRow(rec: DecisionOutputStage['records'][number]): TemplateResult {
    const pct = Math.max(0, Math.min(100, rec.winnerProb * 100));
    const thresholdPct =
      rec.threshold !== null ? Math.max(0, Math.min(100, rec.threshold * 100)) : null;
    const probsText = rec.probabilities
      .map(p => `${p.option}: ${(p.prob * 100).toFixed(1)}%`)
      .join(' · ');

    return html`
      <div class="record-row ${rec.pruned ? 'pruned' : ''}" data-record-id=${rec.questionId}>
        <div class="record-header">
          <span class="record-id">${rec.questionId}</span>
          <span class="badge-row">
            <span class="pill accent">${rec.winner} (${pct.toFixed(1)}%)</span>
            <span class="pill">${rec.outcome}</span>
            ${rec.pruned ? html`<span class="pill">pruned</span>` : nothing}
          </span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct.toFixed(1)}%"></div>
          ${thresholdPct !== null
            ? html`<span
                class="threshold-tick"
                style="left: ${thresholdPct.toFixed(1)}%"
                title="Threshold: ${thresholdPct.toFixed(0)}%"
              ></span>`
            : nothing}
        </div>
        <div class="prob-options">${probsText}</div>
      </div>
    `;
  }

  #renderLayoutPlan(plan: NonNullable<DecisionOutputStage['layoutPlan']>): TemplateResult {
    return html`
      <div class="kv-grid">
        <div class="kv-item">
          <span class="kv-label">Archetype</span>
          <span class="kv-value">${plan.archetype}</span>
        </div>
        <div class="kv-item">
          <span class="kv-label">Root Container</span>
          <span class="kv-value">${plan.rootContainer}</span>
        </div>
        <div class="kv-item">
          <span class="kv-label">Surface ID</span>
          <span class="kv-value">${plan.surfaceId}</span>
        </div>
      </div>
      <div class="block-tree" data-testid="layout-plan-blocks">
        ${plan.blocks.map(
          b => html`
            <div class="block-node" data-block-id=${b.blockId}>
              <code>${b.blockId}: ${b.pattern} -> ${b.component} (${b.pointer})</code>
            </div>
          `,
        )}
      </div>
    `;
  }

  #renderLlmStage2(llmStage2: LlmStage2Trace): TemplateResult {
    return html`
      <div class="badge-row">
        <span class="pill">retries: ${llmStage2.retries}</span>
      </div>
      <div class="kv-item">
        <span class="kv-label">Prompt Preview</span>
        <pre class="code-block">${llmStage2.promptPreview}</pre>
      </div>
      ${llmStage2.rawReply
        ? html`
            <div class="kv-item">
              <span class="kv-label">Raw Reply</span>
              <pre class="code-block">${llmStage2.rawReply}</pre>
            </div>
          `
        : nothing}
    `;
  }

  #renderStage4(candidate: CandidateTrace | null): TemplateResult {
    const wire = candidate?.wireOutput ?? null;
    const badges = wire
      ? html`
          <span class="pill accent">${wire.surfaceId}</span>
          <span class="pill">${wire.operation}</span>
          <span class="pill">${wire.messageCount} messages</span>
          ${wire.messageKinds.map(kind => html`<span class="pill">${kind}</span>`)}
        `
      : html`<span class="pill">0 messages</span>`;

    if (!wire) {
      return this.#renderStageCard(
        4,
        '4. A2UI JSON',
        badges,
        html`<div>No wire messages emitted.</div>`,
      );
    }

    const content =
      this._stage4View === 'json' ? JSON.stringify(wire.messages, null, 2) : wire.decompiledExpress;

    const body = html`
      <div class="stage-toolbar">
        <div class="segmented-group" role="tablist" aria-label="Stage 4 view toggle">
          <button
            type="button"
            class="segmented-btn ${this._stage4View === 'json' ? 'active' : ''}"
            data-stage4-view="json"
            @click=${() => {
              this._stage4View = 'json';
            }}
          >
            A2UI Wire JSON
          </button>
          <button
            type="button"
            class="segmented-btn ${this._stage4View === 'express' ? 'active' : ''}"
            data-stage4-view="express"
            @click=${() => {
              this._stage4View = 'express';
            }}
          >
            Decompiled Express
          </button>
        </div>
        ${this.#renderCopyButton('stage4', content)}
      </div>
      <div class="badge-row">
        <span class="pill">surfaceId: ${wire.surfaceId}</span>
        <span class="pill">operation: ${wire.operation}</span>
        <span class="pill">count: ${wire.messageCount}</span>
        ${wire.messageKinds.map(kind => html`<span class="pill accent">${kind}</span>`)}
      </div>
      <pre class="code-block" data-testid="stage4-code">${content}</pre>
    `;

    return this.#renderStageCard(4, '4. A2UI JSON', badges, body);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'playground-inspector': PlaygroundInspector;
  }
}
