// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {describe, it, test, expect, beforeEach, afterEach} from 'vitest';
import type {A2UILayoutEditor} from '../app.js';

/**
 * Smoke test to ensure that the Vite dev server compilation pipeline correctly
 * transpiles Lit decorators. In Vite 8, a target compilation bug can cause
 * esbuild to serve uncompiled Stage-3 class decorators raw to the browser,
 * resulting in runtime SyntaxErrors.
 */
/**
 * Vite dev server compilation under JSDOM can be slow during cold runs,
 * particularly on CI environments. We use an increased timeout of 30 seconds
 * to prevent sporadic/flaky test timeouts.
 */
const VITE_COMPILE_TIMEOUT = 30000;

test(
  'a2ui-shell compiles and registers without syntax errors',
  async () => {
    const mod = await import('../app.js');
    expect(mod.A2UILayoutEditor).toBeDefined();
    expect(customElements.get('a2ui-shell')).toBeDefined();
  },
  VITE_COMPILE_TIMEOUT,
);

describe('a2ui-shell layout and inspector integration', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  afterEach(() => {
    document.body.innerHTML = '';
    window.history.replaceState({}, '', '/');
  });

  it('renders <playground-inspector> when ?app=playground is active and toggles/resets cleanly', async () => {
    await import('../app.js');
    window.history.replaceState({}, '', '/?app=playground');
    const el = document.createElement('a2ui-shell') as A2UILayoutEditor;
    document.body.appendChild(el);
    await el.updateComplete;

    expect(el.hasAttribute('inspector-layout')).toBe(true);
    const inspector = el.shadowRoot?.querySelector('playground-inspector') as
      | (HTMLElement & {trace: unknown})
      | null;
    expect(inspector).not.toBeNull();
    // Floating drawer toggle is hidden while the inspector panel is open
    expect(el.shadowRoot?.querySelector('.inspector-drawer-toggle')).toBeNull();

    // Click the "x" close button inside the inspector panel header
    const closeBtn = inspector!.shadowRoot?.querySelector<HTMLButtonElement>(
      '[data-testid="toggle-inspector"]',
    );
    expect(closeBtn).not.toBeNull();
    expect(closeBtn!.textContent).toContain('close');
    closeBtn!.click();
    await el.updateComplete;
    expect(el.hasAttribute('inspector-collapsed')).toBe(true);
    expect(el.shadowRoot?.querySelector('playground-inspector')).toBeNull();

    // Re-open via the floating Show Inspector button that appears when collapsed
    const toggleBtn = el.shadowRoot?.querySelector('.inspector-drawer-toggle') as HTMLButtonElement;
    expect(toggleBtn).not.toBeNull();
    expect(toggleBtn.textContent).toContain('Show Inspector');
    toggleBtn.click();
    await el.updateComplete;
    expect(el.hasAttribute('inspector-collapsed')).toBe(false);
    expect(el.shadowRoot?.querySelector('.inspector-drawer-toggle')).toBeNull();
    const reopened = el.shadowRoot?.querySelector('playground-inspector') as
      | (HTMLElement & {trace: unknown})
      | null;
    expect(reopened).not.toBeNull();

    // Dispatch reset-session event
    reopened!.dispatchEvent(new CustomEvent('reset-session', {bubbles: true, composed: true}));
    await el.updateComplete;
    expect(reopened!.trace).toBeNull();
  });

  it('omits <playground-inspector> for ?app=restaurant', async () => {
    await import('../app.js');
    window.history.replaceState({}, '', '/?app=restaurant');
    const el = document.createElement('a2ui-shell') as A2UILayoutEditor;
    document.body.appendChild(el);
    await el.updateComplete;

    expect(el.hasAttribute('inspector-layout')).toBe(false);
    expect(el.shadowRoot?.querySelector('playground-inspector')).toBeNull();
  });
});
