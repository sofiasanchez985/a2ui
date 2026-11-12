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

/**
 * MIME type used by the Format Playground A2A server for in-band turn traces.
 */
export const PLAYGROUND_TRACE_MIME_TYPE = 'application/x-a2ui-playground-trace+json';

/**
 * Stage 1 (Block Input): how the DecisionBlock was produced.
 */
export interface BlockInputStage {
  source: 'llm' | 'direct_tool' | 'tool_call' | 'pasted_json';
  model: string | null;
  llmCalls: number;
  latencyMs: number;
  promptTokens: number;
  outputTokens: number;
  rawText: string | null;
  toolName: string | null;
  toolArgs: Record<string, unknown> | null;
  block: {
    intent: string;
    payload: unknown;
    surfaceId: string | null;
    title: string | null;
    summary: string | null;
    layout: string | null;
    region: string | null;
    operation: string | null;
    mutablePaths: string[];
    actions: Array<{name: string; label: string; context: string[]}>;
  } | null;
  error: string | null;
}

/**
 * Stage 2 (Decision Input): the question batch sent to the decision backend.
 */
export interface DecisionInputStage {
  backendKind: string;
  modelId: string;
  endpointUrl: string | null;
  wireFormat: 'gemini_decisions' | 'systemone' | 'jev' | 'in_process';
  decisionCalls: number;
  cacheHit: boolean;
  shapeSignature: string | null;
  contextJson: string;
  questionCount: number;
  questions: Array<{
    id: string;
    kind: 'boolean' | 'categorical';
    prompt: string;
    options: Array<{value: string; description: string}>;
    trueDescription: string;
    falseDescription: string;
    pruned: boolean;
  }>;
  wireRequest: Record<string, unknown> | null;
}

/**
 * Stage 3 (Decision Output & Plan): probability distribution and chosen LayoutPlan.
 */
export interface DecisionOutputStage {
  stage: 'template_select' | 'archetype' | 'abstain_fallback';
  confidence: number;
  fitScore: number | null;
  selectedTemplateId: string | null;
  fallbackSource: string | null;
  fallbackReason: string | null;
  operation: string;
  records: Array<{
    questionId: string;
    kind: 'boolean' | 'categorical';
    winner: string;
    winnerProb: number;
    threshold: number | null;
    outcome: string;
    pruned: boolean;
    probabilities: Array<{option: string; prob: number}>;
  }>;
  layoutPlan: {
    surfaceId: string;
    archetype: string;
    rootContainer: string;
    title: string;
    summary: string;
    confidence: number;
    fitScore: number;
    blocks: Array<{
      blockId: string;
      pattern: string;
      component: string;
      pointer: string;
      label: string;
      role: string;
      children: string[];
    }>;
  } | null;
  rawAnswers: Array<{
    id: string;
    kind: 'boolean' | 'categorical';
    yes?: number;
    no?: number;
    probabilities?: Record<string, number>;
  }>;
}

/**
 * Stage 2 trace for LLM-based rendering modes (`express` or `direct`).
 */
export interface LlmStage2Trace {
  promptPreview: string;
  rawReply: string | null;
  retries: number;
}

/**
 * Stage 4 (A2UI JSON & Decompiled Express): final wire output.
 */
export interface WireJsonStage {
  surfaceId: string;
  operation: string;
  messageCount: number;
  messageKinds: string[];
  messages: Array<Record<string, unknown>>;
  decompiledExpress: string;
}

/**
 * Per-mode synthesis trace entry (1 entry on normal turns, 3 on `/compare` turns).
 */
export interface CandidateTrace {
  mode: 'decision' | 'tool' | 'express' | 'direct';
  surfaceId: string;
  ok: boolean;
  error: string | null;
  metrics: {
    backend: string;
    modelId: string | null;
    llmCalls: number;
    totalLatencyMs: number;
    decisionLatencyMs: number | null;
    decisionCalls: number | null;
    promptTokens: number;
    outputTokens: number;
    cacheHit: boolean;
  };
  decisionInput: DecisionInputStage | null;
  decisionOutput: DecisionOutputStage | null;
  llmStage2: LlmStage2Trace | null;
  wireOutput: WireJsonStage;
}

/**
 * Complete per-turn trace payload carried in `application/x-a2ui-playground-trace+json`.
 */
export interface PlaygroundTurnTrace {
  schemaVersion: 1;
  turn: number;
  query: string;
  compare: boolean;
  activeMode: 'decision' | 'tool' | 'express' | 'direct';
  stage1: BlockInputStage;
  candidates: CandidateTrace[];
}

/**
 * Result returned by `A2UIClient.sendTurn(...)`.
 */
export interface A2UIClientTurnResult {
  messages: unknown[];
  trace: PlaygroundTurnTrace | null;
  contextId: string | null;
}
