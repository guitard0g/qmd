/**
 * llm.ts - LLM abstraction layer for QMD (OpenRouter)
 *
 * Provides embeddings, text generation, and reranking via OpenRouter.
 */

import { homedir } from "os";
import { join } from "path";
import { existsSync, readFileSync } from "fs";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Options for query expansion
 */
export type ExpandQueryOptions = {
  context?: string;
  includeLexical?: boolean;
  model?: string;
};

/**
 * Supported query types for different search backends
 */
export type QueryType = "lex" | "vec" | "hyde";

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

/**
 * Token type for optional tokenizer support
 */
export type Token = number;

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Get embeddings for multiple texts
   */
  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: ExpandQueryOptions): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;

  /**
   * Optional tokenizer support for precise chunking.
   */
  tokenize?(text: string): Promise<readonly Token[]>;
  detokenize?(tokens: readonly Token[]): Promise<string>;
}

// =============================================================================
// OpenRouter defaults and configuration
// =============================================================================

const DEFAULT_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small";
const DEFAULT_OPENROUTER_RERANK_MODEL = "openai/gpt-4o-mini";
const DEFAULT_OPENROUTER_GENERATE_MODEL = "openai/gpt-4o-mini";

let cachedOpenRouterKey: string | null | undefined;

function readKeyFile(path: string): string | null {
  try {
    if (!existsSync(path)) return null;
    const key = readFileSync(path, "utf-8").trim();
    return key.length > 0 ? key : null;
  } catch {
    return null;
  }
}

export function resolveOpenRouterApiKey(): string | null {
  if (cachedOpenRouterKey !== undefined) return cachedOpenRouterKey;

  const envKey = process.env.OPENROUTER_API_KEY || process.env.QMD_OPENROUTER_API_KEY;
  if (envKey && envKey.trim().length > 0) {
    cachedOpenRouterKey = envKey.trim();
    return cachedOpenRouterKey;
  }

  const envPath = process.env.OPENROUTER_API_KEY_PATH || process.env.QMD_OPENROUTER_API_KEY_PATH;
  if (envPath) {
    const key = readKeyFile(envPath);
    if (key) {
      cachedOpenRouterKey = key;
      return cachedOpenRouterKey;
    }
  }

  const cwdKey = readKeyFile(join(process.cwd(), ".openrouter-api-key"));
  if (cwdKey) {
    cachedOpenRouterKey = cwdKey;
    return cachedOpenRouterKey;
  }

  const configKey = readKeyFile(join(homedir(), ".config", "qmd", "openrouter-api-key"));
  if (configKey) {
    cachedOpenRouterKey = configKey;
    return cachedOpenRouterKey;
  }

  cachedOpenRouterKey = null;
  return null;
}

export type OpenRouterDefaults = {
  embedModel: string;
  rerankModel: string;
  generateModel: string;
  rerankMode: "embeddings" | "chat";
  baseUrl: string;
  appName: string;
  appUrl: string;
};

export function getOpenRouterDefaults(): OpenRouterDefaults {
  const embedModel = process.env.QMD_OPENROUTER_EMBED_MODEL
    || process.env.OPENROUTER_EMBED_MODEL
    || DEFAULT_OPENROUTER_EMBED_MODEL;
  const rerankModel = process.env.QMD_OPENROUTER_RERANK_MODEL
    || process.env.OPENROUTER_RERANK_MODEL
    || DEFAULT_OPENROUTER_RERANK_MODEL;
  const generateModel = process.env.QMD_OPENROUTER_GENERATE_MODEL
    || process.env.OPENROUTER_GENERATE_MODEL
    || DEFAULT_OPENROUTER_GENERATE_MODEL;
  const rerankModeRaw = (process.env.QMD_OPENROUTER_RERANK_MODE || "embeddings").toLowerCase();
  const rerankMode = rerankModeRaw === "chat" ? "chat" : "embeddings";
  const baseUrl = process.env.QMD_OPENROUTER_BASE_URL
    || process.env.OPENROUTER_BASE_URL
    || "https://openrouter.ai/api/v1";
  const appName = process.env.QMD_OPENROUTER_APP_NAME || "qmd";
  const appUrl = process.env.QMD_OPENROUTER_APP_URL || "https://github.com/tobi/qmd";
  return { embedModel, rerankModel, generateModel, rerankMode, baseUrl, appName, appUrl };
}

export function getDefaultModelConfig(): { embedModel: string; rerankModel: string; queryModel: string } {
  const defaults = getOpenRouterDefaults();
  return {
    embedModel: defaults.embedModel,
    rerankModel: defaults.rerankMode === "embeddings" ? defaults.embedModel : defaults.rerankModel,
    queryModel: defaults.generateModel,
  };
}

// =============================================================================
// OpenRouter Implementation (OpenAI-compatible HTTP API)
// =============================================================================

type OpenRouterEmbeddingResponse = {
  data: { embedding: number[]; index?: number }[];
  model?: string;
};

type OpenRouterChatResponse = {
  choices: { message?: { content?: string } }[];
  model?: string;
};

export type OpenRouterLLMConfig = {
  apiKey: string;
  baseUrl?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  rerankMode?: "embeddings" | "chat";
  appName?: string;
  appUrl?: string;
  fetch?: typeof fetch;
};

function safeJsonParse(text: string): any | null {
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const av = a[i]!;
    const bv = b[i]!;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class OpenRouterLLM implements LLM {
  private apiKey: string;
  private baseUrl: string;
  private embedModel: string;
  private generateModel: string;
  private rerankModel: string;
  private rerankMode: "embeddings" | "chat";
  private appName: string;
  private appUrl: string;
  private fetcher: typeof fetch;

  constructor(config: OpenRouterLLMConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || "https://openrouter.ai/api/v1";
    this.embedModel = config.embedModel || DEFAULT_OPENROUTER_EMBED_MODEL;
    this.generateModel = config.generateModel || DEFAULT_OPENROUTER_GENERATE_MODEL;
    this.rerankModel = config.rerankModel || DEFAULT_OPENROUTER_RERANK_MODEL;
    this.rerankMode = config.rerankMode || "embeddings";
    this.appName = config.appName || "qmd";
    this.appUrl = config.appUrl || "https://github.com/tobi/qmd";
    this.fetcher = config.fetch || fetch;
  }

  private resolveModel(input: string | undefined, fallback: string): string {
    if (!input) return fallback;
    const trimmed = input.trim();
    if (!trimmed) return fallback;
    if (!trimmed.includes("/")) return fallback;
    return trimmed;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Authorization": `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
    };
    if (this.appUrl) headers["HTTP-Referer"] = this.appUrl;
    if (this.appName) headers["X-Title"] = this.appName;
    return headers;
  }

  private async post<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const base = this.baseUrl.endsWith("/") ? this.baseUrl.slice(0, -1) : this.baseUrl;
    const response = await this.fetcher(`${base}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });

    const text = await response.text();
    if (!response.ok) {
      let message = text;
      const parsed = safeJsonParse(text);
      if (parsed?.error?.message) message = parsed.error.message;
      if (parsed?.message) message = parsed.message;
      throw new Error(`OpenRouter ${response.status} ${response.statusText}: ${message}`);
    }

    try {
      return JSON.parse(text) as T;
    } catch (error) {
      throw new Error(`OpenRouter response parse error: ${error}`);
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const model = this.resolveModel(options.model, this.embedModel);
    const data = await this.post<OpenRouterEmbeddingResponse>("/embeddings", {
      model,
      input: text,
    });
    const embedding = data.data?.[0]?.embedding;
    if (!embedding) return null;
    return { embedding, model };
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const model = this.resolveModel(options.model, this.embedModel);
    const data = await this.post<OpenRouterEmbeddingResponse>("/embeddings", {
      model,
      input: texts,
    });
    const results: (EmbeddingResult | null)[] = Array(texts.length).fill(null);
    const items = data.data || [];
    items.forEach((item, idx) => {
      const index = item.index ?? idx;
      if (index >= 0 && index < results.length) {
        results[index] = { embedding: item.embedding, model };
      }
    });
    return results;
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const model = this.resolveModel(options.model, this.generateModel);
    const maxTokens = options.maxTokens ?? 150;
    const temperature = options.temperature ?? 0;
    const data = await this.post<OpenRouterChatResponse>("/chat/completions", {
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature,
    });
    const text = data.choices?.[0]?.message?.content ?? "";
    return { text, model, done: true };
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async expandQuery(query: string, options: ExpandQueryOptions = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const context = options.context;
    const model = this.resolveModel(options.model, this.generateModel);

    const system = "You are a search query optimization expert.";
    const user = `Original Query: ${query}\n\n${context ? `Additional Context, ONLY USE IF RELEVANT:\n\n<context>${context}</context>\n\n` : ""}Return JSON with keys:\n- \"lex\": array of 1-3 short lexical keyword queries (can be empty if not requested)\n- \"vec\": array of 1-3 semantic queries\n- \"hyde\": array with at most 1 hypothetical document passage (single line)\n\nRules:\n- Do NOT repeat the same text.\n- Each \"lex\" entry must be a different keyword variation.\n- Each \"vec\" entry must be a different semantic variation.\n- The \"hyde\" entry must be a single line.\n${!includeLexical ? '- If "lex" is not requested, return an empty array for "lex".' : ""}\n`;

    try {
      const data = await this.post<OpenRouterChatResponse>("/chat/completions", {
        model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        response_format: { type: "json_object" },
        temperature: 0.7,
        max_tokens: 800,
      });

      const content = data.choices?.[0]?.message?.content ?? "";
      const parsed = safeJsonParse(content) || {};
      const lex: string[] = Array.isArray(parsed.lex) ? parsed.lex : [];
      const vec: string[] = Array.isArray(parsed.vec) ? parsed.vec : [];
      const hyde: string[] = Array.isArray(parsed.hyde) ? parsed.hyde : [];

      const queryables: Queryable[] = [];
      const seen = new Set<string>();
      const pushUnique = (type: QueryType, text: string) => {
        const trimmed = text.trim();
        if (!trimmed) return;
        const key = `${type}:${trimmed}`;
        if (seen.has(key)) return;
        seen.add(key);
        queryables.push({ type, text: trimmed });
      };

      if (includeLexical) {
        for (const text of lex) pushUnique("lex", text);
      }
      for (const text of vec) pushUnique("vec", text);
      for (const text of hyde) pushUnique("hyde", text);

      if (queryables.length === 0 && content.trim().length > 0) {
        const lines = content.trim().split("\n");
        for (const line of lines) {
          const colonIdx = line.indexOf(":");
          if (colonIdx === -1) continue;
          const type = line.slice(0, colonIdx).trim();
          if (type !== "lex" && type !== "vec" && type !== "hyde") continue;
          if (!includeLexical && type === "lex") continue;
          const text = line.slice(colonIdx + 1).trim();
          pushUnique(type as QueryType, text);
        }
      }

      if (queryables.length === 0) {
        const fallback: Queryable[] = [{ type: "vec", text: query }];
        if (includeLexical) fallback.unshift({ type: "lex", text: query });
        return fallback;
      }
      return queryables;
    } catch (error) {
      console.error("OpenRouter query expansion failed:", error);
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  private async rerankWithEmbeddings(
    query: string,
    documents: RerankDocument[],
    model: string
  ): Promise<RerankResult> {
    const queryText = formatQueryForEmbedding(query);
    const docTexts = documents.map(doc => formatDocForEmbedding(doc.text, doc.title));
    const queryEmbedding = await this.embed(queryText, { model });
    const docEmbeddings = await this.embedBatch(docTexts, { model });

    const results: RerankDocumentResult[] = documents.map((doc, index) => {
      const embedding = docEmbeddings[index]?.embedding;
      let score = 0;
      if (embedding && queryEmbedding?.embedding) {
        const cos = cosineSimilarity(queryEmbedding.embedding, embedding);
        score = clamp01((cos + 1) / 2);
      }
      return { file: doc.file, score, index };
    });

    results.sort((a, b) => b.score - a.score);
    return { results, model };
  }

  private async rerankWithChat(
    query: string,
    documents: RerankDocument[],
    model: string
  ): Promise<RerankResult> {
    const MAX_DOC_CHARS = 4000;
    const formattedDocs = documents.map((doc, index) => {
      const trimmed = doc.text.length > MAX_DOC_CHARS ? `${doc.text.slice(0, MAX_DOC_CHARS)}...` : doc.text;
      return `#${index} (${doc.file})\n${trimmed}`;
    }).join("\n\n");

    const system = "You are a relevance scoring system.";
    const user = `Query: ${query}\n\nScore each document for relevance to the query.\nReturn JSON with a \"scores\" array of length ${documents.length}, where each score is between 0 and 1.\n\nDocuments:\n${formattedDocs}\n`;

    try {
      const data = await this.post<OpenRouterChatResponse>("/chat/completions", {
        model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        response_format: { type: "json_object" },
        temperature: 0,
        max_tokens: 1200,
      });

      const content = data.choices?.[0]?.message?.content ?? "";
      const parsed = safeJsonParse(content);
      const scores = Array.isArray(parsed?.scores) ? parsed.scores : null;
      if (!scores || scores.length !== documents.length) {
        return this.rerankWithEmbeddings(query, documents, this.embedModel);
      }

      const results: RerankDocumentResult[] = documents.map((doc, index) => ({
        file: doc.file,
        score: clamp01(Number(scores[index])),
        index,
      }));
      results.sort((a, b) => b.score - a.score);
      return { results, model };
    } catch (error) {
      console.error("OpenRouter chat rerank failed, falling back to embeddings:", error);
      return this.rerankWithEmbeddings(query, documents, this.embedModel);
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel };
    }
    const rerankModel = this.resolveModel(options.model, this.rerankModel);
    if (this.rerankMode === "chat") {
      return this.rerankWithChat(query, documents, rerankModel);
    }
    const embedModel = options.model && options.model.includes("embedding")
      ? this.resolveModel(options.model, this.embedModel)
      : this.embedModel;
    return this.rerankWithEmbeddings(query, documents, embedModel);
  }

  async dispose(): Promise<void> {
    // No native resources to dispose
  }
}

// =============================================================================
// Default LLM (OpenRouter only)
// =============================================================================

let defaultLLM: LLM | null = null;

export function setDefaultLLM(llm: LLM | null): void {
  defaultLLM = llm;
}

export function getDefaultLLM(): LLM {
  if (defaultLLM) return defaultLLM;

  const apiKey = resolveOpenRouterApiKey();
  if (!apiKey) {
    throw new Error(
      "OpenRouter API key not found. Set OPENROUTER_API_KEY or add .openrouter-api-key."
    );
  }
  const defaults = getOpenRouterDefaults();
  defaultLLM = new OpenRouterLLM({
    apiKey,
    baseUrl: defaults.baseUrl,
    embedModel: defaults.embedModel,
    generateModel: defaults.generateModel,
    rerankModel: defaults.rerankModel,
    rerankMode: defaults.rerankMode,
    appName: defaults.appName,
    appUrl: defaults.appUrl,
  });
  return defaultLLM;
}

export async function disposeDefaultLLM(): Promise<void> {
  if (defaultLLM) {
    await defaultLLM.dispose();
  }
  defaultLLM = null;
}
