/**
 * llm.ts - LLM abstraction layer for QMD (local GGUF models + OpenRouter)
 *
 * Provides embeddings, text generation, and reranking via local node-llama-cpp
 * or remote OpenRouter-backed models.
 */

import {
  getLlama,
  resolveModelFile,
  LlamaChatSession,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
  type Token as LlamaToken,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, readFileSync } from "fs";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
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
 * LLM backend selection
 */
export type LLMBackend = "llama" | "openrouter";

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

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

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL_URI = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL_URI = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL_URI = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL_URI = "hf:ggml-org/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf";

// Local model labels used for cache/DB metadata
const DEFAULT_LOCAL_EMBED_MODEL = "embeddinggemma";
const DEFAULT_LOCAL_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
const DEFAULT_LOCAL_QUERY_MODEL = "Qwen/Qwen3-1.7B";

// OpenRouter defaults
const DEFAULT_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small";
const DEFAULT_OPENROUTER_RERANK_MODEL = "openai/gpt-4o-mini";
const DEFAULT_OPENROUTER_GENERATE_MODEL = "openai/gpt-4o-mini";

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");

// =============================================================================
// Backend selection and defaults
// =============================================================================

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

export function resolveLLMBackend(): LLMBackend {
  const raw = (process.env.QMD_LLM_BACKEND || "llama").toLowerCase();
  if (["llama", "local", "llama-cpp", "node-llama-cpp"].includes(raw)) return "llama";
  if (["openrouter", "open-router", "router"].includes(raw)) return "openrouter";
  if (raw === "auto") {
    return resolveOpenRouterApiKey() ? "openrouter" : "llama";
  }
  return resolveOpenRouterApiKey() ? "openrouter" : "llama";
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
  if (resolveLLMBackend() === "openrouter") {
    const defaults = getOpenRouterDefaults();
    return {
      embedModel: defaults.embedModel,
      rerankModel: defaults.rerankMode === "embeddings" ? defaults.embedModel : defaults.rerankModel,
      queryModel: defaults.generateModel,
    };
  }
  return {
    embedModel: DEFAULT_LOCAL_EMBED_MODEL,
    rerankModel: DEFAULT_LOCAL_RERANK_MODEL,
    queryModel: DEFAULT_LOCAL_QUERY_MODEL,
  };
}

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
  tokenize?(text: string): Promise<readonly LlamaToken[]>;
  detokenize?(tokens: readonly LlamaToken[]): Promise<string>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 2 minutes
const DEFAULT_INACTIVITY_TIMEOUT_MS = 2 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContext: LlamaEmbeddingContext | null = null;
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;

  // Ensure we don't load the same model/context concurrently (which can allocate duplicate VRAM).
  private embedModelLoadPromise: Promise<LlamaModel> | null = null;
  private embedContextCreatePromise: Promise<LlamaEmbeddingContext> | null = null;
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;


  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL_URI;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL_URI;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL_URI;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
  }

  /**
   * Reset the inactivity timer. Called after each model operation.
   * When timer fires, models are unloaded to free memory.
   */
  private touchActivity(): void {
    // Clear existing timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Only set timer if we have disposable contexts and timeout is enabled
    if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        this.unloadIdleResources().catch(err => {
          console.error("Error unloading idle resources:", err);
        });
      }, this.inactivityTimeoutMs);
      // Don't keep process alive just for this timer
      this.inactivityTimer.unref();
    }
  }

  /**
   * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
   */
  private hasLoadedContexts(): boolean {
    return !!(this.embedContext || this.rerankContext);
  }

  /**
   * Unload idle resources but keep the instance alive for future use.
   *
   * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
   * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
   */
  async unloadIdleResources(): Promise<void> {
    // Don't unload if already disposed
    if (this.disposed) {
      return;
    }

    // Clear timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Dispose contexts first
    if (this.embedContext) {
      await this.embedContext.dispose();
      this.embedContext = null;
    }
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Optionally dispose models too (opt-in)
    if (this.disposeModelsOnInactivity) {
      if (this.embedModel) {
        await this.embedModel.dispose();
        this.embedModel = null;
      }
      if (this.generateModel) {
        await this.generateModel.dispose();
        this.generateModel = null;
      }
      if (this.rerankModel) {
        await this.rerankModel.dispose();
        this.rerankModel = null;
      }
      // Reset load promises so models can be reloaded later
      this.embedModelLoadPromise = null;
      this.generateModelLoadPromise = null;
      this.rerankModelLoadPromise = null;
    }

    // Note: We keep llama instance alive - it's lightweight
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    if (!this.llama) {
      this.llama = await getLlama({ logLevel: LlamaLogLevel.error });
    }
    return this.llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    // resolveModelFile handles HF URIs and downloads to the cache dir
    return await resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Load embedding model (lazy)
   */
  private async ensureEmbedModel(): Promise<LlamaModel> {
    if (this.embedModel) {
      return this.embedModel;
    }
    if (this.embedModelLoadPromise) {
      return await this.embedModelLoadPromise;
    }

    this.embedModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.embedModelUri);
      const model = await llama.loadModel({ modelPath });
      this.embedModel = model;
      return model;
    })();

    try {
      return await this.embedModelLoadPromise;
    } finally {
      // Keep the resolved model cached; clear only the in-flight promise.
      this.embedModelLoadPromise = null;
    }
  }

  /**
   * Load embedding context (lazy). Context can be disposed and recreated without reloading the model.
   * Uses promise guard to prevent concurrent context creation race condition.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    if (!this.embedContext) {
      // If context creation is already in progress, wait for it
      if (this.embedContextCreatePromise) {
        return await this.embedContextCreatePromise;
      }

      // Start context creation and store promise so concurrent calls wait
      this.embedContextCreatePromise = (async () => {
        const model = await this.ensureEmbedModel();
        const context = await model.createEmbeddingContext();
        this.embedContext = context;
        return context;
      })();

      try {
        await this.embedContextCreatePromise;
      } finally {
        this.embedContextCreatePromise = null;
      }
    }
    this.touchActivity();
    return this.embedContext;
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    if (!this.generateModel) {
      if (this.generateModelLoadPromise) {
        return await this.generateModelLoadPromise;
      }

      this.generateModelLoadPromise = (async () => {
        const llama = await this.ensureLlama();
        const modelPath = await this.resolveModel(this.generateModelUri);
        const model = await llama.loadModel({ modelPath });
        this.generateModel = model;
        return model;
      })();

      try {
        await this.generateModelLoadPromise;
      } finally {
        this.generateModelLoadPromise = null;
      }
    }
    this.touchActivity();
    if (!this.generateModel) {
      throw new Error("Generate model not loaded");
    }
    return this.generateModel;
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    if (this.rerankModel) {
      return this.rerankModel;
    }
    if (this.rerankModelLoadPromise) {
      return await this.rerankModelLoadPromise;
    }

    this.rerankModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      const model = await llama.loadModel({ modelPath });
      this.rerankModel = model;
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
      this.rerankModelLoadPromise = null;
    }
  }

  /**
   * Load rerank context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const model = await this.ensureRerankModel();
      this.rerankContext = await model.createRankingContext();
    }
    this.touchActivity();
    return this.rerankContext;
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    await this.ensureEmbedContext();  // Ensure model is loaded
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.tokenize(text);
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    await this.ensureEmbedContext();
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const context = await this.ensureEmbedContext();
      const embedding = await context.getEmbeddingFor(text);

      return {
        embedding: Array.from(embedding.vector),
        model: this.embedModelUri,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[], _options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      const context = await this.ensureEmbedContext();

      // node-llama-cpp handles batching internally when we make parallel requests
      const embeddings = await Promise.all(
        texts.map(async (text) => {
          try {
            const embedding = await context.getEmbeddingFor(text);
            return {
              embedding: Array.from(embedding.vector),
              model: this.embedModelUri,
            };
          } catch (err) {
            console.error("Embedding error for text:", err);
            return null;
          }
        })
      );

      return embeddings;
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    // Ensure model is loaded
    await this.ensureGenerateModel();

    // Create fresh context -> sequence -> session for each call
    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    const maxTokens = options.maxTokens ?? 150;
    const temperature = options.temperature ?? 0;

    let result = "";
    try {
      await session.prompt(prompt, {
        maxTokens,
        temperature,
        onTextChunk: (text) => {
          result += text;
        },
      });

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
      await context.dispose();
    }
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: ExpandQueryOptions = {}): Promise<Queryable[]> {
    const llama = await this.ensureLlama();
    await this.ensureGenerateModel();

    const includeLexical = options.includeLexical ?? true;
    const context = options.context;

    const grammar = await llama.createGrammar({
      grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
    });

    const prompt = `You are a search query optimization expert. Your task is to improve retrieval by rewriting queries and generating hypothetical documents.

Original Query: ${query}

${context ? `Additional Context, ONLY USE IF RELEVANT:\n\n<context>${context}</context>` : ""}

## Step 1: Query Analysis
Identify entities, search intent, and missing context.

## Step 2: Generate Hypothetical Document
Write a focused sentence passage that would answer the query. Include specific terminology and domain vocabulary.

## Step 3: Query Rewrites
Generate 2-3 alternative search queries that resolve ambiguities. Use terminology from the hypothetical document.

## Step 4: Final Retrieval Text
Output exactly 1-3 'lex' lines, 1-3 'vec' lines, and MAX ONE 'hyde' line.

<format>
lex: {single search term}
vec: {single vector query}
hyde: {complete hypothetical document passage from Step 2 on a SINGLE LINE}
</format>

<example>
Example (FOR FORMAT ONLY - DO NOT COPY THIS CONTENT):
lex: example keyword 1
lex: example keyword 2
vec: example semantic query
hyde: This is an example of a hypothetical document passage that would answer the example query. It contains multiple sentences and relevant vocabulary.
</example>

<rules>
- DO NOT repeat the same line.
- Each 'lex:' line MUST be a different keyword variation based on the ORIGINAL QUERY.
- Each 'vec:' line MUST be a different semantic variation based on the ORIGINAL QUERY.
- The 'hyde:' line MUST be the full sentence passage from Step 2, but all on one line.
- DO NOT use the example content above.
${!includeLexical ? "- Do NOT output any 'lex:' lines" : ""}
</rules>

Final Output:`;

    // Create fresh context for each call
    const genContext = await this.generateModel!.createContext();
    const sequence = genContext.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    try {
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 1000,
        temperature: 1,
      });

      const lines = result.trim().split("\n");
      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        return { type: type as QueryType, text };
      }).filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      if (!includeLexical) {
        return queryables.filter(q => q.type !== 'lex');
      }
      return queryables;
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    } finally {
      await genContext.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    const context = await this.ensureRerankContext();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Use the proper ranking API - returns [{document: string, score: number}] sorted by score
    const ranked = await context.rankAndSort(query, texts);

    // Map back to our result format using the text-to-doc map
    const results: RerankDocumentResult[] = ranked.map((item) => {
      const docInfo = textToDoc.get(item.document)!;
      return {
        file: docInfo.file,
        score: item.score,
        index: docInfo.index,
      };
    });

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  async dispose(): Promise<void> {
    // Prevent double-dispose
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    // Clear inactivity timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Disposing llama cascades to models and contexts automatically
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
    // Note: llama.dispose() can hang indefinitely, so we use a timeout
    if (this.llama) {
      const disposePromise = this.llama.dispose();
      const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 1000));
      await Promise.race([disposePromise, timeoutPromise]);
    }

    // Clear references
    this.embedContext = null;
    this.rerankContext = null;
    this.embedModel = null;
    this.generateModel = null;
    this.rerankModel = null;
    this.llama = null;

    // Clear any in-flight load/create promises
    this.embedModelLoadPromise = null;
    this.embedContextCreatePromise = null;
    this.generateModelLoadPromise = null;
    this.rerankModelLoadPromise = null;
  }
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
    const user = `Original Query: ${query}

${context ? `Additional Context, ONLY USE IF RELEVANT:\n\n<context>${context}</context>\n\n` : ""}
Return JSON with keys:
- "lex": array of 1-3 short lexical keyword queries (can be empty if not requested)
- "vec": array of 1-3 semantic queries
- "hyde": array with at most 1 hypothetical document passage (single line)

Rules:
- Do NOT repeat the same text.
- Each "lex" entry must be a different keyword variation.
- Each "vec" entry must be a different semantic variation.
- The "hyde" entry must be a single line.
${!includeLexical ? '- If "lex" is not requested, return an empty array for "lex".' : ""}
`;

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
    const user = `Query: ${query}

Score each document for relevance to the query.
Return JSON with a "scores" array of length ${documents.length}, where each score is between 0 and 1.

Documents:
${formattedDocs}
`;

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
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp();
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}

// =============================================================================
// Backend-agnostic default LLM
// =============================================================================

let defaultLLM: LLM | null = null;
let defaultLLMBackend: LLMBackend | null = null;

/**
 * Get the default LLM instance based on backend selection.
 */
export function getDefaultLLM(): LLM {
  const backend = resolveLLMBackend();
  if (defaultLLM && defaultLLMBackend === backend) {
    return defaultLLM;
  }

  if (backend === "openrouter") {
    const apiKey = resolveOpenRouterApiKey();
    if (!apiKey) {
      throw new Error(
        "OpenRouter backend selected but no API key found. Set OPENROUTER_API_KEY or add .openrouter-api-key."
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
    defaultLLMBackend = backend;
    return defaultLLM;
  }

  defaultLLM = getDefaultLlamaCpp();
  defaultLLMBackend = backend;
  return defaultLLM;
}

/**
 * Dispose the default LLM instance if it exists.
 */
export async function disposeDefaultLLM(): Promise<void> {
  if (defaultLLM && defaultLLMBackend === "openrouter") {
    await defaultLLM.dispose();
  }
  await disposeDefaultLlamaCpp();
  defaultLLM = null;
  defaultLLMBackend = null;
}
