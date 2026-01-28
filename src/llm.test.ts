/**
 * llm.test.ts - Unit tests for the OpenRouter-backed LLM abstraction
 *
 * Run with: bun test src/llm.test.ts
 */

import { describe, test, expect } from "bun:test";
import { OpenRouterLLM, type RerankDocument } from "./llm.js";

function createMockFetch() {
  return async (url: string, init?: RequestInit): Promise<Response> => {
    const body = init?.body ? JSON.parse(init.body.toString()) : {};

    if (url.endsWith("/embeddings")) {
      if (Array.isArray(body.input)) {
        const data = body.input.map((_: string, index: number) => ({
          embedding: [index, index + 1, index + 2],
          index,
        }));
        return new Response(JSON.stringify({ data }), { status: 200 });
      }
      return new Response(JSON.stringify({ data: [{ embedding: [0.1, 0.2, 0.3] }] }), { status: 200 });
    }

    if (url.endsWith("/chat/completions")) {
      const content = JSON.stringify({
        lex: ["auth"],
        vec: ["authentication flow"],
        hyde: ["This document describes authentication flows."],
      });
      return new Response(JSON.stringify({ choices: [{ message: { content } }] }), { status: 200 });
    }

    return new Response(JSON.stringify({ message: "Not found" }), { status: 404 });
  };
}

function createLLM() {
  return new OpenRouterLLM({
    apiKey: "test-key",
    fetch: createMockFetch(),
  });
}

describe("OpenRouterLLM", () => {
  test("embed returns an embedding", async () => {
    const llm = createLLM();
    const result = await llm.embed("hello");
    expect(result).not.toBeNull();
    expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
  });

  test("embedBatch returns embeddings with indices", async () => {
    const llm = createLLM();
    const results = await llm.embedBatch(["a", "b", "c"]);
    expect(results).toHaveLength(3);
    expect(results[0]!.embedding).toEqual([0, 1, 2]);
    expect(results[1]!.embedding).toEqual([1, 2, 3]);
    expect(results[2]!.embedding).toEqual([2, 3, 4]);
  });

  test("expandQuery returns queryables", async () => {
    const llm = createLLM();
    const result = await llm.expandQuery("auth");
    const types = result.map(r => r.type);
    expect(types).toContain("lex");
    expect(types).toContain("vec");
    expect(types).toContain("hyde");
  });

  test("rerank returns scored results", async () => {
    const llm = createLLM();
    const docs: RerankDocument[] = [
      { file: "a.md", text: "auth" },
      { file: "b.md", text: "other" },
    ];
    const result = await llm.rerank("auth", docs);
    expect(result.results).toHaveLength(2);
    expect(result.results[0]!.score).toBeGreaterThanOrEqual(0);
  });
});
