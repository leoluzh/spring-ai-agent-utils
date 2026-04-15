/*
 * Copyright 2025 - 2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 */
package org.springaicommunity.agent.tools.extended;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.util.json.JsonParser;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

/**
 * Parallel AI Web Tool for Spring AI.
 * <p>
 * Provides four complementary capabilities using the Parallel Web Systems API —
 * the only search infrastructure built from the ground up for AI agents:
 * <ul>
 *   <li><b>Search</b>   — Semantic objective-driven web search with token-dense excerpts</li>
 *   <li><b>Extract</b>  — Objective-focused content extraction from known URLs</li>
 *   <li><b>Task</b>     — Deep async research with structured JSON output and citations</li>
 *   <li><b>FindAll</b>  — Entity discovery from the web via natural language match criteria</li>
 * </ul>
 *
 * <p>Parallel indexes billions of pages and updates millions daily. Unlike keyword-based
 * APIs, it accepts <b>semantic objectives</b> and returns <b>token-relevant excerpts</b>
 * compressed for LLM context windows — not snippets designed for human clicks.
 *
 * <p><b>Free tier:</b> up to 16,000 free search requests. $0.005/request (10 results).
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public ParallelWebTool parallelWebTool(
 *         @Value("${parallel.api.key}") String apiKey) {
 *     return ParallelWebTool.builder(apiKey)
 *         .maxResults(10)
 *         .searchMode(ParallelWebTool.SearchMode.AGENTIC)
 *         .build();
 * }
 * }</pre>
 *
 * <h3>application.yaml</h3>
 * <pre>
 * parallel:
 *   api:
 *     key: ${PARALLEL_API_KEY}
 * </pre>
 *
 * @author Spring AI Community
 * @see <a href="https://docs.parallel.ai">Parallel AI Documentation</a>
 */
public class ParallelWebTool {

    private static final Logger logger = LoggerFactory.getLogger(ParallelWebTool.class);

    private static final String BASE_URL         = "https://api.parallel.ai";
    private static final String SEARCH_PATH      = "/v1beta/search";
    private static final String EXTRACT_PATH     = "/v1beta/extract";
    private static final String TASK_RUN_PATH    = "/v1/tasks/runs";
    private static final String TASK_STATUS_PATH = "/v1/tasks/runs/{runId}";
    private static final String FINDALL_PATH     = "/v1beta/findall/runs";
    private static final String FINDALL_STATUS   = "/v1beta/findall/runs/{runId}";

    // Beta headers required by Parallel's versioned endpoints
    private static final String BETA_SEARCH  = "search-extract-2025-10-10";
    private static final String BETA_FINDALL = "findall-2025-09-15";

    private static final long POLL_INTERVAL_MS  = 5_000;
    private static final int  MAX_POLL_ATTEMPTS = 60;   // 5 min max for tasks

    private final RestClient restClient;
    private final int        maxResults;
    private final String     searchMode;
    private final ObservationRegistry observationRegistry;

    /**
     * Search mode — controls latency vs. result quality trade-off.
     * <ul>
     *   <li>{@code FAST}     — Low latency (~1-2s), token-efficient. Best for agentic loops.</li>
     *   <li>{@code AGENTIC}  — Balanced, concise excerpts for multi-turn agent use.</li>
     *   <li>{@code ONE_SHOT} — Comprehensive results and longer excerpts. Best for single-call Q&A.</li>
     * </ul>
     */
    public enum SearchMode {
        FAST("fast"),
        AGENTIC("agentic"),
        ONE_SHOT("one-shot");

        private final String value;
        SearchMode(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Task processor — matches AI compute to task complexity.
     * <ul>
     *   <li>{@code BASE}  — Fast, low-cost. Good for simple enrichment.</li>
     *   <li>{@code CORE}  — Balanced quality and speed. Default for most tasks.</li>
     *   <li>{@code PRO}   — Deep research, multi-hop reasoning (1-3 min).</li>
     *   <li>{@code ULTRA} — Maximum accuracy for complex research (3-10 min).</li>
     * </ul>
     */
    public enum Processor {
        BASE("base"),
        CORE("core"),
        PRO("pro"),
        ULTRA("ultra");

        private final String value;
        Processor(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * FindAll generator tier — matches compute to entity discovery complexity.
     */
    public enum FindAllGenerator {
        BASE("base"),
        CORE("core"),
        PRO("pro");

        private final String value;
        FindAllGenerator(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private ParallelWebTool(String apiKey, int maxResults, SearchMode searchMode,
                            ObservationRegistry observationRegistry) {
        Assert.hasText(apiKey, "Parallel API key must not be null or empty");
        this.restClient = RestClient.builder()
                .baseUrl(BASE_URL)
                .defaultHeader("x-api-key",    apiKey)
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("Accept",       MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.maxResults = maxResults;
        this.searchMode = searchMode.getValue();
        this.observationRegistry = observationRegistry;
    }

    // =========================================================================
    // 1. SEARCH — Semantic objective-driven web search
    // =========================================================================

    @Tool(name = "ParallelSearch", description = """
        Searches the web using Parallel's AI-native search engine.
        Unlike keyword APIs, accepts a semantic OBJECTIVE describing what the agent
        needs to accomplish, plus optional keyword queries for multi-angle coverage.

        Returns LLM-ready ranked URLs with token-dense excerpts — compressed for
        reasoning quality, not human browsing. Reduces round-trips vs. traditional search.

        Modes:
        - fast     → lowest latency (~1-2s), best for agentic loops
        - agentic  → balanced, concise excerpts (default)
        - one-shot → comprehensive excerpts for single-call answers

        Usage notes:
        - Write objective as a goal statement: "Find recent papers on RAG for code"
        - Add search_queries for multi-angle coverage of the same topic
        - Use includeDomains to restrict to authoritative sources
        - Use excludeDomains to filter low-quality or irrelevant domains
        - maxCharsPerResult controls excerpt density (default 3000, max 100000)
        - Rate limit: 600 requests/min. Free tier: 16,000 requests.
        - Price: $0.005 per request (10 results) + $0.001 per page extracted.

        After responding, include a "Sources:" section with links from the results.
        """)
    public String search(
            @ToolParam(description = "Semantic objective — what the agent is trying to accomplish. E.g. 'Find the latest benchmarks comparing RAG retrieval strategies'") String objective,
            @ToolParam(description = "Optional keyword search queries to expand coverage. E.g. ['RAG benchmarks 2025', 'retrieval augmented generation evaluation']", required = false) List<String> searchQueries,
            @ToolParam(description = "Search mode: fast, agentic, one-shot. Null = configured default.", required = false) String mode,
            @ToolParam(description = "Max results to return (null = configured default, max 20)", required = false) Integer maxResults,
            @ToolParam(description = "Max characters per result excerpt (null = 3000)", required = false) Integer maxCharsPerResult,
            @ToolParam(description = "Only include results from these domains (e.g. ['arxiv.org', 'nature.com'])", required = false) List<String> includeDomains,
            @ToolParam(description = "Exclude results from these domains", required = false) List<String> excludeDomains,
            @ToolParam(description = "Only include results published after this date (ISO 8601, e.g. 2025-01-01)", required = false) String publishedAfter) {

        if (!StringUtils.hasText(objective)) {
            logger.warn("Empty objective provided to ParallelSearch");
            return JsonParser.toJson(Collections.emptyList());
        }

        int    effectiveMax    = (maxResults != null && maxResults > 0) ? maxResults : this.maxResults;
        String effectiveMode   = StringUtils.hasText(mode) ? mode : this.searchMode;
        int    effectiveChars  = (maxCharsPerResult != null && maxCharsPerResult > 0) ? maxCharsPerResult : 3000;

        logger.debug("ParallelSearch: '{}' (mode={}, max={})", objective, effectiveMode, effectiveMax);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("objective",   objective);
            body.put("mode",        effectiveMode);
            body.put("max_results", effectiveMax);
            body.put("excerpts",    Map.of("max_chars_per_result", effectiveChars));

            if (!CollectionUtils.isEmpty(searchQueries)) body.put("search_queries", searchQueries);

            // Source policy
            if (!CollectionUtils.isEmpty(includeDomains) || !CollectionUtils.isEmpty(excludeDomains)
                    || StringUtils.hasText(publishedAfter)) {
                var sourcePolicy = new java.util.HashMap<String, Object>();
                if (!CollectionUtils.isEmpty(includeDomains)) sourcePolicy.put("include_domains", includeDomains);
                if (!CollectionUtils.isEmpty(excludeDomains)) sourcePolicy.put("exclude_domains", excludeDomains);
                if (StringUtils.hasText(publishedAfter))      sourcePolicy.put("published_after",  publishedAfter);
                body.put("source_policy", sourcePolicy);
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(SEARCH_PATH)
                    .header("parallel-beta", BETA_SEARCH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in ParallelSearch '{}': {}", objective, res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in ParallelSearch '{}': {}", objective, res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from ParallelSearch for: {}", objective);
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            List<SearchResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> {
                        @SuppressWarnings("unchecked")
                        List<String> excerpts = (List<String>) r.getOrDefault("excerpts", Collections.emptyList());
                        return new SearchResult(
                                (String) r.getOrDefault("title",        ""),
                                (String) r.get("url"),
                                (String) r.getOrDefault("publish_date", ""),
                                excerpts
                        );
                    })
                    .toList();

            logger.debug("ParallelSearch '{}' returned {} results", objective, results.size());
            return JsonParser.toJson(results);

        } catch (RestClientException e) {
            logger.error("Error in ParallelSearch for '{}': {}", objective, e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 2. EXTRACT — Objective-focused content extraction from URLs
    // =========================================================================

    @Tool(name = "ParallelExtract", description = """
        Extracts content from known URLs, focused on a specific objective.
        Returns token-dense excerpts or full page content, optimized for LLM consumption.
        Handles JavaScript-heavy pages, PDFs, and complex web content.

        Use when you already have URLs and need their content focused on a topic —
        more targeted than running a full search.

        Typical workflow:
        1. ParallelSearch → discover relevant URLs
        2. ParallelExtract → get deep, objective-focused content from top URLs

        Usage notes:
        - objective guides which parts of the page to prioritize in excerpts
        - fullContent=true returns the entire page (use for shorter pages)
        - maxCharsPerResult controls excerpt length (min 1000, max 100000)
        - By default uses cached index content; set liveFetch=true for freshest data
        - $0.001 per page extracted
        """)
    public String extract(
            @ToolParam(description = "List of URLs to extract content from") List<String> urls,
            @ToolParam(description = "Objective guiding which parts of pages to prioritize in excerpts", required = false) String objective,
            @ToolParam(description = "If true, returns full page content instead of focused excerpts", required = false) Boolean fullContent,
            @ToolParam(description = "Max characters per result excerpt", required = false) Integer maxCharsPerResult,
            @ToolParam(description = "If true, fetches live content instead of cached index", required = false) Boolean liveFetch) {

        if (CollectionUtils.isEmpty(urls)) {
            logger.warn("Empty URL list provided to ParallelExtract");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.debug("ParallelExtract: {} URLs (objective='{}')", urls.size(), objective);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("urls", urls);

            if (StringUtils.hasText(objective))              body.put("objective",     objective);
            if (Boolean.TRUE.equals(fullContent))            body.put("full_content",  true);

            var excerptOpts = new java.util.HashMap<String, Object>();
            if (maxCharsPerResult != null && maxCharsPerResult > 0) {
                excerptOpts.put("max_chars_per_result", maxCharsPerResult);
            }
            if (!excerptOpts.isEmpty() || !Boolean.TRUE.equals(fullContent)) {
                body.put("excerpts", excerptOpts.isEmpty() ? true : excerptOpts);
            }

            if (Boolean.TRUE.equals(liveFetch)) {
                body.put("fetch_policy", Map.of("max_age_seconds", 60));
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(EXTRACT_PATH)
                    .header("parallel-beta", BETA_SEARCH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in ParallelExtract: {}", res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in ParallelExtract: {}", res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from ParallelExtract");
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawErrors =
                    (List<Map<String, Object>>) response.getOrDefault("errors", Collections.emptyList());

            List<ExtractResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> {
                        @SuppressWarnings("unchecked")
                        List<String> excerpts = (List<String>) r.getOrDefault("excerpts", Collections.emptyList());
                        return new ExtractResult(
                                (String) r.get("url"),
                                (String) r.getOrDefault("title",        ""),
                                (String) r.getOrDefault("publish_date", ""),
                                excerpts,
                                (String) r.getOrDefault("full_content", "")
                        );
                    })
                    .toList();

            List<ExtractError> errors = rawErrors.stream()
                    .filter(e -> e != null && e.get("url") != null)
                    .map(e -> new ExtractError(
                            (String) e.get("url"),
                            (String) e.getOrDefault("error_type", ""),
                            toInt(e.getOrDefault("http_status_code", 0))
                    ))
                    .toList();

            if (!errors.isEmpty()) {
                logger.warn("ParallelExtract: {} URLs failed to extract", errors.size());
            }

            ExtractResponse finalResponse = new ExtractResponse(results, errors);
            logger.debug("ParallelExtract: {} succeeded, {} failed", results.size(), errors.size());
            return JsonParser.toJson(finalResponse);

        } catch (RestClientException e) {
            logger.error("Error in ParallelExtract: {}", e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 3. TASK — Deep async research with structured output and citations
    // =========================================================================

    @Tool(name = "ParallelTask", description = """
        Runs deep async research using Parallel's Task API — multi-hop web search
        with LLM reasoning and structured output with citations (Basis framework).

        Every output includes Parallel's Basis: citations, rationale, and confidence levels.

        Processors (cost vs. accuracy trade-off):
        - base  → fast, low-cost. Simple enrichment and lookups.
        - core  → balanced. Most research tasks (default).
        - pro   → deep research, multi-hop (1-3 min). Complex analysis.
        - ultra → maximum accuracy (3-10 min). Enterprise research.

        Use for:
        - Company research and enrichment ("Extract CEO profile and recent news")
        - Competitive analysis across multiple sources
        - Structured data extraction with JSON output schema
        - Multi-hop questions requiring synthesis across many pages

        Usage notes:
        - ASYNC: polls until complete — may take 1-10 minutes for pro/ultra
        - Specify outputSchema for structured JSON output
        - Use previousInteractionId for follow-up questions in the same context
        - Benchmark: pro achieves 62% on BrowseComp vs 8% for Perplexity
        """)
    public String task(
            @ToolParam(description = "Research task, question, or instruction for the agent") String input,
            @ToolParam(description = "Processor tier: base, core, pro, ultra. Null = core.", required = false) String processor,
            @ToolParam(description = "JSON schema string for structured output. Null = plain text output.", required = false) String outputJsonSchema,
            @ToolParam(description = "Previous interaction ID for follow-up questions in the same context", required = false) String previousInteractionId) {

        if (!StringUtils.hasText(input)) {
            logger.warn("Empty input provided to ParallelTask");
            return JsonParser.toJson(Collections.emptyMap());
        }

        String effectiveProcessor = StringUtils.hasText(processor) ? processor : Processor.CORE.getValue();
        logger.info("ParallelTask: '{}' (processor={})", input, effectiveProcessor);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("input",     input);
            body.put("processor", effectiveProcessor);

            // Output schema
            var taskSpec = new java.util.HashMap<String, Object>();
            if (StringUtils.hasText(outputJsonSchema)) {
                try {
                    // Parse the schema string into a Map for proper JSON serialization
                    @SuppressWarnings("unchecked")
                    Map<String, Object> schemaMap = JsonParser.fromJson(outputJsonSchema, Map.class);
                    taskSpec.put("output_schema", Map.of("type", "json", "json_schema", schemaMap));
                } catch (Exception ex) {
                    logger.warn("Failed to parse outputJsonSchema, using text output: {}", ex.getMessage());
                    taskSpec.put("output_schema", Map.of("type", "text"));
                }
            } else {
                taskSpec.put("output_schema", Map.of("type", "text"));
            }
            body.put("task_spec", taskSpec);

            if (StringUtils.hasText(previousInteractionId)) {
                body.put("previous_interaction_id", previousInteractionId);
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> startResponse = restClient.post()
                    .uri(TASK_RUN_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(s -> s.is4xxClientError(), (req, res) ->
                            logger.error("4xx error starting ParallelTask '{}': {}", input, res.getStatusCode()))
                    .onStatus(s -> s.is5xxServerError(), (req, res) ->
                            logger.error("5xx error starting ParallelTask '{}': {}", input, res.getStatusCode()))
                    .body(Map.class);

            if (startResponse == null) {
                logger.error("Null response starting ParallelTask for: {}", input);
                return JsonParser.toJson(Collections.emptyMap());
            }

            String runId         = (String) startResponse.get("run_id");
            String interactionId = (String) startResponse.getOrDefault("interaction_id", "");

            if (!StringUtils.hasText(runId)) {
                logger.error("No run_id returned by ParallelTask for: {}", input);
                return JsonParser.toJson(Collections.emptyMap());
            }

            logger.debug("ParallelTask run started: runId={}, interactionId={}", runId, interactionId);
            return pollTaskResult(runId, interactionId, input);

        } catch (RestClientException e) {
            logger.error("Error starting ParallelTask for '{}': {}", input, e.getMessage());
            return JsonParser.toJson(Collections.emptyMap());
        }
    }

    // =========================================================================
    // 4. FINDALL — Entity discovery via natural language match criteria
    // =========================================================================

    @Tool(name = "ParallelFindAll", description = """
        Discovers entities from the web based on natural language match conditions.
        Turns a description of what you're looking for into a structured dataset
        with citations — like a custom database query against the entire web.

        Use for:
        - Finding companies matching specific criteria ("B2B SaaS with 50-200 employees in Berlin")
        - Discovering people by role and context ("CTOs at Series A AI startups")
        - Building datasets ("All dental practices in Ohio with 4+ Google stars")
        - Lead generation, market research, competitive intelligence

        Generators (recall vs. cost):
        - base → fast discovery, lower recall (~30%)
        - core → balanced (52% recall) — default
        - pro  → highest recall (61%), best for comprehensive datasets

        Usage notes:
        - ASYNC: polls until complete — may take several minutes
        - matchConditions are evaluated as AND — all must be true for a match
        - matchLimit caps the number of returned entities
        - Achieves 3x better recall than Exa and OpenAI Deep Research
        """)
    public String findAll(
            @ToolParam(description = "Overall objective describing what entities to find. E.g. 'Find all dental practices in Ohio with 4+ Google star rating'") String objective,
            @ToolParam(description = "Entity type to discover: companies, people, startups, locations, etc.") String entityType,
            @ToolParam(description = "List of match conditions (all must be true). Each condition needs a name and a natural language description.") List<Map<String, String>> matchConditions,
            @ToolParam(description = "Generator tier: base, core, pro. Null = core.", required = false) String generator,
            @ToolParam(description = "Maximum number of entities to return (null = no limit)", required = false) Integer matchLimit) {

        if (!StringUtils.hasText(objective) || !StringUtils.hasText(entityType)) {
            logger.warn("Missing objective or entityType in ParallelFindAll");
            return JsonParser.toJson(Collections.emptyList());
        }
        if (CollectionUtils.isEmpty(matchConditions)) {
            logger.warn("No matchConditions provided to ParallelFindAll");
            return JsonParser.toJson(Collections.emptyList());
        }

        String effectiveGenerator = StringUtils.hasText(generator) ? generator : FindAllGenerator.CORE.getValue();
        logger.info("ParallelFindAll: '{}' entityType={} (generator={})",
                objective, entityType, effectiveGenerator);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("objective",        objective);
            body.put("entity_type",      entityType);
            body.put("match_conditions", matchConditions);
            body.put("generator",        effectiveGenerator);
            if (matchLimit != null && matchLimit > 0) body.put("match_limit", matchLimit);

            @SuppressWarnings("unchecked")
            Map<String, Object> startResponse = restClient.post()
                    .uri(FINDALL_PATH)
                    .header("parallel-beta", BETA_FINDALL)
                    .body(body)
                    .retrieve()
                    .onStatus(s -> s.is4xxClientError(), (req, res) ->
                            logger.error("4xx error starting ParallelFindAll '{}': {}", objective, res.getStatusCode()))
                    .onStatus(s -> s.is5xxServerError(), (req, res) ->
                            logger.error("5xx error starting ParallelFindAll '{}': {}", objective, res.getStatusCode()))
                    .body(Map.class);

            if (startResponse == null) {
                logger.error("Null response starting ParallelFindAll for: {}", objective);
                return JsonParser.toJson(Collections.emptyList());
            }

            String runId = (String) startResponse.get("run_id");
            if (!StringUtils.hasText(runId)) {
                logger.error("No run_id returned by ParallelFindAll for: {}", objective);
                return JsonParser.toJson(Collections.emptyList());
            }

            logger.debug("ParallelFindAll run started: runId={}", runId);
            return pollFindAllResult(runId, objective);

        } catch (RestClientException e) {
            logger.error("Error starting ParallelFindAll for '{}': {}", objective, e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // Polling helpers
    // =========================================================================

    private String pollTaskResult(String runId, String interactionId, String input) {
        for (int attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
            try {
                Thread.sleep(POLL_INTERVAL_MS);

                @SuppressWarnings("unchecked")
                Map<String, Object> status = restClient.get()
                        .uri(TASK_STATUS_PATH, runId)
                        .retrieve()
                        .body(Map.class);

                if (status == null) continue;

                String taskStatus = (String) status.get("status");
                logger.debug("ParallelTask {} — status: {} ({}/{})",
                        runId, taskStatus, attempt, MAX_POLL_ATTEMPTS);

                if ("completed".equals(taskStatus)) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> output =
                            (Map<String, Object>) status.getOrDefault("output", Collections.emptyMap());

                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> rawCitations =
                            (List<Map<String, Object>>) status.getOrDefault("citations", Collections.emptyList());

                    List<Citation> citations = rawCitations.stream()
                            .filter(c -> c != null && c.get("url") != null)
                            .map(c -> new Citation(
                                    (String) c.getOrDefault("title", ""),
                                    (String) c.get("url"),
                                    (String) c.getOrDefault("confidence", "")
                            ))
                            .toList();

                    TaskResult result = new TaskResult(
                            String.valueOf(output.getOrDefault("text", output)),
                            citations,
                            interactionId,
                            runId
                    );

                    logger.info("ParallelTask '{}' completed with {} citations", input, citations.size());
                    return JsonParser.toJson(result);
                }

                if ("failed".equals(taskStatus) || "error".equals(taskStatus)) {
                    logger.error("ParallelTask {} failed for: {}", runId, input);
                    return JsonParser.toJson(Collections.emptyMap());
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("ParallelTask polling interrupted for runId: {}", runId);
                return JsonParser.toJson(Collections.emptyMap());
            } catch (RestClientException e) {
                logger.warn("Error polling ParallelTask {}: {}", runId, e.getMessage());
            }
        }

        logger.warn("ParallelTask {} timed out after {} attempts", runId, MAX_POLL_ATTEMPTS);
        return JsonParser.toJson(Collections.emptyMap());
    }

    private String pollFindAllResult(String runId, String objective) {
        for (int attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
            try {
                Thread.sleep(POLL_INTERVAL_MS);

                @SuppressWarnings("unchecked")
                Map<String, Object> status = restClient.get()
                        .uri(FINDALL_STATUS, runId)
                        .header("parallel-beta", BETA_FINDALL)
                        .retrieve()
                        .body(Map.class);

                if (status == null) continue;

                String findStatus = (String) status.get("status");
                logger.debug("ParallelFindAll {} — status: {} ({}/{})",
                        runId, findStatus, attempt, MAX_POLL_ATTEMPTS);

                if ("completed".equals(findStatus)) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> rawMatches =
                            (List<Map<String, Object>>) status.getOrDefault("matches", Collections.emptyList());

                    List<FindAllMatch> matches = rawMatches.stream()
                            .filter(m -> m != null)
                            .map(m -> {
                                @SuppressWarnings("unchecked")
                                List<String> sources = (List<String>) m.getOrDefault("sources", Collections.emptyList());
                                @SuppressWarnings("unchecked")
                                Map<String, Object> fields =
                                        (Map<String, Object>) m.getOrDefault("fields", Collections.emptyMap());
                                return new FindAllMatch(
                                        (String) m.getOrDefault("name",        ""),
                                        (String) m.getOrDefault("description", ""),
                                        fields,
                                        sources
                                );
                            })
                            .toList();

                    logger.info("ParallelFindAll '{}' completed with {} matches", objective, matches.size());
                    return JsonParser.toJson(matches);
                }

                if ("failed".equals(findStatus)) {
                    logger.error("ParallelFindAll {} failed for: {}", runId, objective);
                    return JsonParser.toJson(Collections.emptyList());
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("ParallelFindAll polling interrupted for runId: {}", runId);
                return JsonParser.toJson(Collections.emptyList());
            } catch (RestClientException e) {
                logger.warn("Error polling ParallelFindAll {}: {}", runId, e.getMessage());
            }
        }

        logger.warn("ParallelFindAll {} timed out after {} attempts", runId, MAX_POLL_ATTEMPTS);
        return JsonParser.toJson(Collections.emptyList());
    }

    private int toInt(Object value) {
        if (value instanceof Number n) return n.intValue();
        return 0;
    }

    // =========================================================================
    // Result records
    // =========================================================================

    /** Web search result with token-dense excerpts. */
    public record SearchResult(
            String title,
            String url,
            String publishDate,
            List<String> excerpts
    ) {}

    /** Extracted content from a single URL. */
    public record ExtractResult(
            String url,
            String title,
            String publishDate,
            List<String> excerpts,
            String fullContent
    ) {}

    /** Wraps extract results and failed URLs. */
    public record ExtractResponse(List<ExtractResult> results, List<ExtractError> errors) {}

    /** A URL that failed during extraction. */
    public record ExtractError(String url, String errorType, int httpStatusCode) {}

    /** Deep research task result with Basis citations. */
    public record TaskResult(
            String output,
            List<Citation> citations,
            String interactionId,   // use for follow-up questions
            String runId
    ) {}

    /** A citation from the Basis framework — includes confidence level. */
    public record Citation(String title, String url, String confidence) {}

    /** A single entity match returned by FindAll. */
    public record FindAllMatch(
            String name,
            String description,
            Map<String, Object> fields,
            List<String> sources
    ) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder(String apiKey) {
        return new Builder(apiKey);
    }

    public static class Builder {

        private final String apiKey;
        private int        maxResults  = 10;
        private SearchMode searchMode  = SearchMode.AGENTIC;
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        private Builder(String apiKey) {
            Assert.hasText(apiKey, "API key must not be null or empty");
            this.apiKey = apiKey;
        }

        /** Default max results for the Search tool (max 20). */
        public Builder maxResults(int maxResults) {
            Assert.isTrue(maxResults > 0 && maxResults <= 20, "maxResults must be between 1 and 20");
            this.maxResults = maxResults;
            return this;
        }

        /** Default search mode. See {@link SearchMode}. */
        public Builder searchMode(SearchMode searchMode) {
            Assert.notNull(searchMode, "searchMode must not be null");
            this.searchMode = searchMode;
            return this;
        }

        /** Observation registry for metrics and tracing. Defaults to {@link ObservationRegistry#NOOP}. */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry must not be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public ParallelWebTool build() {
            return new ParallelWebTool(this.apiKey, this.maxResults, this.searchMode,
                    this.observationRegistry);
        }
    }
}