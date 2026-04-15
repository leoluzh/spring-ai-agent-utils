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

import io.micrometer.observation.Observation;
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
 * LLMLayer Web Tool for Spring AI.
 * <p>
 * Provides seven complementary capabilities using the LLMLayer API v2 —
 * a unified web toolkit for AI agents with zero model markup:
 * <ul>
 *   <li><b>Answer</b>   — LLM-grounded answer using any model (20+ supported)</li>
 *   <li><b>SearchWeb</b>— Multi-vertical web search (general, news, scholar, shopping…)</li>
 *   <li><b>Scrape</b>   — Clean content from any URL (markdown, html, screenshot, pdf)</li>
 *   <li><b>Crawl</b>    — Recursive site traversal with per-page content extraction</li>
 *   <li><b>Map</b>      — Discover all URLs on a site without extracting content</li>
 *   <li><b>Pdf</b>      — Extract text and metadata from any PDF URL</li>
 *   <li><b>YouTube</b>  — Retrieve transcripts and metadata from YouTube videos</li>
 * </ul>
 *
 * <p><b>Pricing:</b> $0.004/search + model cost at provider rate (no markup).
 * Starts with $2 free credits.
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public LLMLayerWebTool llmLayerWebTool(
 *         @Value("${llmlayer.api.key}") String apiKey) {
 *     return LLMLayerWebTool.builder(apiKey)
 *         .defaultModel("openai/gpt-4o-mini")
 *         .defaultLocation("us")
 *         .build();
 * }
 * }</pre>
 *
 * <h3>application.yaml</h3>
 * <pre>
 * llmlayer:
 *   api:
 *     key: ${LLMLAYER_API_KEY}
 * </pre>
 *
 * @author Spring AI Community
 * @see <a href="https://docs.llmlayer.ai">LLMLayer API Documentation</a>
 */
public class LLMLayerWebTool {

    private static final Logger logger = LoggerFactory.getLogger(LLMLayerWebTool.class);

    private static final String BASE_URL       = "https://api.llmlayer.ai";
    private static final String ANSWER_PATH    = "/api/v2/answer";
    private static final String SEARCH_PATH    = "/api/v2/search_web";
    private static final String SCRAPE_PATH    = "/api/v2/scrape";
    private static final String CRAWL_PATH     = "/api/v2/crawl";
    private static final String MAP_PATH       = "/api/v2/map";
    private static final String PDF_PATH       = "/api/v2/get_pdf_content";
    private static final String YOUTUBE_PATH   = "/api/v2/get_youtube_transcript";

    private final RestClient restClient;
    private final String     defaultModel;
    private final String     defaultLocation;
    private ObservationRegistry observationRegistry;

    /**
     * Supported LLM models — passed through at cost with no markup.
     * Format: provider/model-name
     */
    public enum Model {
        GPT_4O_MINI("openai/gpt-4o-mini"),
        GPT_4O("openai/gpt-4o"),
        GPT_41_MINI("openai/gpt-4.1-mini"),
        GPT_41("openai/gpt-4.1"),
        O3("openai/o3"),
        O4_MINI("openai/o4-mini"),
        CLAUDE_SONNET_4("anthropic/claude-sonnet-4"),
        CLAUDE_HAIKU("anthropic/claude-haiku-4-5"),
        LLAMA_3_3_70B("groq/llama-3.3-70b-versatile"),
        KIMI_K2("groq/kimi-k2"),
        DEEPSEEK_REASONER("deepseek/deepseek-reasoner"),
        DEEPSEEK_CHAT("deepseek/deepseek-chat");

        private final String value;
        Model(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Search verticals supported by the SearchWeb tool.
     */
    public enum SearchType {
        GENERAL("general"),
        NEWS("news"),
        SCHOLAR("scholar"),
        SHOPPING("shopping"),
        VIDEOS("videos"),
        IMAGES("images");

        private final String value;
        SearchType(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Recency filters for time-sensitive searches.
     */
    public enum Recency {
        HOUR("hour"),
        DAY("day"),
        WEEK("week"),
        MONTH("month"),
        YEAR("year");

        private final String value;
        Recency(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private LLMLayerWebTool(String apiKey, String defaultModel, String defaultLocation,
                            ObservationRegistry observationRegistry) {
        Assert.hasText(apiKey, "LLMLayer API key must not be null or empty");
        this.restClient = RestClient.builder()
                .baseUrl(BASE_URL)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .defaultHeader("Content-Type",  MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("Accept",        MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.defaultModel    = defaultModel;
        this.defaultLocation = defaultLocation;
        this.observationRegistry = observationRegistry;
    }

    // =========================================================================
    // 1. ANSWER — LLM-grounded answer with web search
    // =========================================================================

    @Tool(name = "LLMLayerAnswer", description = """
        Generates an LLM answer grounded in real-time web search results.
        Supports 20+ models (OpenAI, Anthropic, Groq, DeepSeek) with zero markup —
        you pay only $0.004 per search plus the provider's token rate.

        Use for:
        - Questions requiring current information beyond training cutoff
        - News, events, prices, or any time-sensitive facts
        - Research questions needing synthesis across multiple sources
        - Structured JSON answers (set answerType=json with jsonSchema)

        Usage notes:
        - Specify model as "provider/model-name" (e.g. "openai/gpt-4o-mini")
        - Use searchType=news for current events, scholar for research papers
        - Use domainFilter with "-domain.com" prefix to EXCLUDE a domain
        - citations=true embeds inline citation markers in the answer
        - maxQueries (1-5) expands coverage at the cost of additional search fees
        - Always include Sources from the response in your final reply
        """)
    public String answer(
            @ToolParam(description = "Question or instruction for the LLM") String query,
            @ToolParam(description = "LLM model id, e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4, groq/llama-3.3-70b-versatile. Null = configured default.", required = false) String model,
            @ToolParam(description = "Search vertical: general, news, scholar, shopping, videos, images. Null = general.", required = false) String searchType,
            @ToolParam(description = "Recency filter: hour, day, week, month, year. Null = no filter.", required = false) String recency,
            @ToolParam(description = "Domain filter list. Use -domain.com to exclude (e.g. [\"-reddit.com\", \"reuters.com\"])", required = false) List<String> domainFilter,
            @ToolParam(description = "If true, embeds inline citation markers in the answer", required = false) Boolean citations,
            @ToolParam(description = "Number of search sub-queries (1-5). More = better coverage, higher cost.", required = false) Integer maxQueries,
            @ToolParam(description = "Max LLM output tokens (null = model default)", required = false) Integer maxTokens,
            @ToolParam(description = "Two-letter country code for geo bias (e.g. us, br, de). Null = configured default.", required = false) String location) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to LLMLayerAnswer");
            return JsonParser.toJson(Collections.emptyMap());
        }

        String effectiveModel    = StringUtils.hasText(model)    ? model    : this.defaultModel;
        String effectiveLocation = StringUtils.hasText(location) ? location : this.defaultLocation;

        logger.debug("LLMLayerAnswer: '{}' (model={}, searchType={})", query, effectiveModel, searchType);

        Observation observation = Observation.createNotStarted("llmlayer.answer", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation",          "answer")
                .lowCardinalityKeyValue("llmlayer.answer.model",       effectiveModel)
                .lowCardinalityKeyValue("llmlayer.answer.search_type", searchType != null ? searchType : "general")
                .highCardinalityKeyValue("llmlayer.answer.query",      query);

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("query",          query);
                body.put("model",          effectiveModel);
                body.put("return_sources", true);
                body.put("location",       effectiveLocation);

                if (StringUtils.hasText(searchType))           body.put("search_type",  searchType);
                if (StringUtils.hasText(recency))              body.put("recency",       recency);
                if (!CollectionUtils.isEmpty(domainFilter))    body.put("domain_filter", domainFilter);
                if (Boolean.TRUE.equals(citations))            body.put("citations",     true);
                if (maxQueries != null && maxQueries > 0)      body.put("max_queries",   Math.min(maxQueries, 5));
                if (maxTokens  != null && maxTokens  > 0)      body.put("max_tokens",    maxTokens);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(ANSWER_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerAnswer '{}': {}", query, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerAnswer '{}': {}", query, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerAnswer for: {}", query);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawSources =
                        (List<Map<String, Object>>) response.getOrDefault("sources", Collections.emptyList());

                List<Source> sources = rawSources.stream()
                        .filter(s -> s != null && s.get("link") != null)
                        .map(s -> new Source(
                                (String) s.getOrDefault("title", ""),
                                (String) s.get("link"),
                                (String) s.getOrDefault("snippet", "")
                        ))
                        .toList();

                AnswerResult result = new AnswerResult(
                        (String)  response.getOrDefault("answer",         ""),
                        sources,
                        (String)  response.getOrDefault("model",          effectiveModel),
                        toDouble( response.getOrDefault("llmlayer_cost",  0.0)),
                        toDouble( response.getOrDefault("model_cost",     0.0)),
                        toInt(    response.getOrDefault("input_tokens",   0)),
                        toInt(    response.getOrDefault("output_tokens",  0))
                );

                observation.highCardinalityKeyValue("llmlayer.answer.sources.count", String.valueOf(sources.size()));

                logger.debug("LLMLayerAnswer completed with {} sources", sources.size());
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerAnswer for '{}': {}", query, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // 2. SEARCH WEB — Multi-vertical web search (raw results, no LLM)
    // =========================================================================

    @Tool(name = "LLMLayerSearchWeb", description = """
        Searches the web across multiple content verticals and returns raw results
        (titles, URLs, snippets) WITHOUT an LLM-generated answer.

        Use when you want to:
        - Collect URLs to scrape with LLMLayerScrape
        - Find academic papers (searchType=scholar) before fetching their PDFs
        - Search for news, videos, images, or shopping results
        - Have full control over result processing before passing to an LLM

        Supported verticals: general, news, scholar, shopping, videos, images

        Usage notes:
        - Use -domain.com prefix in domainFilter to exclude a domain
        - Combine scholar results with LLMLayerPdf for full paper content
        - Combine news results with LLMLayerScrape for full article text
        - Cheaper than Answer when you just need URLs, not a synthesized response
        """)
    public String searchWeb(
            @ToolParam(description = "Search query") String query,
            @ToolParam(description = "Search vertical: general, news, scholar, shopping, videos, images. Null = general.", required = false) String searchType,
            @ToolParam(description = "Recency filter: hour, day, week, month, year", required = false) String recency,
            @ToolParam(description = "Two-letter country code for geo bias (e.g. us, br)", required = false) String location,
            @ToolParam(description = "Domain filter list. Use -domain.com to exclude.", required = false) List<String> domainFilter) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to LLMLayerSearchWeb");
            return JsonParser.toJson(Collections.emptyList());
        }

        String effectiveLocation = StringUtils.hasText(location) ? location : this.defaultLocation;
        String effectiveType     = StringUtils.hasText(searchType) ? searchType : SearchType.GENERAL.getValue();

        logger.debug("LLMLayerSearchWeb: '{}' (type={}, location={})", query, effectiveType, effectiveLocation);

        Observation observation = Observation.createNotStarted("llmlayer.search_web", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation",              "search_web")
                .lowCardinalityKeyValue("llmlayer.search_web.search_type", effectiveType)
                .highCardinalityKeyValue("llmlayer.search_web.query",      query);

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("query",       query);
                body.put("search_type", effectiveType);
                body.put("location",    effectiveLocation);

                if (StringUtils.hasText(recency))           body.put("recency",       recency);
                if (!CollectionUtils.isEmpty(domainFilter)) body.put("domain_filter", domainFilter);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(SEARCH_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerSearchWeb '{}': {}", query, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerSearchWeb '{}': {}", query, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerSearchWeb for: {}", query);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawResults =
                        (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

                List<SearchResult> results = rawResults.stream()
                        .filter(r -> r != null && r.get("link") != null)
                        .map(r -> new SearchResult(
                                (String) r.getOrDefault("title",   ""),
                                (String) r.get("link"),
                                (String) r.getOrDefault("snippet", ""),
                                (String) r.getOrDefault("date",    ""),
                                (String) r.getOrDefault("source",  "")
                        ))
                        .toList();

                observation.highCardinalityKeyValue("llmlayer.search_web.results.count", String.valueOf(results.size()));

                logger.debug("LLMLayerSearchWeb '{}' returned {} results", query, results.size());
                return JsonParser.toJson(results);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerSearchWeb for '{}': {}", query, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 3. SCRAPE — Clean content extraction from a URL
    // =========================================================================

    @Tool(name = "LLMLayerScrape", description = """
        Extracts clean content from any public URL.
        Handles JavaScript rendering, proxies, and anti-bot protection automatically.

        Supported output formats:
        - markdown  → clean text, best for LLM consumption
        - html      → raw page HTML
        - screenshot→ visual capture of the page (returns base64 URL)
        - pdf       → rendered PDF of the page

        Use after LLMLayerSearchWeb to get full article/page content from discovered URLs.

        Usage notes:
        - Multiple formats can be requested in one call
        - 15-second timeout per request
        - For actual PDF documents (not web-rendered), use LLMLayerPdf instead
        """)
    public String scrape(
            @ToolParam(description = "Full URL to scrape (e.g. https://example.com/article)") String url,
            @ToolParam(description = "Output formats: markdown, html, screenshot, pdf. Null = [markdown]", required = false) List<String> formats) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to LLMLayerScrape");
            return JsonParser.toJson(Collections.emptyMap());
        }

        List<String> effectiveFormats = CollectionUtils.isEmpty(formats) ? List.of("markdown") : formats;
        logger.debug("LLMLayerScrape: {} (formats={})", url, effectiveFormats);

        Observation observation = Observation.createNotStarted("llmlayer.scrape", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation", "scrape")
                .highCardinalityKeyValue("llmlayer.scrape.url", url);

        return observation.observe(() -> {
            try {
                Map<String, Object> body = Map.of(
                        "url",     url,
                        "formats", effectiveFormats
                );

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(SCRAPE_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerScrape for {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerScrape for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerScrape for: {}", url);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> meta = (Map<String, Object>) response.getOrDefault("metadata", Collections.emptyMap());

                ScrapeResult result = new ScrapeResult(
                        url,
                        (String) response.getOrDefault("markdown",   ""),
                        (String) response.getOrDefault("html",       ""),
                        (String) response.getOrDefault("screenshot", ""),
                        (String) response.getOrDefault("title",      meta.getOrDefault("title", "").toString()),
                        toInt(response.getOrDefault("statusCode", 200))
                );

                observation.highCardinalityKeyValue("llmlayer.scrape.status_code", String.valueOf(result.statusCode()));

                logger.debug("LLMLayerScrape of '{}' completed (status={})", url, result.statusCode());
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerScrape for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // 4. CRAWL — Recursive site traversal
    // =========================================================================

    @Tool(name = "LLMLayerCrawl", description = """
        Recursively crawls a website starting from a seed URL,
        returning content from all discovered pages.

        Use to build a comprehensive knowledge base from a site for RAG,
        or to index full documentation/blog content.

        Usage notes:
        - Returns pages as a stream internally; this tool waits for completion
        - Specify formats per page: markdown, html, screenshot, pdf
        - Combine with LLMLayerMap to preview structure first
        - Use allowedDomains to restrict crawling to specific subdomains
        """)
    public String crawl(
            @ToolParam(description = "Seed URL to start crawling from (e.g. https://docs.example.com)") String url,
            @ToolParam(description = "Output formats per page: markdown, html, screenshot, pdf. Null = [markdown]", required = false) List<String> formats,
            @ToolParam(description = "Maximum number of pages to crawl", required = false) Integer maxPages,
            @ToolParam(description = "Restrict crawling to these domains/subdomains", required = false) List<String> allowedDomains) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to LLMLayerCrawl");
            return JsonParser.toJson(Collections.emptyList());
        }

        List<String> effectiveFormats = CollectionUtils.isEmpty(formats) ? List.of("markdown") : formats;
        logger.info("LLMLayerCrawl starting at: {} (formats={})", url, effectiveFormats);

        Observation observation = Observation.createNotStarted("llmlayer.crawl", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation", "crawl")
                .highCardinalityKeyValue("llmlayer.crawl.url", url);

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("url",     url);
                body.put("formats", effectiveFormats);

                if (maxPages != null && maxPages > 0)          body.put("max_pages",       maxPages);
                if (!CollectionUtils.isEmpty(allowedDomains))  body.put("allowed_domains", allowedDomains);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(CRAWL_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerCrawl for {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerCrawl for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerCrawl for: {}", url);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawPages =
                        (List<Map<String, Object>>) response.getOrDefault("pages", Collections.emptyList());

                List<CrawlPage> pages = rawPages.stream()
                        .filter(p -> p != null && p.get("url") != null)
                        .map(p -> new CrawlPage(
                                (String) p.get("url"),
                                (String) p.getOrDefault("markdown",   ""),
                                (String) p.getOrDefault("title",      ""),
                                toInt(p.getOrDefault("statusCode", 200))
                        ))
                        .toList();

                observation.highCardinalityKeyValue("llmlayer.crawl.pages.count", String.valueOf(pages.size()));

                logger.info("LLMLayerCrawl of '{}' returned {} pages", url, pages.size());
                return JsonParser.toJson(pages);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerCrawl for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 5. MAP — Discover site URL structure
    // =========================================================================

    @Tool(name = "LLMLayerMap", description = """
        Discovers and returns all URLs on a website without extracting page content.
        Much faster and cheaper than crawling — use it first to plan what to crawl.

        Typical workflow:
        1. LLMLayerMap → see all available URLs
        2. Filter by relevance to your task
        3. LLMLayerScrape or LLMLayerCrawl → fetch only what you need

        Usage notes:
        - Returns URL list with HTTP status codes
        - Use to understand site structure before committing to a full crawl
        """)
    public String map(
            @ToolParam(description = "Base URL of the site to map (e.g. https://docs.example.com)") String url) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to LLMLayerMap");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.debug("LLMLayerMap: {}", url);

        Observation observation = Observation.createNotStarted("llmlayer.map", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation", "map")
                .highCardinalityKeyValue("llmlayer.map.url", url);

        return observation.observe(() -> {
            try {
                Map<String, Object> body = Map.of("url", url);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(MAP_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerMap for {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerMap for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerMap for: {}", url);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> rawLinks =
                        (List<Map<String, Object>>) response.getOrDefault("links", Collections.emptyList());

                List<MapLink> links = rawLinks.stream()
                        .filter(l -> l != null && l.get("url") != null)
                        .map(l -> new MapLink(
                                (String) l.get("url"),
                                toInt(l.getOrDefault("statusCode", 200))
                        ))
                        .toList();

                observation.highCardinalityKeyValue("llmlayer.map.urls.count", String.valueOf(links.size()));

                logger.debug("LLMLayerMap of '{}' found {} URLs", url, links.size());
                return JsonParser.toJson(links);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerMap for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 6. PDF — Extract text and metadata from PDF documents
    // =========================================================================

    @Tool(name = "LLMLayerPdf", description = """
        Extracts text content and metadata from any publicly accessible PDF URL.
        Ideal for academic papers, reports, whitepapers, and technical documents.

        Typical workflow with LLMLayerSearchWeb:
        1. Search with searchType=scholar → get papers with pdfUrl fields
        2. Use this tool on the pdfUrl → get full paper text for LLM analysis

        Returns: extracted text, page count, author, title, and other metadata.

        Usage notes:
        - Only works on publicly accessible PDFs (no login required)
        - For web pages rendered as PDF, use LLMLayerScrape with formats=[pdf]
        """)
    public String extractPdf(
            @ToolParam(description = "Full URL of the PDF document to extract (e.g. https://arxiv.org/pdf/2307.06435.pdf)") String url) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to LLMLayerPdf");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.debug("LLMLayerPdf: {}", url);

        Observation observation = Observation.createNotStarted("llmlayer.pdf", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation", "pdf")
                .highCardinalityKeyValue("llmlayer.pdf.url", url);

        return observation.observe(() -> {
            try {
                Map<String, Object> body = Map.of("url", url);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(PDF_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerPdf for {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerPdf for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerPdf for: {}", url);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> meta = (Map<String, Object>) response.getOrDefault("metadata", Collections.emptyMap());

                PdfResult result = new PdfResult(
                        url,
                        (String) response.getOrDefault("text",  ""),
                        (String) meta.getOrDefault("title",     ""),
                        (String) meta.getOrDefault("author",    ""),
                        toInt(   meta.getOrDefault("pages",     0)),
                        (String) meta.getOrDefault("createdAt", "")
                );

                observation.highCardinalityKeyValue("llmlayer.pdf.pages", String.valueOf(result.pages()));

                logger.debug("LLMLayerPdf extracted {} chars from {}", result.text().length(), url);
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerPdf for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // 7. YOUTUBE — Retrieve transcript and metadata from YouTube videos
    // =========================================================================

    @Tool(name = "LLMLayerYouTube", description = """
        Retrieves the transcript and metadata from any YouTube video.
        Returns the full spoken content, title, description, channel, and stats.

        Use when:
        - The user references a YouTube video and wants its content summarized
        - You need to analyze spoken content from a video for RAG
        - Extracting educational content from tutorials or lectures

        Returns: transcript text, title, description, author, view count, likes, publish date.

        Usage notes:
        - Requires a valid YouTube video URL (youtube.com/watch?v=... or youtu.be/...)
        - Transcript must be available (auto-generated or manually uploaded)
        - Specify language for multilingual videos (default: auto-detect)
        """)
    public String youtube(
            @ToolParam(description = "Full YouTube video URL (e.g. https://youtube.com/watch?v=dQw4w9WgXcQ)") String url,
            @ToolParam(description = "Language code for transcript (e.g. en, pt, es). Null = auto-detect.", required = false) String language) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to LLMLayerYouTube");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.debug("LLMLayerYouTube: {} (language={})", url, language);

        Observation observation = Observation.createNotStarted("llmlayer.youtube", observationRegistry)
                .lowCardinalityKeyValue("llmlayer.operation", "youtube")
                .highCardinalityKeyValue("llmlayer.youtube.url", url);

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("url", url);
                if (StringUtils.hasText(language)) body.put("language", language);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(YOUTUBE_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in LLMLayerYouTube for {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in LLMLayerYouTube for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from LLMLayerYouTube for: {}", url);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> meta = (Map<String, Object>) response.getOrDefault("metadata", Collections.emptyMap());

                YouTubeResult result = new YouTubeResult(
                        url,
                        (String) response.getOrDefault("transcript",  ""),
                        (String) meta.getOrDefault("title",           ""),
                        (String) meta.getOrDefault("description",     ""),
                        (String) meta.getOrDefault("author",          ""),
                        toInt(   meta.getOrDefault("views",           0)),
                        toInt(   meta.getOrDefault("likes",           0)),
                        (String) meta.getOrDefault("date",            "")
                );

                observation.highCardinalityKeyValue("llmlayer.youtube.transcript.length",
                        String.valueOf(result.transcript().length()));

                logger.debug("LLMLayerYouTube: retrieved {} chars of transcript for '{}'",
                        result.transcript().length(), result.title());
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in LLMLayerYouTube for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    private double toDouble(Object value) {
        if (value instanceof Number n) return n.doubleValue();
        return 0.0;
    }

    private int toInt(Object value) {
        if (value instanceof Number n) return n.intValue();
        return 0;
    }

    // =========================================================================
    // Result records
    // =========================================================================

    public record AnswerResult(
            String answer,
            List<Source> sources,
            String model,
            double llmlayerCost,
            double modelCost,
            int inputTokens,
            int outputTokens
    ) {}

    public record Source(String title, String url, String snippet) {}

    public record SearchResult(
            String title,
            String url,
            String snippet,
            String date,
            String source
    ) {}

    public record ScrapeResult(
            String url,
            String markdown,
            String html,
            String screenshot,
            String title,
            int statusCode
    ) {}

    public record CrawlPage(String url, String markdown, String title, int statusCode) {}

    public record MapLink(String url, int statusCode) {}

    public record PdfResult(
            String url,
            String text,
            String title,
            String author,
            int pages,
            String createdAt
    ) {}

    public record YouTubeResult(
            String url,
            String transcript,
            String title,
            String description,
            String author,
            int views,
            int likes,
            String date
    ) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder(String apiKey) {
        return new Builder(apiKey);
    }

    public static class Builder {

        private final String apiKey;
        private String defaultModel    = Model.GPT_4O_MINI.getValue();
        private String defaultLocation = "us";
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        private Builder(String apiKey) {
            Assert.hasText(apiKey, "API key must not be null or empty");
            this.apiKey = apiKey;
        }

        /** Default LLM model for the Answer tool. */
        public Builder defaultModel(String model) {
            Assert.hasText(model, "model must not be null or empty");
            this.defaultModel = model;
            return this;
        }

        /** Convenience overload accepting the Model enum. */
        public Builder defaultModel(Model model) {
            Assert.notNull(model, "model must not be null");
            this.defaultModel = model.getValue();
            return this;
        }

        /** Default geo bias for search and answer (ISO country code). */
        public Builder defaultLocation(String location) {
            Assert.hasText(location, "location must not be null or empty");
            this.defaultLocation = location;
            return this;
        }

        /** Observation registry for metrics and tracing. Defaults to {@link ObservationRegistry#NOOP}. */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry must not be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public LLMLayerWebTool build() {
            return new LLMLayerWebTool(this.apiKey, this.defaultModel, this.defaultLocation,
                    this.observationRegistry);
        }
    }
}