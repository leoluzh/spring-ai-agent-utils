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
 * Tavily Web Tool for Spring AI.
 * <p>
 * Provides five complementary capabilities using the Tavily API,
 * optimized for LLMs and RAG pipelines:
 * <ul>
 *   <li><b>Search</b>   — Web search optimized for LLMs with optional inline answer</li>
 *   <li><b>Extract</b>  — Clean content extraction from known URLs</li>
 *   <li><b>Crawl</b>    — Recursive site traversal with natural language instructions</li>
 *   <li><b>Map</b>      — Discover site URL structure without extracting content</li>
 *   <li><b>Research</b> — Async deep research report with multi-source synthesis</li>
 * </ul>
 *
 * <p>Free tier: <b>1,000 credits/month, no credit card required.</b>
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public TavilyWebTool tavilyWebTool(
 *         @Value("${tavily.api.key}") String apiKey) {
 *     return TavilyWebTool.builder(apiKey)
 *         .maxResults(5)
 *         .searchDepth(TavilyWebTool.SearchDepth.BASIC)
 *         .build();
 * }
 * }</pre>
 *
 * <h3>application.yaml</h3>
 * <pre>
 * tavily:
 *   api:
 *     key: ${TAVILY_API_KEY}
 * </pre>
 *
 * @author Spring AI Community
 * @see <a href="https://docs.tavily.com">Tavily API Documentation</a>
 */
public class TavilyWebTool {

    private static final Logger logger = LoggerFactory.getLogger(TavilyWebTool.class);

    private static final String BASE_URL       = "https://api.tavily.com";
    private static final String SEARCH_PATH    = "/search";
    private static final String EXTRACT_PATH   = "/extract";
    private static final String CRAWL_PATH     = "/crawl";
    private static final String MAP_PATH       = "/map";
    private static final String RESEARCH_PATH  = "/research";
    private static final String RESEARCH_GET   = "/research/{id}";

    private static final long POLL_INTERVAL_MS  = 3_000;
    private static final int  MAX_POLL_ATTEMPTS = 30; // 90s max

    private final RestClient  restClient;
    private final int         maxResults;
    private final String      searchDepth;

    /**
     * Search depth — controls quality vs cost trade-off.
     * Basic = 1 credit/request. Advanced = 2 credits/request.
     */
    public enum SearchDepth {
        BASIC("basic"),
        ADVANCED("advanced");

        private final String value;
        SearchDepth(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Topic category for search — enables domain-specific ranking.
     */
    public enum Topic {
        GENERAL("general"),
        NEWS("news"),
        FINANCE("finance");

        private final String value;
        Topic(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Extraction depth — basic is cheaper, advanced handles tables and embedded content.
     * Basic = 1 credit/5 URLs. Advanced = 2 credits/5 URLs.
     */
    public enum ExtractDepth {
        BASIC("basic"),
        ADVANCED("advanced");

        private final String value;
        ExtractDepth(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private TavilyWebTool(String apiKey, int maxResults, SearchDepth searchDepth) {
        Assert.hasText(apiKey, "Tavily API key must not be null or empty");
        this.restClient = RestClient.builder()
                .baseUrl(BASE_URL)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .defaultHeader("Content-Type",  MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("Accept",        MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.maxResults  = maxResults;
        this.searchDepth = searchDepth.getValue();
    }

    // =========================================================================
    // 1. SEARCH — Web search optimized for LLMs
    // =========================================================================

    @Tool(name = "TavilySearch", description = """
        Searches the web using Tavily — a search engine optimized for LLMs and RAG.
        Returns relevant results with titles, URLs, content snippets, and relevance scores.
        Optionally includes an LLM-generated answer grounded in the results.

        Use for:
        - Current events and news (use topic=news)
        - Financial data and reports (use topic=finance)
        - General research questions
        - Any information beyond your training cutoff

        Usage notes:
        - basic depth = 1 credit (fast), advanced = 2 credits (more thorough)
        - include_answer=basic for a quick answer, advanced for detailed summary
        - Use includeDomains to restrict to trusted sources
        - Wrap exact phrases in quotes in your query for precise matching
        - After responding, include a "Sources:" section with markdown links

        Free tier: 1,000 credits/month, no credit card required.
        """)
    public String search(
            @ToolParam(description = "Search query. Wrap exact phrases in quotes for precise matching.") String query,
            @ToolParam(description = "Max results to return (null = configured default, max 20)", required = false) Integer maxResults,
            @ToolParam(description = "Search depth: basic (1 credit) or advanced (2 credits, more thorough). Null = configured default.", required = false) String searchDepth,
            @ToolParam(description = "Topic category: general, news, finance. Null = general.", required = false) String topic,
            @ToolParam(description = "Include LLM answer: basic (quick), advanced (detailed). Null = no answer.", required = false) String includeAnswer,
            @ToolParam(description = "Only include results from these domains (max 300)", required = false) List<String> includeDomains,
            @ToolParam(description = "Exclude results from these domains (max 150)", required = false) List<String> excludeDomains,
            @ToolParam(description = "Only include results published after this date (YYYY-MM-DD)", required = false) String startPublishedDate,
            @ToolParam(description = "Only include results published before this date (YYYY-MM-DD)", required = false) String endPublishedDate) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to TavilySearch");
            return JsonParser.toJson(Collections.emptyList());
        }

        int    effectiveMax   = (maxResults != null && maxResults > 0) ? maxResults : this.maxResults;
        String effectiveDepth = StringUtils.hasText(searchDepth) ? searchDepth : this.searchDepth;

        logger.debug("TavilySearch: '{}' (depth={}, limit={})", query, effectiveDepth, effectiveMax);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("query",        query);
            body.put("max_results",  effectiveMax);
            body.put("search_depth", effectiveDepth);
            body.put("include_raw_content", "markdown");

            if (StringUtils.hasText(topic))              body.put("topic",                topic);
            if (StringUtils.hasText(includeAnswer))      body.put("include_answer",       includeAnswer);
            if (!CollectionUtils.isEmpty(includeDomains)) body.put("include_domains",     includeDomains);
            if (!CollectionUtils.isEmpty(excludeDomains)) body.put("exclude_domains",     excludeDomains);
            if (StringUtils.hasText(startPublishedDate)) body.put("start_published_date", startPublishedDate);
            if (StringUtils.hasText(endPublishedDate))   body.put("end_published_date",   endPublishedDate);

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(SEARCH_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in TavilySearch '{}': {}", query, res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in TavilySearch '{}': {}", query, res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from TavilySearch for: {}", query);
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            List<SearchResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> new SearchResult(
                            (String)  r.getOrDefault("title",       ""),
                            (String)  r.get("url"),
                            (String)  r.getOrDefault("content",     ""),
                            (String)  r.getOrDefault("raw_content", ""),
                            toDouble( r.getOrDefault("score",       0.0)),
                            (String)  r.getOrDefault("published_date", "")
                    ))
                    .toList();

            String answer = (String) response.getOrDefault("answer", null);

            SearchResponse finalResponse = new SearchResponse(
                    query,
                    results,
                    answer
            );

            logger.debug("TavilySearch '{}' returned {} results", query, results.size());
            return JsonParser.toJson(finalResponse);

        } catch (RestClientException e) {
            logger.error("Error in TavilySearch for '{}': {}", query, e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 2. EXTRACT — Clean content extraction from known URLs
    // =========================================================================

    @Tool(name = "TavilyExtract", description = """
        Extracts clean, structured content from one or more known URLs.
        Returns raw markdown or text content, ready for LLM consumption.

        Use when you already have URLs and need their full content —
        more targeted than Search when the sources are already known.

        Typical workflow:
        1. Use TavilySearch to find relevant URLs
        2. Filter by relevance score (> 0.5 recommended)
        3. Use TavilyExtract to get full content from top URLs

        Usage notes:
        - basic depth: 1 credit per 5 successful extractions
        - advanced depth: 2 credits per 5 extractions (handles tables, embedded content)
        - Failed extractions are NOT charged
        - Use query parameter to focus extraction on specific aspects
        - Returns failed_results list for URLs that could not be processed
        """)
    public String extract(
            @ToolParam(description = "List of URLs to extract content from") List<String> urls,
            @ToolParam(description = "Optional query to focus the extraction on specific content", required = false) String query,
            @ToolParam(description = "Extraction depth: basic or advanced. Advanced handles tables and embedded content.", required = false) String extractDepth) {

        if (CollectionUtils.isEmpty(urls)) {
            logger.warn("Empty URL list provided to TavilyExtract");
            return JsonParser.toJson(Collections.emptyList());
        }

        String effectiveDepth = StringUtils.hasText(extractDepth) ? extractDepth : ExtractDepth.BASIC.getValue();
        logger.debug("TavilyExtract: {} URLs (depth={})", urls.size(), effectiveDepth);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("urls",          urls);
            body.put("extract_depth", effectiveDepth);
            body.put("format",        "markdown");

            if (StringUtils.hasText(query)) body.put("query", query);

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(EXTRACT_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in TavilyExtract: {}", res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in TavilyExtract: {}", res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from TavilyExtract");
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            @SuppressWarnings("unchecked")
            List<String> failedUrls =
                    (List<String>) response.getOrDefault("failed_results", Collections.emptyList());

            List<ExtractResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> new ExtractResult(
                            (String) r.get("url"),
                            (String) r.getOrDefault("raw_content", "")
                    ))
                    .toList();

            if (!failedUrls.isEmpty()) {
                logger.warn("TavilyExtract: {} URLs failed to extract: {}", failedUrls.size(), failedUrls);
            }

            ExtractResponse finalResponse = new ExtractResponse(results, failedUrls);
            logger.debug("TavilyExtract: {} succeeded, {} failed", results.size(), failedUrls.size());
            return JsonParser.toJson(finalResponse);

        } catch (RestClientException e) {
            logger.error("Error in TavilyExtract: {}", e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 3. CRAWL — Recursive site traversal
    // =========================================================================

    @Tool(name = "TavilyCrawl", description = """
        Recursively traverses a website starting from a base URL,
        extracting content from all discovered pages.

        Use when you need comprehensive content from an entire site or section,
        such as full documentation, a blog, or a product catalog.

        Unique feature: natural language 'instructions' parameter lets you guide
        the crawler in plain English (e.g. "only follow pages about Python SDK").

        Usage notes:
        - max_depth controls how deep to follow links from the base URL
        - max_breadth controls how many links per page to follow
        - limit controls total pages to process (default 50)
        - Use selectPaths regex to restrict to specific URL patterns
        - Prefer TavilyMap first to preview structure, then Crawl for content
        - basic depth: 1 credit/5 pages — advanced: 2 credits/5 pages
        """)
    public String crawl(
            @ToolParam(description = "Base URL to start crawling from (e.g. https://docs.example.com)") String url,
            @ToolParam(description = "Natural language instructions to guide the crawler (e.g. 'only follow API reference pages')", required = false) String instructions,
            @ToolParam(description = "Max depth of links to follow from base URL (default 1)", required = false) Integer maxDepth,
            @ToolParam(description = "Max links to follow per page (default 20)", required = false) Integer maxBreadth,
            @ToolParam(description = "Total max pages to process (default 50)", required = false) Integer limit,
            @ToolParam(description = "Regex patterns to restrict crawling to specific URL paths (e.g. [\"/docs/.*\"])", required = false) List<String> selectPaths,
            @ToolParam(description = "Regex patterns to exclude specific URL paths (e.g. [\"/private/.*\"])", required = false) List<String> excludePaths) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to TavilyCrawl");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.info("TavilyCrawl starting at: {} (depth={}, limit={})", url, maxDepth, limit);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("url",    url);
            body.put("format", "markdown");

            if (StringUtils.hasText(instructions))      body.put("instructions",  instructions);
            if (maxDepth   != null && maxDepth   > 0)   body.put("max_depth",     maxDepth);
            if (maxBreadth != null && maxBreadth > 0)   body.put("max_breadth",   maxBreadth);
            if (limit      != null && limit      > 0)   body.put("limit",         limit);
            if (!CollectionUtils.isEmpty(selectPaths))  body.put("select_paths",  selectPaths);
            if (!CollectionUtils.isEmpty(excludePaths)) body.put("exclude_paths", excludePaths);

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(CRAWL_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in dailyCrawl for {}: {}", url, res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in TavilyCrawl for {}: {}", url, res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from TavilyCrawl for: {}", url);
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            List<CrawlResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> new CrawlResult(
                            (String) r.get("url"),
                            (String) r.getOrDefault("raw_content", "")
                    ))
                    .toList();

            logger.info("TavilyCrawl of '{}' returned {} pages", url, results.size());
            return JsonParser.toJson(results);

        } catch (RestClientException e) {
            logger.error("Error in TavilyCrawl for {}: {}", url, e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 4. MAP — Discover site URL structure
    // =========================================================================

    @Tool(name = "TavilyMap", description = """
        Creates a structured map of all URLs on a website without extracting content.
        Much faster and cheaper than crawling when you only need the site structure.

        Use to:
        - Preview a site's structure before deciding what to crawl
        - Find specific pages or sections by URL pattern
        - Understand how a documentation site is organized

        Typical workflow:
        1. TavilyMap to discover URLs and structure
        2. Filter URLs by relevance
        3. TavilyExtract or TavilyCrawl for the specific pages you need

        Usage notes:
        - Use selectPaths to focus on specific sections (e.g. [\"/api/.*\"])
        - Use instructions to guide URL selection in natural language
        - Significantly cheaper than crawl — use it first to plan
        """)
    public String map(
            @ToolParam(description = "Base URL of the site to map (e.g. https://docs.example.com)") String url,
            @ToolParam(description = "Natural language instructions to guide URL selection", required = false) String instructions,
            @ToolParam(description = "Max depth from base URL to follow (default 1)", required = false) Integer maxDepth,
            @ToolParam(description = "Max links to follow per page (default 20)", required = false) Integer maxBreadth,
            @ToolParam(description = "Total max links to process (default 50)", required = false) Integer limit,
            @ToolParam(description = "Regex patterns to include only specific URL paths (e.g. [\"/docs/.*\"])", required = false) List<String> selectPaths,
            @ToolParam(description = "Regex patterns to exclude specific URL paths", required = false) List<String> excludePaths) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to TavilyMap");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.debug("TavilyMap: {}", url);

        try {
            var body = new java.util.HashMap<String, Object>();
            body.put("url", url);

            if (StringUtils.hasText(instructions))      body.put("instructions",  instructions);
            if (maxDepth   != null && maxDepth   > 0)   body.put("max_depth",     maxDepth);
            if (maxBreadth != null && maxBreadth > 0)   body.put("max_breadth",   maxBreadth);
            if (limit      != null && limit      > 0)   body.put("limit",         limit);
            if (!CollectionUtils.isEmpty(selectPaths))  body.put("select_paths",  selectPaths);
            if (!CollectionUtils.isEmpty(excludePaths)) body.put("exclude_paths", excludePaths);

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(MAP_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error in TavilyMap for {}: {}", url, res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error in TavilyMap for {}: {}", url, res.getStatusCode()))
                    .body(Map.class);

            if (response == null) {
                logger.warn("Null response from TavilyMap for: {}", url);
                return JsonParser.toJson(Collections.emptyList());
            }

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> rawResults =
                    (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

            List<MapResult> results = rawResults.stream()
                    .filter(r -> r != null && r.get("url") != null)
                    .map(r -> new MapResult((String) r.get("url")))
                    .toList();

            logger.debug("TavilyMap of '{}' found {} URLs", url, results.size());
            return JsonParser.toJson(results);

        } catch (RestClientException e) {
            logger.error("Error in TavilyMap for {}: {}", url, e.getMessage());
            return JsonParser.toJson(Collections.emptyList());
        }
    }

    // =========================================================================
    // 5. RESEARCH — Async deep research report
    // =========================================================================

    @Tool(name = "TavilyResearch", description = """
        Performs comprehensive deep research on a topic by running multiple searches,
        analyzing sources, and generating a detailed research report with citations.

        Use for complex, multi-faceted questions that require synthesis across
        many sources — not for simple factual lookups.

        Examples:
        - "What are the latest developments in fusion energy research in 2025?"
        - "Compare the architectures of GPT-4, Claude 3, and Gemini Ultra"
        - "What is the current state of quantum computing hardware?"

        Usage notes:
        - Async operation: polls until complete (can take 30–120 seconds)
        - Returns a structured markdown report with inline citations
        - Much more thorough than a single Search + answer
        - Higher credit cost — reserve for questions that truly need deep research
        """)
    public String research(
            @ToolParam(description = "Research topic or question requiring deep multi-source analysis") String query) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to TavilyResearch");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.info("TavilyResearch starting: '{}'", query);

        try {
            // 1. Submit the research task
            Map<String, Object> body = Map.of("query", query);

            @SuppressWarnings("unchecked")
            Map<String, Object> startResponse = restClient.post()
                    .uri(RESEARCH_PATH)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                            logger.error("4xx error starting TavilyResearch '{}': {}", query, res.getStatusCode()))
                    .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                            logger.error("5xx error starting TavilyResearch '{}': {}", query, res.getStatusCode()))
                    .body(Map.class);

            if (startResponse == null) {
                logger.error("Null response starting TavilyResearch for: {}", query);
                return JsonParser.toJson(Collections.emptyMap());
            }

            String requestId = (String) startResponse.get("request_id");
            if (!StringUtils.hasText(requestId)) {
                logger.error("No request_id in TavilyResearch response for: {}", query);
                return JsonParser.toJson(Collections.emptyMap());
            }

            logger.debug("TavilyResearch task created: {}", requestId);

            // 2. Poll until completed
            return pollResearchResult(requestId, query);

        } catch (RestClientException e) {
            logger.error("Error starting TavilyResearch for '{}': {}", query, e.getMessage());
            return JsonParser.toJson(Collections.emptyMap());
        }
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    private String pollResearchResult(String requestId, String query) {
        for (int attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
            try {
                Thread.sleep(POLL_INTERVAL_MS);

                @SuppressWarnings("unchecked")
                Map<String, Object> status = restClient.get()
                        .uri(RESEARCH_GET, requestId)
                        .retrieve()
                        .body(Map.class);

                if (status == null) continue;

                String researchStatus = (String) status.get("status");
                logger.debug("TavilyResearch {} — status: {} ({}/{})",
                        requestId, researchStatus, attempt, MAX_POLL_ATTEMPTS);

                if ("completed".equals(researchStatus)) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> rawSources =
                            (List<Map<String, Object>>) status.getOrDefault("sources", Collections.emptyList());

                    List<ResearchSource> sources = rawSources.stream()
                            .filter(s -> s != null && s.get("url") != null)
                            .map(s -> new ResearchSource(
                                    (String) s.getOrDefault("title", ""),
                                    (String) s.get("url")
                            ))
                            .toList();

                    ResearchResult result = new ResearchResult(
                            (String) status.getOrDefault("content", ""),
                            sources,
                            requestId
                    );

                    logger.info("TavilyResearch '{}' completed with {} sources", query, sources.size());
                    return JsonParser.toJson(result);
                }

                if ("failed".equals(researchStatus)) {
                    logger.error("TavilyResearch {} failed for query: {}", requestId, query);
                    return JsonParser.toJson(Collections.emptyMap());
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("TavilyResearch polling interrupted for: {}", requestId);
                return JsonParser.toJson(Collections.emptyMap());
            } catch (RestClientException e) {
                logger.warn("Error polling TavilyResearch {}: {}", requestId, e.getMessage());
            }
        }

        logger.warn("TavilyResearch {} timed out after {} attempts", requestId, MAX_POLL_ATTEMPTS);
        return JsonParser.toJson(Collections.emptyMap());
    }

    private double toDouble(Object value) {
        if (value instanceof Number n) return n.doubleValue();
        return 0.0;
    }

    // =========================================================================
    // Result records
    // =========================================================================

    /** Wraps search results plus optional LLM-generated answer. */
    public record SearchResponse(String query, List<SearchResult> results, String answer) {}

    /** Single search result with relevance score. */
    public record SearchResult(
            String title,
            String url,
            String content,
            String rawContent,
            double score,
            String publishedDate
    ) {}

    /** Wraps extracted results and failed URLs. */
    public record ExtractResponse(List<ExtractResult> results, List<String> failedResults) {}

    /** Extracted content from a single URL. */
    public record ExtractResult(String url, String rawContent) {}

    /** Single page returned from a crawl. */
    public record CrawlResult(String url, String rawContent) {}

    /** Single URL discovered during a map operation. */
    public record MapResult(String url) {}

    /** Completed deep research report with sources. */
    public record ResearchResult(String content, List<ResearchSource> sources, String requestId) {}

    /** A source citation from the research report. */
    public record ResearchSource(String title, String url) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder(String apiKey) {
        return new Builder(apiKey);
    }

    public static class Builder {

        private final String apiKey;
        private int         maxResults  = 5;
        private SearchDepth searchDepth = SearchDepth.BASIC;

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

        /** Default search depth. Basic = 1 credit, Advanced = 2 credits. */
        public Builder searchDepth(SearchDepth searchDepth) {
            Assert.notNull(searchDepth, "searchDepth must not be null");
            this.searchDepth = searchDepth;
            return this;
        }

        public TavilyWebTool build() {
            return new TavilyWebTool(this.apiKey, this.maxResults, this.searchDepth);
        }
    }
}