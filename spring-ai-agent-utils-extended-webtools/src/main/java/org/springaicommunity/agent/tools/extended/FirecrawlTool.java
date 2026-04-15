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
 * Firecrawl Web Tool for Spring AI.
 * <p>
 * Provides five complementary web data capabilities using the Firecrawl API:
 * <ul>
 *   <li><b>Scrape</b>  — Convert any URL to clean markdown or structured JSON</li>
 *   <li><b>Search</b>  — Find pages and extract full content in one call</li>
 *   <li><b>Agent</b>   — Autonomous multi-step data gathering via natural language</li>
 *   <li><b>Map</b>     — Discover full site structure instantly</li>
 *   <li><b>Crawl</b>   — Navigate entire sites without sitemaps (async + polling)</li>
 * </ul>
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public FirecrawlTool firecrawlTool(@Value("${firecrawl.api.key}") String apiKey) {
 *     return FirecrawlTool.builder(apiKey)
 *         .searchLimit(5)
 *         .crawlLimit(10)
 *         .agentModel(FirecrawlTool.AgentModel.SPARK_1_MINI)
 *         .build();
 * }
 * }</pre>
 *
 * <h3>application.yaml</h3>
 * <pre>
 * firecrawl:
 *   api:
 *     key: ${FIRECRAWL_API_KEY}
 * </pre>
 *
 * @author Spring AI Community
 * @see <a href="https://docs.firecrawl.dev">Firecrawl API Docs</a>
 */
public class FirecrawlTool {

    private static final Logger logger = LoggerFactory.getLogger(FirecrawlTool.class);

    private static final String BASE_URL       = "https://api.firecrawl.dev";
    private static final String SCRAPE_PATH    = "/v1/scrape";
    private static final String SEARCH_PATH    = "/v2/search";
    private static final String AGENT_PATH     = "/v2/agent";
    private static final String MAP_PATH       = "/v1/map";
    private static final String CRAWL_PATH     = "/v1/crawl";
    private static final String CRAWL_STATUS   = "/v1/crawl/{id}";

    private static final long POLL_INTERVAL_MS  = 3_000;
    private static final int  MAX_POLL_ATTEMPTS = 20;

    private final RestClient restClient;
    private final int        searchLimit;
    private final int        crawlLimit;
    private final String     agentModel;
    private final ObservationRegistry observationRegistry;

    /**
     * Available models for the Agent endpoint.
     * @see <a href="https://docs.firecrawl.dev/api-reference/endpoint/agent">Agent docs</a>
     */
    public enum AgentModel {
        /** Fast retrieval — cheaper, good for straightforward questions. */
        SPARK_1_MINI("spark-1-mini"),
        /** Advanced extraction — better for complex or ambiguous research. */
        SPARK_1_PRO("spark-1-pro");

        private final String value;
        AgentModel(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private FirecrawlTool(String apiKey, int searchLimit, int crawlLimit, AgentModel agentModel,
                          ObservationRegistry observationRegistry) {
        Assert.hasText(apiKey, "Firecrawl API key must not be null or empty");
        this.restClient = RestClient.builder()
                .baseUrl(BASE_URL)
                .defaultHeader("Authorization", "Bearer " + apiKey)
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("Accept",       MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.searchLimit = searchLimit;
        this.crawlLimit  = crawlLimit;
        this.agentModel  = agentModel.getValue();
        this.observationRegistry = observationRegistry;
    }

    // =========================================================================
    // 1. SCRAPE — Convert any URL to markdown
    // =========================================================================

    /**
     * Converts any URL to clean markdown content.
     * Handles JavaScript rendering and anti-bot protection automatically.
     */
    @Tool(name = "FirecrawlScrape", description = """
        Converts any URL to clean markdown or structured content.
        Use when the user provides a specific URL to read, or when you need
        the full text of a known page to ground your response.

        Handles JavaScript-rendered sites, paywalls, and anti-bot protection.
        Returns: title, URL, and full markdown content of the page.

        Usage notes:
        - Prefer this over Search when the exact URL is already known
        - 1 credit per page scraped
        """)
    public String scrape(
            @ToolParam(description = "Full URL to scrape (e.g. https://example.com/page)") String url) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to scrape");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.debug("Scraping URL: {}", url);

        Observation observation = Observation.createNotStarted("firecrawl.scrape", observationRegistry)
                .lowCardinalityKeyValue("firecrawl.operation", "scrape")
                .highCardinalityKeyValue("firecrawl.scrape.url", url);

        return observation.observe(() -> {
            try {
                Map<String, Object> body = Map.of(
                        "url",             url,
                        "formats",         List.of("markdown"),
                        "onlyMainContent", true
                );

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(SCRAPE_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error scraping {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error scraping {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null || !Boolean.TRUE.equals(response.get("success"))) {
                    logger.warn("Invalid response scraping: {}", url);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> data = (Map<String, Object>) response.get("data");
                return JsonParser.toJson(toScrapeResult(data));

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error scraping {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // 2. SEARCH — Find pages and extract content in one call
    // =========================================================================

    /**
     * Searches the web and returns full page content (not just snippets).
     */
    @Tool(name = "FirecrawlSearch", description = """
        Searches the web and returns the FULL markdown content of each result — not just snippets.
        Combines web search with scraping in a single call, ideal for RAG pipelines
        and responses that need to be grounded in real, current sources.

        Use when you need up-to-date information beyond your training cutoff.

        IMPORTANT: After answering, always include a "Sources:" section with
        markdown links to the pages used:
            Sources:
            - [Page Title](https://example.com)

        Usage notes:
        - More powerful than a plain search — returns full page text
        - Domain filtering available via allowedDomains / blockedDomains (client-side)
        - For quota efficiency, use site operators in query: "Spring AI site:spring.io"
        """)
    public String search(
            @ToolParam(description = "Search query to execute") String query,
            @ToolParam(description = "Max results to return (null = use configured default)", required = false) Integer limit,
            @ToolParam(description = "Only include results from these domains (client-side filter)", required = false) List<String> allowedDomains,
            @ToolParam(description = "Never include results from these domains (client-side filter)", required = false) List<String> blockedDomains) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to search");
            return JsonParser.toJson(Collections.emptyList());
        }

        int effectiveLimit = (limit != null && limit > 0) ? limit : this.searchLimit;
        logger.debug("Searching: '{}' (limit={})", query, effectiveLimit);

        Observation observation = Observation.createNotStarted("firecrawl.search", observationRegistry)
                .lowCardinalityKeyValue("firecrawl.operation", "search")
                .highCardinalityKeyValue("firecrawl.search.query", query)
                .highCardinalityKeyValue("firecrawl.search.limit", String.valueOf(effectiveLimit));

        return observation.observe(() -> {
            try {
                Map<String, Object> body = Map.of(
                        "query", query,
                        "limit", effectiveLimit,
                        "scrapeOptions", Map.of(
                                "formats",         List.of("markdown"),
                                "onlyMainContent", true
                        )
                );

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(SEARCH_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error searching '{}': {}", query, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error searching '{}': {}", query, res.getStatusCode()))
                        .body(Map.class);

                if (response == null || !Boolean.TRUE.equals(response.get("success"))) {
                    logger.warn("Invalid response for query: {}", query);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> data =
                        (List<Map<String, Object>>) response.getOrDefault("data", Collections.emptyList());

                List<SearchResult> results = data.stream()
                        .filter(r -> r != null && r.get("url") != null)
                        .map(r -> new SearchResult(
                                (String) r.get("title"),
                                (String) r.get("url"),
                                (String) r.getOrDefault("description", ""),
                                (String) r.getOrDefault("markdown", "")
                        ))
                        .filter(r -> matchesDomainFilter(r.url(), allowedDomains, blockedDomains))
                        .toList();

                observation.highCardinalityKeyValue("firecrawl.search.results.count", String.valueOf(results.size()));

                logger.debug("Search '{}' returned {} results", query, results.size());
                return JsonParser.toJson(results);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error searching '{}': {}", query, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 3. AGENT — Autonomous multi-step data gathering
    // =========================================================================

    /**
     * Autonomous agent that searches, navigates, and extracts data based on a prompt.
     * Does not require knowing the URLs upfront.
     */
    @Tool(name = "FirecrawlAgent", description = """
        Autonomous AI agent that searches, navigates, and gathers data from the web
        based on a natural language prompt — no URLs required.

        Use when the task requires multi-step reasoning, visiting multiple pages,
        or when the exact sources are not known in advance.

        Examples of good prompts:
        - "Find the pricing plans for Notion"
        - "Get all YC W24 companies with their founders and websites"
        - "Find the latest release notes for Spring AI"

        Usage notes:
        - Takes 30–60 seconds (agent browses and reasons before responding)
        - More expensive than scrape/search — uses variable credits per task
        - Optionally constrain to specific URLs via the urls parameter
        - Returns a natural language result with source URLs
        """)
    public String agent(
            @ToolParam(description = "Natural language description of the data to gather") String prompt,
            @ToolParam(description = "Optional list of URLs to constrain the search to", required = false) List<String> urls,
            @ToolParam(description = "Maximum credits to spend on this task (null = no limit)", required = false) Integer maxCredits) {

        if (!StringUtils.hasText(prompt)) {
            logger.warn("Empty prompt provided to agent");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.info("Starting agent task: '{}'", prompt);

        Observation observation = Observation.createNotStarted("firecrawl.agent", observationRegistry)
                .lowCardinalityKeyValue("firecrawl.operation", "agent")
                .lowCardinalityKeyValue("firecrawl.agent.model", this.agentModel)
                .highCardinalityKeyValue("firecrawl.agent.prompt", prompt);

        return observation.observe(() -> {
            try {
                var bodyBuilder = new java.util.HashMap<String, Object>();
                bodyBuilder.put("prompt", prompt);
                bodyBuilder.put("model",  this.agentModel);
                if (!CollectionUtils.isEmpty(urls))           bodyBuilder.put("urls",       urls);
                if (maxCredits != null && maxCredits > 0)     bodyBuilder.put("maxCredits", maxCredits);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(AGENT_PATH)
                        .body(bodyBuilder)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error in agent task '{}': {}", prompt, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error in agent task '{}': {}", prompt, res.getStatusCode()))
                        .body(Map.class);

                if (response == null || !Boolean.TRUE.equals(response.get("success"))) {
                    logger.warn("Invalid response from agent for prompt: {}", prompt);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> data = (Map<String, Object>) response.get("data");
                if (data == null) return JsonParser.toJson(Collections.emptyMap());

                AgentResult result = new AgentResult(
                        (String) data.getOrDefault("result", ""),
                        castToStringList(data.get("sources"))
                );

                observation.highCardinalityKeyValue("firecrawl.agent.sources.count", String.valueOf(result.sources().size()));

                logger.info("Agent task completed. Sources used: {}", result.sources().size());
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in agent task '{}': {}", prompt, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });
    }

    // =========================================================================
    // 4. MAP — Discover full site structure
    // =========================================================================

    /**
     * Discovers all URLs on a website without scraping their content.
     * Much faster and cheaper than crawling when you only need the structure.
     */
    @Tool(name = "FirecrawlMap", description = """
        Discovers and returns all URLs on a website instantly, without scraping content.
        Use to understand a site's structure before deciding what to scrape or crawl,
        or when you need a list of pages matching a specific search term.

        Returns: list of URLs with titles and descriptions.

        Usage notes:
        - Much faster and cheaper than crawl (1 credit per website, not per page)
        - Combine with scrape to selectively extract only the pages you need
        - Use search parameter to filter URLs by relevance
        - ignoreCache forces a fresh discovery instead of using cached results
        """)
    public String map(
            @ToolParam(description = "Base URL of the site to map (e.g. https://docs.example.com)") String url,
            @ToolParam(description = "Optional keyword to filter discovered URLs by relevance", required = false) String search,
            @ToolParam(description = "If true, bypasses cached results for a fresh discovery", required = false) Boolean ignoreCache) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to map");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.debug("Mapping site: {}", url);

        Observation observation = Observation.createNotStarted("firecrawl.map", observationRegistry)
                .lowCardinalityKeyValue("firecrawl.operation", "map")
                .highCardinalityKeyValue("firecrawl.map.url", url);

        return observation.observe(() -> {
            try {
                var bodyBuilder = new java.util.HashMap<String, Object>();
                bodyBuilder.put("url", url);
                if (StringUtils.hasText(search))          bodyBuilder.put("search",      search);
                if (Boolean.TRUE.equals(ignoreCache))     bodyBuilder.put("ignoreCache", true);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(MAP_PATH)
                        .body(bodyBuilder)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error mapping {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error mapping {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null || !Boolean.TRUE.equals(response.get("success"))) {
                    logger.warn("Invalid response mapping: {}", url);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> links =
                        (List<Map<String, Object>>) response.getOrDefault("links", Collections.emptyList());

                List<MapResult> results = links.stream()
                        .filter(l -> l != null && l.get("url") != null)
                        .map(l -> new MapResult(
                                (String) l.getOrDefault("title", ""),
                                (String) l.get("url"),
                                (String) l.getOrDefault("description", "")
                        ))
                        .toList();

                observation.highCardinalityKeyValue("firecrawl.map.urls.count", String.valueOf(results.size()));

                logger.debug("Map of '{}' found {} URLs", url, results.size());
                return JsonParser.toJson(results);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error mapping {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 5. CRAWL — Navigate entire sites without sitemaps
    // =========================================================================

    /**
     * Recursively crawls an entire site starting from a base URL.
     * Asynchronous operation — polls until complete or timeout is reached.
     */
    @Tool(name = "FirecrawlCrawl", description = """
        Recursively navigates and extracts content from an entire website,
        following internal links without requiring a sitemap.

        Use when you need comprehensive content from many pages of a site,
        such as indexing full documentation, a blog, or a wiki for RAG.

        Usage notes:
        - Async operation: may take 30s–several minutes depending on site size
        - Use Map first to preview structure, then Crawl for full content
        - Use prompt to describe what to crawl in natural language (v2 feature)
        - 1 credit per page crawled
        - Prefer Scrape for a single page, Search for ad-hoc queries
        """)
    public String crawl(
            @ToolParam(description = "Base URL to start crawling from (e.g. https://docs.example.com)") String url,
            @ToolParam(description = "Max pages to crawl (null = use configured default)", required = false) Integer limit,
            @ToolParam(description = "Natural language description of what to crawl (e.g. 'only API reference pages')", required = false) String prompt) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to crawl");
            return JsonParser.toJson(Collections.emptyList());
        }

        int effectiveLimit = (limit != null && limit > 0) ? limit : this.crawlLimit;
        logger.info("Starting crawl of '{}' (limit={})", url, effectiveLimit);

        Observation observation = Observation.createNotStarted("firecrawl.crawl", observationRegistry)
                .lowCardinalityKeyValue("firecrawl.operation", "crawl")
                .highCardinalityKeyValue("firecrawl.crawl.url", url)
                .highCardinalityKeyValue("firecrawl.crawl.limit", String.valueOf(effectiveLimit));

        return observation.observe(() -> {
            try {
                var bodyBuilder = new java.util.HashMap<String, Object>();
                bodyBuilder.put("url",   url);
                bodyBuilder.put("limit", effectiveLimit);
                bodyBuilder.put("scrapeOptions", Map.of(
                        "formats",         List.of("markdown"),
                        "onlyMainContent", true
                ));
                if (StringUtils.hasText(prompt)) bodyBuilder.put("prompt", prompt);

                @SuppressWarnings("unchecked")
                Map<String, Object> startResponse = restClient.post()
                        .uri(CRAWL_PATH)
                        .body(bodyBuilder)
                        .retrieve()
                        .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                logger.error("4xx error starting crawl of {}: {}", url, res.getStatusCode()))
                        .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                logger.error("5xx error starting crawl of {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (startResponse == null || !Boolean.TRUE.equals(startResponse.get("success"))) {
                    logger.error("Failed to start crawl of: {}", url);
                    return JsonParser.toJson(Collections.emptyList());
                }

                String crawlId = (String) startResponse.get("id");
                logger.debug("Crawl started with id: {}", crawlId);
                return pollCrawlResult(crawlId, observation);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error starting crawl of {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    private String pollCrawlResult(String crawlId, Observation observation) {
        for (int attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
            try {
                Thread.sleep(POLL_INTERVAL_MS);

                @SuppressWarnings("unchecked")
                Map<String, Object> status = restClient.get()
                        .uri(CRAWL_STATUS, crawlId)
                        .retrieve()
                        .body(Map.class);

                if (status == null) continue;

                String crawlStatus = (String) status.get("status");
                logger.debug("Crawl {} — status: {} ({}/{})", crawlId, crawlStatus, attempt, MAX_POLL_ATTEMPTS);

                if ("completed".equals(crawlStatus)) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> pages =
                            (List<Map<String, Object>>) status.getOrDefault("data", Collections.emptyList());

                    List<ScrapeResult> results = pages.stream()
                            .filter(p -> p != null && p.get("metadata") != null)
                            .map(p -> {
                                @SuppressWarnings("unchecked")
                                Map<String, Object> meta = (Map<String, Object>) p.get("metadata");
                                return new ScrapeResult(
                                        (String) meta.getOrDefault("title", ""),
                                        (String) meta.getOrDefault("sourceURL", ""),
                                        (String) p.getOrDefault("markdown", "")
                                );
                            })
                            .toList();

                    observation.highCardinalityKeyValue("firecrawl.crawl.pages.count", String.valueOf(results.size()));

                    logger.info("Crawl {} completed — {} pages", crawlId, results.size());
                    return JsonParser.toJson(results);
                }

                if ("failed".equals(crawlStatus)) {
                    logger.error("Crawl {} failed", crawlId);
                    return JsonParser.toJson(Collections.emptyList());
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("Crawl {} polling interrupted", crawlId);
                return JsonParser.toJson(Collections.emptyList());
            } catch (RestClientException e) {
                logger.warn("Error polling crawl {}: {}", crawlId, e.getMessage());
            }
        }

        logger.warn("Crawl {} timed out after {} attempts", crawlId, MAX_POLL_ATTEMPTS);
        return JsonParser.toJson(Collections.emptyList());
    }

    private ScrapeResult toScrapeResult(Map<String, Object> data) {
        if (data == null) return new ScrapeResult("", "", "");
        @SuppressWarnings("unchecked")
        Map<String, Object> meta = (Map<String, Object>) data.getOrDefault("metadata", Collections.emptyMap());
        return new ScrapeResult(
                (String) meta.getOrDefault("title", ""),
                (String) meta.getOrDefault("sourceURL", ""),
                (String) data.getOrDefault("markdown", "")
        );
    }

    private boolean matchesDomainFilter(String url, List<String> allowed, List<String> blocked) {
        if (CollectionUtils.isEmpty(allowed) && CollectionUtils.isEmpty(blocked)) return true;
        try {
            final String host = new java.net.URI(url).getHost() != null ? new java.net.URI(url).getHost().toLowerCase() : null;
            if (host == null) return false;
            //host = host.toLowerCase();
            if (!CollectionUtils.isEmpty(allowed)) {
                boolean inAllowed = allowed.stream().map(String::toLowerCase)
                        .anyMatch(d -> host.equals(d) || host.endsWith("." + d));
                if (!inAllowed) return false;
            }
            if (!CollectionUtils.isEmpty(blocked)) {
                boolean inBlocked = blocked.stream().map(String::toLowerCase)
                        .anyMatch(d -> host.equals(d) || host.endsWith("." + d));
                if (inBlocked) return false;
            }
            return true;
        } catch (java.net.URISyntaxException e) {
            logger.warn("Could not parse URL for domain filter: {}", url);
            return true;
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> castToStringList(Object obj) {
        if (obj instanceof List<?> list) {
            return list.stream()
                    .filter(String.class::isInstance)
                    .map(String.class::cast)
                    .toList();
        }
        return Collections.emptyList();
    }

    // =========================================================================
    // Result records
    // =========================================================================

    /** Result from scrape and crawl endpoints. */
    public record ScrapeResult(String title, String url, String markdown) {}

    /** Result from the search endpoint — includes full markdown content. */
    public record SearchResult(String title, String url, String description, String markdown) {}

    /** Result from the agent endpoint — natural language answer with sources. */
    public record AgentResult(String result, List<String> sources) {}

    /** Result from the map endpoint — URL discovery without content. */
    public record MapResult(String title, String url, String description) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder(String apiKey) {
        return new Builder(apiKey);
    }

    public static class Builder {

        private final String apiKey;
        private int        searchLimit = 5;
        private int        crawlLimit  = 10;
        private AgentModel agentModel  = AgentModel.SPARK_1_MINI;
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        private Builder(String apiKey) {
            Assert.hasText(apiKey, "API key must not be null or empty");
            this.apiKey = apiKey;
        }

        /** Default number of results returned by the Search tool. */
        public Builder searchLimit(int searchLimit) {
            Assert.isTrue(searchLimit > 0, "searchLimit must be positive");
            this.searchLimit = searchLimit;
            return this;
        }

        /** Default max pages for the Crawl tool. */
        public Builder crawlLimit(int crawlLimit) {
            Assert.isTrue(crawlLimit > 0, "crawlLimit must be positive");
            this.crawlLimit = crawlLimit;
            return this;
        }

        /** Model used by the Agent tool. Default: SPARK_1_MINI. */
        public Builder agentModel(AgentModel agentModel) {
            Assert.notNull(agentModel, "agentModel must not be null");
            this.agentModel = agentModel;
            return this;
        }

        /** Observation registry for metrics and tracing. Defaults to {@link ObservationRegistry#NOOP}. */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry must not be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public FirecrawlTool build() {
            return new FirecrawlTool(this.apiKey, this.searchLimit, this.crawlLimit,
                    this.agentModel, this.observationRegistry);
        }
    }
}