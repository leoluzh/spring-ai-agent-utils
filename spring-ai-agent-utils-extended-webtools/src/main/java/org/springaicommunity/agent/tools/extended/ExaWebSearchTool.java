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
import org.springframework.http.MediaType;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

/**
 * Exa Web Search Tool for Spring AI.
 * <p>
 * Provides four complementary capabilities using the Exa neural search API:
 * <ul>
 *   <li><b>Search</b>      — Neural/semantic web search with full content extraction</li>
 *   <li><b>Contents</b>    — Fetch clean content from known URLs directly</li>
 *   <li><b>FindSimilar</b> — Discover pages semantically similar to a given URL</li>
 *   <li><b>Answer</b>      — LLM-grounded answer with citations from live search</li>
 * </ul>
 *
 * <p>Exa's key differentiator is its <b>embeddings-based index</b> — unlike keyword search,
 * it finds pages by meaning, making it ideal for research, RAG pipelines, and
 * finding content that traditional search engines miss.
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public ExaWebSearchTool exaWebSearchTool(
 *         @Value("${exa.api.key}") String apiKey,
 *         ObservationRegistry observationRegistry) {
 *     return ExaWebSearchTool.builder(apiKey)
 *         .numResults(10)
 *         .searchType(ExaWebSearchTool.SearchType.AUTO)
 *         .observationRegistry(observationRegistry)
 *         .build();
 * }
 * }</pre>
 *
 * <h3>application.yaml</h3>
 * <pre>
 * exa:
 *   api:
 *     key: ${EXA_API_KEY}
 * </pre>
 *
 * @author Spring AI Community
 * @see <a href="https://docs.exa.ai">Exa API Documentation</a>
 */
public class ExaWebSearchTool {

    private static final Logger logger = LoggerFactory.getLogger(ExaWebSearchTool.class);

    private static final String BASE_URL         = "https://api.exa.ai";
    private static final String SEARCH_PATH      = "/search";
    private static final String CONTENTS_PATH    = "/contents";
    private static final String FIND_SIMILAR_PATH = "/findSimilar";
    private static final String ANSWER_PATH      = "/answer";

    private final RestClient restClient;
    private final int        numResults;
    private final String     searchType;
    private final ObservationRegistry observationRegistry;

    /**
     * Search type for the Exa search endpoint.
     * <p>
     * Exa offers five search modes with different latency/quality trade-offs:
     * <ul>
     *   <li>{@code AUTO}    — balanced, default (combines neural + other methods)</li>
     *   <li>{@code NEURAL}  — pure embedding-based semantic search</li>
     *   <li>{@code FAST}    — sub-350ms, optimized for real-time applications</li>
     *   <li>{@code DEEP}    — agentic multi-pass search, highest quality (~3.5s)</li>
     *   <li>{@code INSTANT} — lowest latency, optimized for real-time</li>
     * </ul>
     */
    public enum SearchType {
        AUTO("auto"),
        NEURAL("neural"),
        FAST("fast"),
        DEEP("deep"),
        INSTANT("instant");

        private final String value;
        SearchType(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Content category filter for focused search.
     */
    public enum Category {
        COMPANY("company"),
        RESEARCH_PAPER("research paper"),
        NEWS("news"),
        TWEET("tweet"),
        PERSONAL_SITE("personal site"),
        FINANCIAL_REPORT("financial report"),
        PEOPLE("people");

        private final String value;
        Category(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private ExaWebSearchTool(String apiKey, int numResults, SearchType searchType,
                             ObservationRegistry observationRegistry) {
        Assert.hasText(apiKey, "Exa API key must not be null or empty");
        this.restClient = RestClient.builder()
                .baseUrl(BASE_URL)
                .defaultHeader("x-api-key",    apiKey)
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("Accept",       MediaType.APPLICATION_JSON_VALUE)
                .build();
        this.numResults = numResults;
        this.searchType = searchType.getValue();
        this.observationRegistry = observationRegistry;
    }

    // =========================================================================
    // 1. SEARCH — Neural/semantic web search
    // =========================================================================

    /**
     * Searches the web using Exa's neural (embedding-based) search engine.
     * Unlike keyword search, Exa finds pages by meaning — ideal for research queries.
     */
    @Tool(name = "ExaSearch", description = """
        Searches the web using Exa's neural (embedding-based) search engine.
        Unlike keyword search, Exa finds pages by MEANING — ideal for research,
        finding opinions, academic papers, news, or content traditional search misses.

        Supports filtering by domain, date range, and content category
        (news, research paper, company, tweet, personal site, financial report).

        Returns: title, URL, published date, author, content text, and highlights.

        Usage notes:
        - Phrase queries as statements, not questions: "latest advances in transformers"
          rather than "what are the latest advances in transformers?"
        - Use includeDomains for trusted sources (e.g. ["arxiv.org", "nature.com"])
        - Use category for focused searches (e.g. "research paper", "news")
        - For real-time info, combine with startPublishedDate filter
        - After responding, include a "Sources:" section with markdown links
        """)
    public String search(
            @ToolParam(description = "Search query — phrase as a statement for best neural results") String query,
            @ToolParam(description = "Number of results (null = configured default, max 100)", required = false) Integer numResults,
            @ToolParam(description = "Search mode: auto, neural, fast, deep, instant (null = auto)", required = false) String type,
            @ToolParam(description = "Content category filter: company, research paper, news, tweet, personal site, financial report, people", required = false) String category,
            @ToolParam(description = "Only include results from these domains (e.g. [\"arxiv.org\"])", required = false) List<String> includeDomains,
            @ToolParam(description = "Exclude results from these domains", required = false) List<String> excludeDomains,
            @ToolParam(description = "Only include results published after this date (ISO 8601, e.g. 2025-01-01T00:00:00Z)", required = false) String startPublishedDate,
            @ToolParam(description = "Only include results published before this date (ISO 8601)", required = false) String endPublishedDate) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to ExaSearch");
            return JsonParser.toJson(Collections.emptyList());
        }

        int effectiveLimit = (numResults != null && numResults > 0) ? numResults : this.numResults;
        String effectiveType = StringUtils.hasText(type) ? type : this.searchType;

        logger.debug("Exa search: '{}' (type={}, limit={})", query, effectiveType, effectiveLimit);

        Observation observation = Observation.createNotStarted("exa.search", observationRegistry)
                .lowCardinalityKeyValue("exa.operation", "search")
                .lowCardinalityKeyValue("exa.search.type", effectiveType)
                .lowCardinalityKeyValue("exa.search.category", category != null ? category : "none")
                .highCardinalityKeyValue("exa.search.query", query)
                .highCardinalityKeyValue("exa.search.limit", String.valueOf(effectiveLimit));

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("query",      query);
                body.put("numResults", effectiveLimit);
                body.put("type",       effectiveType);
                body.put("contents",   Map.of(
                        "text",       true,
                        "highlights", Map.of("maxCharacters", 1000)
                ));

                if (StringUtils.hasText(category))         body.put("category",           category);
                if (!CollectionUtils.isEmpty(includeDomains)) body.put("includeDomains",  includeDomains);
                if (!CollectionUtils.isEmpty(excludeDomains)) body.put("excludeDomains",  excludeDomains);
                if (StringUtils.hasText(startPublishedDate))  body.put("startPublishedDate", startPublishedDate);
                if (StringUtils.hasText(endPublishedDate))    body.put("endPublishedDate",   endPublishedDate);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(SEARCH_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(s -> s.is4xxClientError(), (req, res) ->
                                logger.error("4xx error in ExaSearch for '{}': {}", query, res.getStatusCode()))
                        .onStatus(s -> s.is5xxServerError(), (req, res) ->
                                logger.error("5xx error in ExaSearch for '{}': {}", query, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from ExaSearch for query: {}", query);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> results =
                        (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

                List<SearchResult> parsed = results.stream()
                        .filter(r -> r != null && r.get("url") != null)
                        .map(this::toSearchResult)
                        .toList();

                observation.highCardinalityKeyValue("exa.search.results.count", String.valueOf(parsed.size()));

                logger.debug("ExaSearch '{}' returned {} results", query, parsed.size());
                return JsonParser.toJson(parsed);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in ExaSearch for '{}': {}", query, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 2. CONTENTS — Fetch clean content from known URLs
    // =========================================================================

    /**
     * Fetches clean, parsed content from a list of known URLs.
     * Cheaper than search when the URLs are already known.
     */
    @Tool(name = "ExaContents", description = """
        Fetches clean, LLM-ready content (text + highlights + summary) from a
        list of known URLs without performing a search.

        Use when you already have the URLs and just need their content —
        more efficient than running a search for pages you already know.

        Typical workflow:
        1. Use ExaSearch or ExaFindSimilar to discover URLs
        2. Use ExaContents to get the full content of selected URLs

        Returns: title, URL, published date, author, text, and highlights per URL.
        """)
    public String contents(
            @ToolParam(description = "List of URLs to fetch content from") List<String> urls,
            @ToolParam(description = "If true, include AI-generated summary of each page", required = false) Boolean includeSummary) {

        if (CollectionUtils.isEmpty(urls)) {
            logger.warn("Empty URL list provided to ExaContents");
            return JsonParser.toJson(Collections.emptyList());
        }

        logger.debug("ExaContents fetching {} URLs", urls.size());

        Observation observation = Observation.createNotStarted("exa.contents", observationRegistry)
                .lowCardinalityKeyValue("exa.operation", "contents")
                .highCardinalityKeyValue("exa.contents.url_count", String.valueOf(urls.size()));

        return observation.observe(() -> {
            try {
                var contentsOptions = new java.util.HashMap<String, Object>();
                contentsOptions.put("text",       true);
                contentsOptions.put("highlights", Map.of("maxCharacters", 1000));
                if (Boolean.TRUE.equals(includeSummary)) contentsOptions.put("summary", true);

                Map<String, Object> body = Map.of(
                        "ids",      urls,
                        "contents", contentsOptions
                );

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(CONTENTS_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(s -> s.is4xxClientError(), (req, res) ->
                                logger.error("4xx error in ExaContents: {}", res.getStatusCode()))
                        .onStatus(s -> s.is5xxServerError(), (req, res) ->
                                logger.error("5xx error in ExaContents: {}", res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from ExaContents");
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> results =
                        (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

                List<SearchResult> parsed = results.stream()
                        .filter(r -> r != null && r.get("url") != null)
                        .map(this::toSearchResult)
                        .toList();

                observation.highCardinalityKeyValue("exa.contents.returned", String.valueOf(parsed.size()));

                logger.debug("ExaContents returned {} pages", parsed.size());
                return JsonParser.toJson(parsed);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in ExaContents: {}", e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });
    }

    // =========================================================================
    // 3. FIND SIMILAR — Discover semantically similar pages
    // =========================================================================

    /**
     * Finds pages semantically similar to a given URL using Exa's neural index.
     */
    @Tool(name = "ExaFindSimilar", description = """
        Finds web pages semantically similar to a given URL using Exa's neural index.
        Use when you found a good source and want to discover related content —
        papers citing the same ideas, articles covering similar topics, competitors, etc.

        Examples:
        - Found a great research paper → find more papers on the same topic
        - Found a company's page → find similar companies
        - Found a blog post → find related posts from other authors

        Returns: list of similar pages with title, URL, published date, and content.

        Usage notes:
        - Set excludeSourceDomain=true to avoid returning pages from the same site
        - Combine with includeDomains to restrict to trusted sources
        - Use category filter to narrow to a content type (e.g. "research paper")
        """)
    public String findSimilar(
            @ToolParam(description = "URL to find similar pages for") String url,
            @ToolParam(description = "Number of similar results to return (null = configured default)", required = false) Integer numResults,
            @ToolParam(description = "If true, excludes results from the same domain as the input URL", required = false) Boolean excludeSourceDomain,
            @ToolParam(description = "Only include results from these domains", required = false) List<String> includeDomains,
            @ToolParam(description = "Exclude results from these domains", required = false) List<String> excludeDomains,
            @ToolParam(description = "Content category filter: company, research paper, news, tweet, personal site, financial report", required = false) String category) {

        if (!StringUtils.hasText(url)) {
            logger.warn("Empty URL provided to ExaFindSimilar");
            return JsonParser.toJson(Collections.emptyList());
        }

        int effectiveLimit = (numResults != null && numResults > 0) ? numResults : this.numResults;
        logger.debug("ExaFindSimilar for: {} (limit={})", url, effectiveLimit);

        Observation observation = Observation.createNotStarted("exa.find_similar", observationRegistry)
                .lowCardinalityKeyValue("exa.operation", "find_similar")
                .highCardinalityKeyValue("exa.find_similar.url", url)
                .highCardinalityKeyValue("exa.find_similar.limit", String.valueOf(effectiveLimit));

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("url",        url);
                body.put("numResults", effectiveLimit);
                body.put("contents",   Map.of(
                        "text",       true,
                        "highlights", Map.of("maxCharacters", 800)
                ));

                if (Boolean.TRUE.equals(excludeSourceDomain)) body.put("excludeSourceDomain", true);
                if (!CollectionUtils.isEmpty(includeDomains))  body.put("includeDomains",      includeDomains);
                if (!CollectionUtils.isEmpty(excludeDomains))  body.put("excludeDomains",      excludeDomains);
                if (StringUtils.hasText(category))             body.put("category",            category);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(FIND_SIMILAR_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(s -> s.is4xxClientError(), (req, res) ->
                                logger.error("4xx error in ExaFindSimilar for {}: {}", url, res.getStatusCode()))
                        .onStatus(s -> s.is5xxServerError(), (req, res) ->
                                logger.error("5xx error in ExaFindSimilar for {}: {}", url, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from ExaFindSimilar for: {}", url);
                    return JsonParser.toJson(Collections.emptyList());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> results =
                        (List<Map<String, Object>>) response.getOrDefault("results", Collections.emptyList());

                List<SearchResult> parsed = results.stream()
                        .filter(r -> r != null && r.get("url") != null)
                        .map(this::toSearchResult)
                        .toList();

                observation.highCardinalityKeyValue("exa.find_similar.returned", String.valueOf(parsed.size()));

                logger.debug("ExaFindSimilar for '{}' returned {} results", url, parsed.size());
                return JsonParser.toJson(parsed);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in ExaFindSimilar for {}: {}", url, e.getMessage());
                return JsonParser.toJson(Collections.emptyList());
            }
        });

    }

    // =========================================================================
    // 4. ANSWER — LLM-grounded answer with citations
    // =========================================================================

    /**
     * Gets an LLM-generated answer grounded in live Exa search results with citations.
     */
    @Tool(name = "ExaAnswer", description = """
        Gets an LLM-generated answer to a question, grounded in live Exa search results.
        Returns a direct answer for factual questions or a detailed summary with citations
        for open-ended research questions.

        Use when you need:
        - A concise, cited answer to a factual question
        - A research summary with source attribution
        - Current information beyond your training cutoff

        Examples:
        - "What are the latest findings on gut microbiome's influence on mental health?"
        - "What is the current state of fusion energy research?"
        - "Who won the 2025 Nobel Prize in Physics?"

        Returns: answer text and list of citation sources with URLs.

        Usage notes:
        - More convenient than Search + manual synthesis for Q&A use cases
        - Always include the citations section from the response in your reply
        - For deep research, prefer ExaSearch with type=deep for more control
        """)
    public String answer(
            @ToolParam(description = "Question to answer — can be factual or open-ended research") String query,
            @ToolParam(description = "If true, includes full text of source pages in response (increases cost)", required = false) Boolean includeText) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to ExaAnswer");
            return JsonParser.toJson(Collections.emptyMap());
        }

        logger.debug("ExaAnswer: '{}'", query);

        Observation observation = Observation.createNotStarted("exa.answer", observationRegistry)
                .lowCardinalityKeyValue("exa.operation", "answer")
                .highCardinalityKeyValue("exa.answer.query", query);

        return observation.observe(() -> {
            try {
                var body = new java.util.HashMap<String, Object>();
                body.put("query", query);
                if (Boolean.TRUE.equals(includeText)) body.put("text", true);

                @SuppressWarnings("unchecked")
                Map<String, Object> response = restClient.post()
                        .uri(ANSWER_PATH)
                        .body(body)
                        .retrieve()
                        .onStatus(s -> s.is4xxClientError(), (req, res) ->
                                logger.error("4xx error in ExaAnswer for '{}': {}", query, res.getStatusCode()))
                        .onStatus(s -> s.is5xxServerError(), (req, res) ->
                                logger.error("5xx error in ExaAnswer for '{}': {}", query, res.getStatusCode()))
                        .body(Map.class);

                if (response == null) {
                    logger.warn("Null response from ExaAnswer for: {}", query);
                    return JsonParser.toJson(Collections.emptyMap());
                }

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> citationMaps =
                        (List<Map<String, Object>>) response.getOrDefault("citations", Collections.emptyList());

                List<Citation> citations = citationMaps.stream()
                        .filter(c -> c != null && c.get("url") != null)
                        .map(c -> new Citation(
                                (String) c.getOrDefault("title", ""),
                                (String) c.get("url"),
                                (String) c.getOrDefault("publishedDate", ""),
                                (String) c.getOrDefault("author", ""),
                                (String) c.getOrDefault("text", "")
                        ))
                        .toList();

                AnswerResult result = new AnswerResult(
                        (String) response.getOrDefault("answer", ""),
                        citations
                );

                observation.highCardinalityKeyValue("exa.answer.citations.count", String.valueOf(citations.size()));

                logger.debug("ExaAnswer returned answer with {} citations", citations.size());
                return JsonParser.toJson(result);

            } catch (RestClientException e) {
                observation.error(e);
                logger.error("Error in ExaAnswer for '{}': {}", query, e.getMessage());
                return JsonParser.toJson(Collections.emptyMap());
            }
        });

    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    @SuppressWarnings("unchecked")
    private SearchResult toSearchResult(Map<String, Object> r) {
        List<String> highlights = (List<String>) r.getOrDefault("highlights", Collections.emptyList());
        return new SearchResult(
                (String) r.getOrDefault("title",         ""),
                (String) r.getOrDefault("url",           ""),
                (String) r.getOrDefault("publishedDate", ""),
                (String) r.getOrDefault("author",        ""),
                (String) r.getOrDefault("text",          ""),
                (String) r.getOrDefault("summary",       ""),
                highlights
        );
    }

    // =========================================================================
    // Result records
    // =========================================================================

    /** Result from search, contents, and findSimilar endpoints. */
    public record SearchResult(
            String title,
            String url,
            String publishedDate,
            String author,
            String text,
            String summary,
            List<String> highlights
    ) {}

    /** Answer from the /answer endpoint with grounded citations. */
    public record AnswerResult(String answer, List<Citation> citations) {}

    /** A single source citation returned by /answer. */
    public record Citation(
            String title,
            String url,
            String publishedDate,
            String author,
            String text
    ) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder(String apiKey) {
        return new Builder(apiKey);
    }

    public static class Builder {

        private final String apiKey;
        private int        numResults = 10;
        private SearchType searchType = SearchType.AUTO;
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        private Builder(String apiKey) {
            Assert.hasText(apiKey, "API key must not be null or empty");
            this.apiKey = apiKey;
        }

        /** Default number of results for search and findSimilar. */
        public Builder numResults(int numResults) {
            Assert.isTrue(numResults > 0, "numResults must be positive");
            this.numResults = numResults;
            return this;
        }

        /** Default search mode. See {@link SearchType} for options. */
        public Builder searchType(SearchType searchType) {
            Assert.notNull(searchType, "searchType must not be null");
            this.searchType = searchType;
            return this;
        }

        /** Observation registry for metrics and tracing. Defaults to {@link ObservationRegistry#NOOP}. */
        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry must not be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public ExaWebSearchTool build() {
            return new ExaWebSearchTool(this.apiKey, this.numResults, this.searchType,this.observationRegistry);
        }
    }
}