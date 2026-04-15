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
import org.springframework.web.util.UriComponentsBuilder;

/**
 * DuckDuckGo Search Tool for Spring AI.
 * <p>
 * Wraps two distinct DuckDuckGo endpoints — both <b>free with no API key required</b>:
 *
 * <ul>
 *   <li><b>Instant Answer API</b> ({@code api.duckduckgo.com}) — official, returns
 *       structured instant answers: abstract summaries, definitions, infobox facts,
 *       related topics and !bang redirects. Does NOT return full web search results.</li>
 *   <li><b>Web Search</b> ({@code html.duckduckgo.com} / {@code lite.duckduckgo.com})
 *       — unofficial scraping of DDG's HTML/Lite interfaces returning full organic
 *       results, news, and more. Rate-limited; use a proxy for high-volume usage.</li>
 * </ul>
 *
 * <p><b>Privacy:</b> DuckDuckGo does not track users or store personal information.
 * No API key, registration, or account required for either endpoint.
 *
 * <h3>Spring Boot Configuration</h3>
 * <pre>{@code
 * @Bean
 * public DuckDuckGoSearchTool duckDuckGoSearchTool(ObservationRegistry observationRegistry) {
 *     return DuckDuckGoSearchTool.builder()
 *         .maxResults(10)
 *         .defaultRegion("us-en")
 *         .observationRegistry(observationRegistry)
 *         .build();
 * }
 * }</pre>
 *
 * @author Spring AI Community
 * @see <a href="https://duckduckgo.com/duckduckgo-help-pages/results/instant-answer-api/">
 *      DuckDuckGo Instant Answer API</a>
 */
public class DuckDuckGoSearchTool {

    private static final Logger logger = LoggerFactory.getLogger(DuckDuckGoSearchTool.class);

    // ── Official Instant Answer API (no key, no rate-limit concerns) ──────────
    private static final String INSTANT_ANSWER_BASE = "https://api.duckduckgo.com";

    // ── Unofficial HTML / Lite endpoints (scraping, rate-limited) ────────────
    private static final String HTML_SEARCH_BASE    = "https://html.duckduckgo.com";
    private static final String LITE_SEARCH_BASE    = "https://lite.duckduckgo.com";

    // ── Autocomplete suggestions ──────────────────────────────────────────────
    private static final String SUGGEST_BASE        = "https://ac.duckduckgo.com";

    private final RestClient instantClient;   // for api.duckduckgo.com
    private final RestClient htmlClient;      // for html.duckduckgo.com
    private final RestClient liteClient;      // for lite.duckduckgo.com
    private final RestClient suggestClient;   // for ac.duckduckgo.com

    private final int    maxResults;
    private final String defaultRegion;
    private final ObservationRegistry observationRegistry;

    /**
     * Time limit filter for search results.
     */
    public enum TimeLimit {
        DAY("d"),
        WEEK("w"),
        MONTH("m"),
        YEAR("y");

        private final String value;
        TimeLimit(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    /**
     * Safe search level.
     */
    public enum SafeSearch {
        ON("1"),
        MODERATE("-1"),
        OFF("-2");

        private final String value;
        SafeSearch(String value) { this.value = value; }
        public String getValue() { return value; }
    }

    private DuckDuckGoSearchTool(int maxResults, String defaultRegion, ObservationRegistry observationRegistry) {
        // Shared headers that mimic a real browser — reduces rate-limiting risk
        var baseBuilder = RestClient.builder()
                .defaultHeader("User-Agent",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
                                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                .defaultHeader("Accept-Language", "en-US,en;q=0.9")
                .defaultHeader("Accept",
                        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");

        this.instantClient = RestClient.builder()
                .baseUrl(INSTANT_ANSWER_BASE)
                .defaultHeader("Accept", MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader("User-Agent",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
                                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                .build();

        this.htmlClient    = baseBuilder.baseUrl(HTML_SEARCH_BASE).build();
        this.liteClient    = baseBuilder.baseUrl(LITE_SEARCH_BASE).build();
        this.suggestClient = RestClient.builder()
                .baseUrl(SUGGEST_BASE)
                .defaultHeader("Accept", MediaType.APPLICATION_JSON_VALUE)
                .build();

        this.maxResults    = maxResults;
        this.defaultRegion = defaultRegion;
        this.observationRegistry = observationRegistry;
    }

    // =========================================================================
    // 1. INSTANT ANSWER — Official DDG API (abstract, definition, infobox)
    // =========================================================================

    @Tool(name = "duckduckgo_instant_answer", description = """
        Queries DuckDuckGo's official Instant Answer API (api.duckduckgo.com).
        Returns structured instant answers: abstract summaries, definitions,
        infobox facts, related topics, and !bang redirects.

        FREE — no API key, no account, no rate-limit concerns.

        Best for:
        - Definitions and concept explanations ("what is photosynthesis")
        - Entity summaries (people, places, organizations, films)
        - Quick factual lookups (capital cities, unit conversions, formulas)
        - Related topics discovery

        IMPORTANT LIMITATION: This is NOT a full web search API. It does not
        return a list of ranked web pages — only DuckDuckGo's curated instant
        answers. For full web results, use DuckDuckGoWebSearch instead.

        Returns: abstract text, abstract URL, definition, answer, infobox facts,
        and a list of related topic titles and URLs.
        """)
    public String instantAnswer(
            @ToolParam(description = "Search query (entity, concept, or question)") String query,
            @ToolParam(description = "Skip disambiguation pages and return the first result directly", required = false) Boolean skipDisambig) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to DuckDuckGoInstantAnswer");
            return JsonParser.toJson(Collections.emptyMap());
        }

        return Observation.createNotStarted("spring.ai.tool", this.observationRegistry)
                .contextualName("duckduckgo-instant-answer")
                .lowCardinalityKeyValue("tool.name", "DuckDuckGoInstantAnswer")
                .observe(() -> {
                    logger.debug("DuckDuckGoInstantAnswer: '{}'", query);

                    try {
                        UriComponentsBuilder uriBuilder = UriComponentsBuilder.newInstance()
                                .path("/")
                                .queryParam("q",        query)
                                .queryParam("format",   "json")
                                .queryParam("no_html",  "1")
                                .queryParam("no_redirect", "1");

                        if (Boolean.TRUE.equals(skipDisambig)) {
                            uriBuilder.queryParam("skip_disambig", "1");
                        }

                        String uri = uriBuilder.toUriString();

                        @SuppressWarnings("unchecked")
                        Map<String, Object> raw = instantClient.get()
                                .uri(uri)
                                .retrieve()
                                .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                        logger.error("4xx error in DuckDuckGoInstantAnswer '{}': {}", query, res.getStatusCode()))
                                .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                        logger.error("5xx error in DuckDuckGoInstantAnswer '{}': {}", query, res.getStatusCode()))
                                .body(Map.class);

                        if (raw == null) {
                            logger.warn("Null response from DuckDuckGoInstantAnswer for: {}", query);
                            return JsonParser.toJson(Collections.emptyMap());
                        }

                        // Parse RelatedTopics (may contain nested groups)
                        @SuppressWarnings("unchecked")
                        List<Object> relatedRaw =
                                (List<Object>) raw.getOrDefault("RelatedTopics", Collections.emptyList());

                        List<RelatedTopic> relatedTopics = relatedRaw.stream()
                                .filter(item -> item instanceof Map)
                                .map(item -> {
                                    @SuppressWarnings("unchecked")
                                    Map<String, Object> m = (Map<String, Object>) item;
                                    // Items may be flat topics or nested topic groups
                                    if (m.containsKey("FirstURL")) {
                                        return new RelatedTopic(
                                                (String) m.getOrDefault("Text",     ""),
                                                (String) m.getOrDefault("FirstURL", "")
                                        );
                                    }
                                    return null;
                                })
                                .filter(t -> t != null)
                                .toList();

                        // Parse Infobox if present
                        @SuppressWarnings("unchecked")
                        Map<String, Object> infoboxRaw =
                                (Map<String, Object>) raw.getOrDefault("Infobox", Collections.emptyMap());

                        List<InfoboxEntry> infoboxEntries = Collections.emptyList();
                        if (!CollectionUtils.isEmpty(infoboxRaw)) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> content =
                                    (List<Map<String, Object>>) infoboxRaw.getOrDefault("content", Collections.emptyList());
                            infoboxEntries = content.stream()
                                    .filter(e -> e != null && e.get("label") != null)
                                    .map(e -> new InfoboxEntry(
                                            (String) e.getOrDefault("label", ""),
                                            String.valueOf(e.getOrDefault("value", ""))
                                    ))
                                    .toList();
                        }

                        InstantAnswerResult result = new InstantAnswerResult(
                                (String) raw.getOrDefault("AbstractText",   ""),
                                (String) raw.getOrDefault("AbstractURL",    ""),
                                (String) raw.getOrDefault("AbstractSource", ""),
                                (String) raw.getOrDefault("Definition",     ""),
                                (String) raw.getOrDefault("DefinitionURL",  ""),
                                (String) raw.getOrDefault("Answer",         ""),
                                (String) raw.getOrDefault("AnswerType",     ""),
                                (String) raw.getOrDefault("Type",           ""),
                                relatedTopics,
                                infoboxEntries
                        );

                        logger.debug("DuckDuckGoInstantAnswer: type={}, relatedTopics={}",
                                result.type(), result.relatedTopics().size());
                        return JsonParser.toJson(result);

                    } catch (RestClientException e) {
                        logger.error("Error in DuckDuckGoInstantAnswer for '{}': {}", query, e.getMessage());
                        return JsonParser.toJson(Collections.emptyMap());
                    }
                });
    }

    // =========================================================================
    // 2. WEB SEARCH — Lite HTML endpoint (unofficial, full results)
    // =========================================================================

    @Tool(name = "duckduckgo_search", description = """
        Searches the web via DuckDuckGo's lite interface and returns organic results.
        Returns titles, URLs, and snippets of matching web pages.

        FREE — no API key required. Privacy-focused: no tracking, no ads profiling.

        Use for:
        - General web searches when exact factual answers aren't needed
        - Finding documentation, tutorials, or blog posts
        - Discovering relevant web pages to scrape for full content

        Supports advanced search operators in the query:
        - site:example.com — restrict to a domain
        - intitle:keyword — keyword in page title
        - inurl:keyword   — keyword in URL
        - filetype:pdf    — find PDF files
        - "exact phrase"  — exact match

        Usage notes:
        - Rate-limited by DDG — avoid high-frequency automated calls
        - Use timelimit to restrict to recent results (d=day, w=week, m=month, y=year)
        - Region format: us-en, uk-en, br-pt, de-de, fr-fr, wt-wt (worldwide)
        - This is UNOFFICIAL scraping — may break if DDG changes their HTML structure
        """)
    public String search(
            @ToolParam(description = "Search query. Supports site:, intitle:, inurl:, filetype:, and \"exact phrase\" operators.") String query,
            @ToolParam(description = "Max results to return (null = configured default)", required = false) Integer maxResults,
            @ToolParam(description = "Region/language code: us-en, uk-en, br-pt, de-de, wt-wt (worldwide). Null = configured default.", required = false) String region,
            @ToolParam(description = "Time limit: d (day), w (week), m (month), y (year). Null = no limit.", required = false) String timelimit,
            @ToolParam(description = "Safe search: on, moderate, off. Null = moderate.", required = false) String safesearch) {
        return webSearch(query, maxResults, region, timelimit, safesearch);
    }


    @Tool(name = "duckduckgo_web_search", description = """
        Searches the web via DuckDuckGo's lite interface and returns organic results.
        Returns titles, URLs, and snippets of matching web pages.

        FREE — no API key required. Privacy-focused: no tracking, no ads profiling.

        Use for:
        - General web searches when exact factual answers aren't needed
        - Finding documentation, tutorials, or blog posts
        - Discovering relevant web pages to scrape for full content

        Supports advanced search operators in the query:
        - site:example.com — restrict to a domain
        - intitle:keyword — keyword in page title
        - inurl:keyword   — keyword in URL
        - filetype:pdf    — find PDF files
        - "exact phrase"  — exact match

        Usage notes:
        - Rate-limited by DDG — avoid high-frequency automated calls
        - Use timelimit to restrict to recent results (d=day, w=week, m=month, y=year)
        - Region format: us-en, uk-en, br-pt, de-de, fr-fr, wt-wt (worldwide)
        - This is UNOFFICIAL scraping — may break if DDG changes their HTML structure
        """)
    public String webSearch(
            @ToolParam(description = "Search query. Supports site:, intitle:, inurl:, filetype:, and \"exact phrase\" operators.") String query,
            @ToolParam(description = "Max results to return (null = configured default)", required = false) Integer maxResults,
            @ToolParam(description = "Region/language code: us-en, uk-en, br-pt, de-de, wt-wt (worldwide). Null = configured default.", required = false) String region,
            @ToolParam(description = "Time limit: d (day), w (week), m (month), y (year). Null = no limit.", required = false) String timelimit,
            @ToolParam(description = "Safe search: on, moderate, off. Null = moderate.", required = false) String safesearch) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to DuckDuckGoWebSearch");
            return JsonParser.toJson(Collections.emptyList());
        }

        return Observation.createNotStarted("spring.ai.tool", this.observationRegistry)
                .contextualName("duckduckgo-web-search")
                .lowCardinalityKeyValue("tool.name", "DuckDuckGoWebSearch")
                .observe(() -> {
                    int    effectiveMax    = (maxResults != null && maxResults > 0) ? maxResults : this.maxResults;
                    String effectiveRegion = StringUtils.hasText(region) ? region : this.defaultRegion;

                    logger.debug("DuckDuckGoWebSearch: '{}' (region={}, timelimit={})",
                            query, effectiveRegion, timelimit);

                    try {
                        // DDG Lite uses a simple HTML POST form — parse results from the HTML response
                        UriComponentsBuilder uriBuilder = UriComponentsBuilder.newInstance()
                                .path("/lite/")
                                .queryParam("q",  query)
                                .queryParam("kl", effectiveRegion);    // region/language

                        if (StringUtils.hasText(timelimit)) uriBuilder.queryParam("df", timelimit);
                        if (StringUtils.hasText(safesearch)) {
                            String safeValue = switch (safesearch.toLowerCase()) {
                                case "on"   -> "1";
                                case "off"  -> "-2";
                                default     -> "-1"; // moderate
                            };
                            uriBuilder.queryParam("kp", safeValue);
                        }

                        String html = liteClient.get()
                                .uri(uriBuilder.toUriString())
                                .header("Accept", "text/html")
                                .retrieve()
                                .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                        logger.error("4xx error in DuckDuckGoWebSearch '{}': {}", query, res.getStatusCode()))
                                .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                        logger.error("5xx error in DuckDuckGoWebSearch '{}': {}", query, res.getStatusCode()))
                                .body(String.class);

                        if (!StringUtils.hasText(html)) {
                            logger.warn("Empty HTML response from DuckDuckGoWebSearch for: {}", query);
                            return JsonParser.toJson(Collections.emptyList());
                        }

                        List<WebResult> results = parseLiteHtml(html, effectiveMax);
                        logger.debug("DuckDuckGoWebSearch '{}' returned {} results", query, results.size());
                        return JsonParser.toJson(results);

                    } catch (RestClientException e) {
                        logger.error("Error in DuckDuckGoWebSearch for '{}': {}", query, e.getMessage());
                        return JsonParser.toJson(Collections.emptyList());
                    }
                });
    }

    // =========================================================================
    // 3. NEWS SEARCH — DDG News via HTML endpoint
    // =========================================================================

    @Tool(name = "duckduckgo_news_search", description = """
        Searches DuckDuckGo News and returns recent news articles with titles,
        URLs, sources, and publication dates.

        FREE — no API key required. Privacy-focused.

        Use for:
        - Breaking news and current events
        - Recent developments on any topic
        - Monitoring a company, person, or subject in the news

        Usage notes:
        - Combine with timelimit=d for news from the last 24 hours
        - Region affects which news sources are prioritized
        - Returns source name and date alongside URL and snippet
        """)
    public String newsSearch(
            @ToolParam(description = "News search query") String query,
            @ToolParam(description = "Max results to return (null = configured default)", required = false) Integer maxResults,
            @ToolParam(description = "Region/language code: us-en, uk-en, br-pt, wt-wt. Null = configured default.", required = false) String region,
            @ToolParam(description = "Time limit: d (day), w (week), m (month). Null = no limit.", required = false) String timelimit) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to DuckDuckGoNewsSearch");
            return JsonParser.toJson(Collections.emptyList());
        }

        return Observation.createNotStarted("spring.ai.tool", this.observationRegistry)
                .contextualName("duckduckgo-news-search")
                .lowCardinalityKeyValue("tool.name", "DuckDuckGoNewsSearch")
                .observe(() -> {
                    int    effectiveMax    = (maxResults != null && maxResults > 0) ? maxResults : this.maxResults;
                    String effectiveRegion = StringUtils.hasText(region) ? region : this.defaultRegion;

                    logger.debug("DuckDuckGoNewsSearch: '{}' (region={}, timelimit={})",
                            query, effectiveRegion, timelimit);

                    try {
                        // DDG News uses the same lite endpoint with ia=news parameter
                        UriComponentsBuilder uriBuilder = UriComponentsBuilder.newInstance()
                                .path("/lite/")
                                .queryParam("q",       query)
                                .queryParam("kl",      effectiveRegion)
                                .queryParam("ia",      "news")
                                .queryParam("iar",     "news");

                        if (StringUtils.hasText(timelimit)) uriBuilder.queryParam("df", timelimit);

                        String html = liteClient.get()
                                .uri(uriBuilder.toUriString())
                                .header("Accept", "text/html")
                                .retrieve()
                                .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                        logger.error("4xx error in DuckDuckGoNewsSearch '{}': {}", query, res.getStatusCode()))
                                .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                        logger.error("5xx error in DuckDuckGoNewsSearch '{}': {}", query, res.getStatusCode()))
                                .body(String.class);

                        if (!StringUtils.hasText(html)) {
                            logger.warn("Empty HTML response from DuckDuckGoNewsSearch for: {}", query);
                            return JsonParser.toJson(Collections.emptyList());
                        }

                        List<WebResult> results = parseLiteHtml(html, effectiveMax);
                        logger.debug("DuckDuckGoNewsSearch '{}' returned {} results", query, results.size());
                        return JsonParser.toJson(results);

                    } catch (RestClientException e) {
                        logger.error("Error in DuckDuckGoNewsSearch for '{}': {}", query, e.getMessage());
                        return JsonParser.toJson(Collections.emptyList());
                    }
                });
    }

    // =========================================================================
    // 4. SUGGESTIONS — Autocomplete query suggestions
    // =========================================================================

    @Tool(name = "duckduckgo_suggestions", description = """
        Returns DuckDuckGo autocomplete suggestions for a partial query.

        FREE — no API key required.

        Use for:
        - Discovering related search terms before running a full search
        - Expanding a vague topic into specific searchable queries
        - Understanding how DuckDuckGo interprets a topic

        Returns a list of suggested query phrases ordered by relevance.
        """)
    public String suggestions(
            @ToolParam(description = "Partial query to get suggestions for") String query,
            @ToolParam(description = "Region/language code: us-en, br-pt, wt-wt. Null = configured default.", required = false) String region) {

        if (!StringUtils.hasText(query)) {
            logger.warn("Empty query provided to DuckDuckGoSuggestions");
            return JsonParser.toJson(Collections.emptyList());
        }

        return Observation.createNotStarted("spring.ai.tool", this.observationRegistry)
                .contextualName("duckduckgo-suggestions")
                .lowCardinalityKeyValue("tool.name", "DuckDuckGoSuggestions")
                .observe(() -> {
                    String effectiveRegion = StringUtils.hasText(region) ? region : this.defaultRegion;
                    logger.debug("DuckDuckGoSuggestions: '{}' (region={})", query, effectiveRegion);

                    try {
                        String uri = UriComponentsBuilder.newInstance()
                                .path("/ac/")
                                .queryParam("q",  query)
                                .queryParam("kl", effectiveRegion)
                                .toUriString();

                        @SuppressWarnings("unchecked")
                        List<Map<String, Object>> raw = suggestClient.get()
                                .uri(uri)
                                .retrieve()
                                .onStatus(HttpStatusCode::is4xxClientError, (req, res) ->
                                        logger.error("4xx in DuckDuckGoSuggestions '{}': {}", query, res.getStatusCode()))
                                .onStatus(HttpStatusCode::is5xxServerError, (req, res) ->
                                        logger.error("5xx in DuckDuckGoSuggestions '{}': {}", query, res.getStatusCode()))
                                .body(List.class);

                        if (CollectionUtils.isEmpty(raw)) {
                            return JsonParser.toJson(Collections.emptyList());
                        }

                        List<Suggestion> suggestions = raw.stream()
                                .filter(s -> s != null && s.get("phrase") != null)
                                .map(s -> new Suggestion((String) s.get("phrase")))
                                .toList();

                        logger.debug("DuckDuckGoSuggestions '{}': {} suggestions", query, suggestions.size());
                        return JsonParser.toJson(suggestions);

                    } catch (RestClientException e) {
                        logger.error("Error in DuckDuckGoSuggestions for '{}': {}", query, e.getMessage());
                        return JsonParser.toJson(Collections.emptyList());
                    }
                });
    }

    // =========================================================================
    // HTML parsing helpers — DDG Lite returns simple HTML tables
    // =========================================================================

    /**
     * Parses result links and snippets from DDG Lite's minimal HTML.
     * DDG Lite renders results as an HTML table — no JavaScript required.
     * Format: <a class="result-link"> for titles/URLs, followed by snippet text.
     */
    private List<WebResult> parseLiteHtml(String html, int limit) {
        var results = new java.util.ArrayList<WebResult>();

        // DDG Lite uses anchors with result URLs in href attributes
        // Pattern: <a class="result-link" href="URL">Title</a>
        //          followed by snippet in the next <td>
        var linkPattern = java.util.regex.Pattern.compile(
                "<a[^>]+class=[\"']result-link[\"'][^>]*href=[\"']([^\"']+)[\"'][^>]*>([^<]+)</a>" +
                        "(?:[\\s\\S]*?<td[^>]*>([^<]*(?:<[^/][^>]*>[^<]*</[^>]*>)*[^<]*)</td>)?",
                java.util.regex.Pattern.CASE_INSENSITIVE
        );

        // Fallback: also catch plain <a> tags pointing to external URLs
        var fallbackPattern = java.util.regex.Pattern.compile(
                "<a[^>]+href=[\"'](https?://[^\"']+)[\"'][^>]*>([^<]{10,200})</a>",
                java.util.regex.Pattern.CASE_INSENSITIVE
        );

        var matcher = linkPattern.matcher(html);
        while (matcher.find() && results.size() < limit) {
            String url     = cleanText(matcher.group(1));
            String title   = cleanText(matcher.group(2));
            String snippet = matcher.groupCount() >= 3 ? cleanText(matcher.group(3)) : "";

            if (StringUtils.hasText(url) && StringUtils.hasText(title)
                    && !url.contains("duckduckgo.com")) {
                results.add(new WebResult(title, url, snippet));
            }
        }

        // If primary pattern found nothing, try the fallback
        if (results.isEmpty()) {
            matcher = fallbackPattern.matcher(html);
            while (matcher.find() && results.size() < limit) {
                String url   = cleanText(matcher.group(1));
                String title = cleanText(matcher.group(2));
                if (StringUtils.hasText(url) && !url.contains("duckduckgo.com")) {
                    results.add(new WebResult(title, url, ""));
                }
            }
        }

        return results;
    }

    /** Strips HTML tags and trims whitespace from a string fragment. */
    private String cleanText(String raw) {
        if (!StringUtils.hasText(raw)) return "";
        return raw.replaceAll("<[^>]+>", " ")
                .replaceAll("&amp;",  "&")
                .replaceAll("&lt;",   "<")
                .replaceAll("&gt;",   ">")
                .replaceAll("&quot;", "\"")
                .replaceAll("&#39;",  "'")
                .replaceAll("&nbsp;", " ")
                .replaceAll("\\s+",   " ")
                .trim();
    }

    // =========================================================================
    // Result records
    // =========================================================================

    /** Full result from the Instant Answer API. */
    public record InstantAnswerResult(
            String abstractText,
            String abstractUrl,
            String abstractSource,
            String definition,
            String definitionUrl,
            String answer,
            String answerType,
            String type,          // "A" = article, "D" = disambiguation, "C" = category, etc.
            List<RelatedTopic> relatedTopics,
            List<InfoboxEntry> infoboxEntries
    ) {}

    /** A single related topic from the Instant Answer response. */
    public record RelatedTopic(String text, String firstUrl) {}

    /** A key-value fact from the infobox section. */
    public record InfoboxEntry(String label, String value) {}

    /** A web search result with title, URL, and snippet. */
    public record WebResult(String title, String url, String snippet) {}

    /** An autocomplete suggestion phrase. */
    public record Suggestion(String phrase) {}

    // =========================================================================
    // Builder
    // =========================================================================

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private int    maxResults     = 10;
        private String defaultRegion  = "wt-wt"; // worldwide, no region bias
        private ObservationRegistry observationRegistry = ObservationRegistry.NOOP;

        public Builder maxResults(int maxResults) {
            Assert.isTrue(maxResults > 0, "maxResults must be positive");
            this.maxResults = maxResults;
            return this;
        }

        /**
         * Default region/language code.
         * Examples: us-en, uk-en, br-pt, de-de, fr-fr, wt-wt (worldwide).
         */
        public Builder defaultRegion(String region) {
            Assert.hasText(region, "region must not be empty");
            this.defaultRegion = region;
            return this;
        }

        public Builder observationRegistry(ObservationRegistry observationRegistry) {
            Assert.notNull(observationRegistry, "observationRegistry must not be null");
            this.observationRegistry = observationRegistry;
            return this;
        }

        public DuckDuckGoSearchTool build() {
            return new DuckDuckGoSearchTool(this.maxResults, this.defaultRegion, this.observationRegistry);
        }
    }
}