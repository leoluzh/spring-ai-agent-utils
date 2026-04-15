/*
 * Copyright 2025 - 2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springaicommunity.agent.tools.extended;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import org.springframework.ai.util.json.JsonParser;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for {@link ParallelWebTool}.
 *
 * @author Spring AI Community
 */
class ParallelWebToolTests {

	@Nested
	@DisplayName("Builder Tests")
	class BuilderTests {

		@Test
		@DisplayName("Should create tool with valid API key")
		void shouldCreateToolWithValidApiKey() {
			ParallelWebTool tool = ParallelWebTool.builder("test-api-key").build();
			assertThat(tool).isNotNull();
		}

		@Test
		@DisplayName("Should throw exception when API key is null")
		void shouldThrowExceptionWhenApiKeyIsNull() {
			assertThatThrownBy(() -> ParallelWebTool.builder(null))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("API key must not be null or empty");
		}

		@Test
		@DisplayName("Should set custom max results")
		void shouldSetCustomMaxResults() {
			ParallelWebTool tool = ParallelWebTool.builder("test-api-key")
				.maxResults(15)
				.build();
			assertThat(tool).isNotNull();
		}

		@Test
		@DisplayName("Should throw exception for zero max results")
		void shouldThrowExceptionForZeroMaxResults() {
			assertThatThrownBy(() -> ParallelWebTool.builder("test-api-key").maxResults(0))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("maxResults must be between 1 and 20");
		}

		@Test
		@DisplayName("Should throw exception for max results over 20")
		void shouldThrowExceptionForMaxResultsOver20() {
			assertThatThrownBy(() -> ParallelWebTool.builder("test-api-key").maxResults(21))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("maxResults must be between 1 and 20");
		}

		@Test
		@DisplayName("Should set custom search mode")
		void shouldSetCustomSearchMode() {
			ParallelWebTool tool = ParallelWebTool.builder("test-api-key")
				.searchMode(ParallelWebTool.SearchMode.ONE_SHOT)
				.build();
			assertThat(tool).isNotNull();
		}

		@Test
		@DisplayName("Should throw exception for null search mode")
		void shouldThrowExceptionForNullSearchMode() {
			assertThatThrownBy(() -> ParallelWebTool.builder("test-api-key").searchMode(null))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("searchMode must not be null");
		}
	}

	@Nested
	@DisplayName("Search Query Tests")
	class SearchQueryTests {
		private final ParallelWebTool tool = ParallelWebTool.builder("test-api-key").build();

		@Test
		@DisplayName("Should return empty list for null objective")
		void shouldReturnEmptyListForNullObjective() {
			String result = tool.search(null, null, null, null, null, null, null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyList()));
		}
	}

	@Nested
	@DisplayName("Extract Query Tests")
	class ExtractQueryTests {
		private final ParallelWebTool tool = ParallelWebTool.builder("test-api-key").build();

		@Test
		@DisplayName("Should return empty list for null URLs")
		void shouldReturnEmptyListForNullUrls() {
			String result = tool.extract(null, null, null, null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyList()));
		}
	}

	@Nested
	@DisplayName("Task Query Tests")
	class TaskQueryTests {
		private final ParallelWebTool tool = ParallelWebTool.builder("test-api-key").build();

		@Test
		@DisplayName("Should return empty map for null input")
		void shouldReturnEmptyMapForNullInput() {
			String result = tool.task(null, null, null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyMap()));
		}
	}

	@Nested
	@DisplayName("FindAll Query Tests")
	class FindAllQueryTests {
		private final ParallelWebTool tool = ParallelWebTool.builder("test-api-key").build();

		@Test
		@DisplayName("Should return empty list for null objective")
		void shouldReturnEmptyListForNullObjective() {
			String result = tool.findAll(null, "type", List.of(Map.of("name", "cond", "description", "desc")), null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyList()));
		}

		@Test
		@DisplayName("Should return empty list for null entity type")
		void shouldReturnEmptyListForNullEntityType() {
			String result = tool.findAll("obj", null, List.of(Map.of("name", "cond", "description", "desc")), null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyList()));
		}

		@Test
		@DisplayName("Should return empty list for null conditions")
		void shouldReturnEmptyListForNullConditions() {
			String result = tool.findAll("obj", "type", null, null, null);
			assertThat(result).isEqualTo(JsonParser.toJson(Collections.emptyList()));
		}
	}

	@Nested
	@DisplayName("Result Record Tests")
	class ResultRecordTests {

		@Test
		@DisplayName("Should create SearchResult")
		void shouldCreateSearchResult() {
			var result = new ParallelWebTool.SearchResult("Title", "url", "date", List.of("excerpt"));
			assertThat(result.title()).isEqualTo("Title");
			assertThat(result.excerpts()).containsExactly("excerpt");
		}

		@Test
		@DisplayName("Should create ExtractResult")
		void shouldCreateExtractResult() {
			var result = new ParallelWebTool.ExtractResult("url", "title", "date", List.of("excerpt"), "full");
			assertThat(result.url()).isEqualTo("url");
			assertThat(result.fullContent()).isEqualTo("full");
		}

		@Test
		@DisplayName("Should create TaskResult")
		void shouldCreateTaskResult() {
			var citation = new ParallelWebTool.Citation("Title", "url", "high");
			var result = new ParallelWebTool.TaskResult("output", List.of(citation), "interactionId", "runId");
			assertThat(result.output()).isEqualTo("output");
			assertThat(result.runId()).isEqualTo("runId");
			assertThat(result.citations()).containsExactly(citation);
		}

		@Test
		@DisplayName("Should create FindAllMatch")
		void shouldCreateFindAllMatch() {
			var result = new ParallelWebTool.FindAllMatch("Name", "desc", Map.of("field", "value"), List.of("source"));
			assertThat(result.name()).isEqualTo("Name");
			assertThat(result.fields()).containsEntry("field", "value");
			assertThat(result.sources()).containsExactly("source");
		}
	}
}