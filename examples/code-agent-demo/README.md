# Code Agent Demo

An interactive AI coding assistant built with Spring AI and [spring-ai-agent-utils](../../spring-ai-agent-utils). Features file operations, shell execution, web access, MCP integration, and extensible skills.

## Overview

Command-line AI assistant with:

- **Code Operations**: Read, write, edit files; search with grep
- **Shell Execution**: Run commands synchronously or in background
- **Web Research**: Search and fetch web content with AI summarization
- **Task Management**: Track multi-step operations with todo lists
- **User Interaction**: Ask questions and collect answers during execution
- **Skills System**: Load custom capabilities from Markdown files
- **MCP Integration**: Connect to Model Context Protocol servers (AirBnB demo included)
- **Multi-Model Support**: Anthropic Claude, OpenAI GPT, or Google Gemini

## Tools

- **[AskUserQuestionTool](../../spring-ai-agent-utils/docs/AskUserQuestionTool.md)** - Interactive Q&A during execution
- **[SkillsTool](../../spring-ai-agent-utils/docs/SkillsTool.md)** - Load custom skills from Markdown files
- **[ShellTools](../../spring-ai-agent-utils/docs/ShellTools.md)** - Execute shell commands
- **[FileSystemTools](../../spring-ai-agent-utils/docs/FileSystemTools.md)** - File read/write/edit operations
- **[GrepTool](../../spring-ai-agent-utils/docs/GrepTool.md)** - Regex-based code search
- **[SmartWebFetchTool](../../spring-ai-agent-utils/docs/SmartWebFetchTool.md)** - AI-powered web content extraction
- **[BraveWebSearchTool](../../spring-ai-agent-utils/docs/BraveWebSearchTool.md)** - Web search
- **[TodoWriteTool](../../spring-ai-agent-utils/docs/TodoWriteTool.md)** - Task tracking
- **MCP Tools** - External tool integration via Model Context Protocol

## Prerequisites

- Java 17+
- Maven 3.6+
- API key for at least one AI provider:
  - `ANTHROPIC_API_KEY` (Claude - recommended)
  - `OPENAI_API_KEY` (GPT)
  - `GOOGLE_CLOUD_PROJECT` (Gemini)
- Optional: `BRAVE_API_KEY` for web search

## Quick Start

```bash
# 1. Set environment variables
export ANTHROPIC_API_KEY=your-key-here
export BRAVE_API_KEY=your-brave-key  # optional

# 2. Run
cd examples/code-agent-demo
mvn spring-boot:run
```

Example interaction:
```
> USER: Search for TODO comments in Java files
> ASSISTANT: [Uses GrepTool to search]

> USER: What tools do I have available?
> ASSISTANT: [Lists configured tools]
```

## Configuration

Key settings in [application.properties](src/main/resources/application-old.properties):

```properties
# Active provider (uncomment one in pom.xml)
spring.ai.anthropic.api-key=${ANTHROPIC_API_KEY}
spring.ai.anthropic.chat.options.model=claude-sonnet-4-5-20250929

# Skills location (classpath resource)
agent.skills.paths=classpath:/.claude/skills

# MCP servers
spring.ai.mcp.client.stdio.servers-configuration=classpath:/mcp-servers-config.json

# Model metadata for system prompt
agent.model=claude-sonnet-4-5-20250929
agent.model.knowledge.cutoff=2025-01-01
```

### Switching Models

1. Edit [pom.xml](pom.xml) - comment/uncomment desired provider
2. Update [application.properties](src/main/resources/application-old.properties) with matching config
3. Rebuild: `mvn clean install`

### Custom Skills

Add Markdown files to [src/main/resources/.claude/skills/](src/main/resources/.claude/skills/):

```markdown
---
name: my-skill
description: When to use this skill
allowed-tools: Read, Bash
---
Skill prompt instructions here...
```

See [SkillsTool docs](../../spring-ai-agent-utils/docs/SkillsTool.md) for details.

## Architecture

### ChatClient Setup

[Application.java:51-94](src/main/java/org/springaicommunity/agent/Application.java#L51-L94) configures the ChatClient:

```java
ChatClient chatClient = chatClientBuilder
    .defaultSystem(systemPrompt)                        // MAIN_AGENT_SYSTEM_PROMPT_V2.md
    .defaultToolCallbacks(mcpToolCallbackProvider)      // MCP tools
    .defaultToolCallbacks(skillsTool)                   // Skills
    .defaultTools(AskUserQuestionTool, TodoWriteTool,   // Agent tools
                  ShellTools, FileSystemTools,
                  SmartWebFetchTool, BraveWebSearchTool, GrepTool)
    .defaultAdvisors(ToolCallAdvisor,                   // Tool execution
                     MessageChatMemoryAdvisor)          // 500-message memory
    .build();
```

### Key Features

- **System Prompt**: [MAIN_AGENT_SYSTEM_PROMPT_V2.md](../../spring-ai-agent-utils/src/main/resources/prompt/MAIN_AGENT_SYSTEM_PROMPT_V2.md) - comprehensive agent behavior config
- **MCP Integration**: Auto-connects to configured MCP servers (AirBnB example included)
- **User Interaction**: `AskUserQuestionTool` enables multi-choice questions during execution
- **Logging**: Optional `MyLoggingAdvisor` for debugging (currently commented out)

## Usage Examples

**Code Search**
```
> USER: Find all classes extending SpringBootApplication
[Uses GrepTool with pattern]
```

**Interactive Questions**
```
> USER: Help me choose a testing framework
> ASSISTANT: [Uses AskUserQuestionTool to present options: JUnit, TestNG, etc.]
```

**Multi-Step Tasks**
```
> USER: Create a DateUtil class, write tests, and run them
[Uses TodoWriteTool to plan → FileSystemTools to write → ShellTools to test]
```

**Web Research**
```
> USER: Find latest Spring AI advisor patterns
[Uses BraveWebSearchTool + SmartWebFetchTool]
```

## Project Structure

```
code-agent-demo/
├── src/main/java/.../agent/
│   ├── Application.java           # ChatClient setup and main loop
│   └── MyLoggingAdvisor.java      # Optional debug advisor
├── src/main/resources/
│   ├── .claude/skills/            # Custom skills directory
│   ├── application.properties     # Configuration
│   └── mcp-servers-config.json    # MCP server definitions
└── pom.xml                        # Dependencies
```

## How It Works

1. Application starts, builds ChatClient with tools and advisors
2. User enters prompts in console loop
3. Agent processes request using system prompt guidance
4. Agent selects and invokes appropriate tools
5. Results synthesized into response
6. Conversation memory maintains context (500 messages)

## Customization

**Adjust memory**:
```java
MessageWindowChatMemory.builder().maxMessages(1000).build()
```

**Tool configuration**:
```java
BraveWebSearchTool.builder(apiKey).resultCount(20).build()
SmartWebFetchTool.builder(client).maxContentLength(150_000).build()
```

**Enable debug logging**:
Uncomment `MyLoggingAdvisor` in [Application.java:90-93](src/main/java/org/springaicommunity/agent/Application.java#L90-L93)

## Resources

- [spring-ai-agent-utils Documentation](../../spring-ai-agent-utils/README.md)
- [Tool Documentation](../../spring-ai-agent-utils/docs/)
- [Spring AI Docs](https://docs.spring.io/spring-ai/reference/)

## License

Apache License 2.0
