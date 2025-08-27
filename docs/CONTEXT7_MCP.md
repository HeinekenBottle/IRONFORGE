# Context7 MCP Integration and Documentation Standards

IRONFORGE is optimized for Context7 MCP (Model Context Protocol) integration, providing up-to-date, version-specific documentation access directly in your IDE or AI agent. This guide covers setup, usage patterns, and the documentation standards we follow for MCP compatibility.

## Quick Setup

### Context7 MCP Server (Recommended)

Context7 MCP by Upstash provides two key tools:
- `resolve-library-id`: Converts library names to Context7-compatible IDs
- `get-library-docs`: Retrieves focused documentation with topic filtering

### Multi-Client Configuration

#### Cursor
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

#### Claude Desktop
```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

#### VS Code
```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

#### JetBrains AI Assistant
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

#### Warp
```json
{
  "Context7": {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp"],
    "env": {},
    "working_directory": null,
    "start_on_launch": true
  }
}
```

### Testing Setup

```bash
# Test with MCP Inspector
npx -y @modelcontextprotocol/inspector npx @upstash/context7-mcp@latest
```

## Topic Filtering with Context7 MCP

Context7 MCP supports topic-focused documentation retrieval. Use these topics with IRONFORGE:

### Available Topics
- `engines` - Core discovery, scoring, validation functions
- `cli` - Command-line interface and automation
- `sdk` - Configuration and I/O utilities
- `reporting` - Dashboard and visualization
- `validation` - Quality gates and validation rails
- `integration` - Container system and lazy loading
- `data_engine` - Data processing and graph building
- `motifs` - Pattern analysis and discovery

### Usage in Prompts

```text
Create a discovery pipeline using IRONFORGE engines. use context7 topic:engines

Set up IRONFORGE configuration for production deployment. use context7 topic:sdk

Generate a minidash report with custom motifs. use context7 topic:reporting
```

## IRONFORGE Public API Surface

**Centralized imports (MCP-optimized):**

```python
from ironforge.api import (
    # Engines
    run_discovery, score_confluence, validate_run, build_minidash,
    # SDK
    Config, load_config, materialize_run_dir,
    # Integration
    get_ironforge_container, initialize_ironforge_lazy_loading
)
```

**Engine facade (alternative):**

```python
from ironforge.engines import COMMAND_MAP, run_discovery, score_confluence
```

Legacy deep imports remain available but are not recommended for new code.

## Documentation Conventions (Context7-Friendly)

- Centralized imports: prefer `ironforge.api` to deep module paths
- Short, runnable examples with explicit imports
- Parameter and return types annotated where practical
- Keep examples under 25 lines; link to full guides for details
- Stable section anchors and headings: Engines, Reporting, Validation, SDK

## Example: Minimal End-to-End Run

```python
from ironforge.api import (
    run_discovery, score_confluence, validate_run, build_minidash,
    load_config, materialize_run_dir
)
from pathlib import Path
import pandas as pd

cfg = load_config("configs/dev.yml")
run_dir = materialize_run_dir(cfg)

# 1) Discovery
patterns = run_discovery(["data/shards/NQ_M5/shard_0001"], str(run_dir / "patterns"), cfg)

# 2) Scoring
scores_path = score_confluence(patterns, str(run_dir / "confluence"), None, 65.0)

# 3) Reporting
activity = pd.DataFrame()
confluence = pd.read_parquet(scores_path)
out_html, out_png = build_minidash(activity, confluence, [], run_dir / "minidash.html", run_dir / "minidash.png")
print(out_html, out_png)

# 4) Validation
results = validate_run(str(run_dir))
print(results.get("status"))
```

## Authoring Guidelines for New Docs

- Place user-facing docs in `docs/` with clear titles and summaries
- Update `docs/API_REFERENCE.md` and `README.md` when adding a new public function
- Keep `docs/context.json` in sync with CLI and entrypoint contracts
- Include importable code paths in backticks for easy MCP fetching

## Troubleshooting

- If a deep import is required, ensure the module has a docstring and typed signatures
- Lint/type-check with `ruff`, `black`, `mypy` before publishing docs
- For broken imports, fall back to legacy paths listed in API_REFERENCE.md and open an issue to re-export via `ironforge.api`

