# IRONFORGE Agents Overview

Version: 1.1.0  
Last Updated: 2025-08-30

## Purpose

This page summarizes the multi-agent ecosystem supporting IRONFORGE workflows and documentation quality gates. It provides links to agent-specific READMEs and planning prompts.

## Core Principles

- Golden invariants preserved: 6 events, 4 edge intents, 45D/51D nodes, 20D edges, HTF last‑closed, session isolation
- Public surfaces documented: CLI and `ironforge.api`
- Performance guardrails: target <3s per session for discovery context; keep examples runnable

## Key Agents and Entrypoints

- Pipeline Orchestrator: agents/pipeline_orchestrator/README.md
- Pattern Intelligence Analyst: agents/pattern_intelligence_analyst/README.md
- Session Boundary Guardian: agents/session_boundary_guardian/README.md
- Minidash Enhancer: agents/minidash_enhancer/README.md
- TGAT Attention Analyzer: agents/tgat_attention_analyzer/README.md
- Contract Compliance Enforcer: agents/contract_compliance_enforcer/README.md
- Pipeline Performance Monitor: agents/pipeline_performance_monitor/README.md

## Cross‑Links

- Root index: docs/README.md
- API Reference: docs/03-API-REFERENCE.md
- Architecture: docs/04-ARCHITECTURE.md
- Operations & Flows: docs/specialized/operations.md, docs/specialized/flows.md

## Validation

- Ensure agent docs reference `ironforge.api` imports for examples
- Link to CLI help where applicable (discover‑temporal, score‑session, validate‑run, report‑minimal)
- Maintain archive preservation: historical plans remain in agents/*/planning with labels