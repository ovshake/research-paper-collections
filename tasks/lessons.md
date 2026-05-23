# Lessons

## Subagent model selection
**Date:** 2026-05-02
**Trigger:** Launched six web-search agents without `model: "opus"`. User corrected: "relaunch using opus 4.7 and always use opus 4.7 for the subagents."

**Rule:** Every `Agent` tool call must include `model: "opus"`. The global `~/.claude/CLAUDE.md` already states "Always use Opus as subagents or local agents. Do not use any other model, ever." — this is non-negotiable, not advisory.

**Failure pattern to avoid:** Reaching for `subagent_type` (web-search-agent, general-purpose, etc.) without overriding the model. Default subagent definitions can specify other models in their frontmatter; the override is exactly the case for that.

**Self-check before any Agent call:** Does the call have `model: "opus"`? If not, add it.
