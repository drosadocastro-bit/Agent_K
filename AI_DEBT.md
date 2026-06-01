# Agent K AI Debt Register

Last audit: 2026-05-31

This file tracks known integrity, safety, maintenance, and auditability debt in
Agent K. Each item records what was found, why it matters, and the intended fix.

## Audit Baseline

- Command: `python -m pytest --tb=short`
- Result: `63 passed`
- Editor diagnostics: none found
- Runtime dependencies: none
- Dev dependencies: `pytest>=9.0.0`
- Dynamic execution scan: no `eval`, `exec`, `subprocess`, `os.system`, or
  `shell=True` use found in source

## Open Items

### AK-D1: Duplicate HTTP Runner Plumbing

- Status: open
- Severity: low
- Area: maintainability
- Found in:
  - `src/agent_k/agents/ollama.py`
  - `src/agent_k/agents/manatuabon.py`
- What was found: both live runners hand-roll JSON HTTP request/response,
  timeout, decode, and error handling logic.
- Why it matters: duplicated transport code increases the chance that one
  runner gets timeout, JSON, or HTTP-error fixes while the other drifts.
- Fix: extract a tiny standard-library HTTP JSON helper used by both runners.
  Keep it opt-in and local-first; no runtime dependency should be added.

### AK-D2: Configurable Remote Hosts Are Not Explicitly Guarded

- Status: open
- Severity: medium
- Area: security / offline-first posture
- Found in:
  - `src/agent_k/cli.py`
  - `src/agent_k/agents/ollama.py`
  - `src/agent_k/agents/manatuabon.py`
- What was found: the live CLI defaults to localhost, but `--host` accepts any
  URL. This is user-initiated, not a server-side SSRF vector, but it weakens
  the documented air-gapped/local-first posture.
- Why it matters: an accidental remote host could send scenario prompts,
  context, or captured outputs outside the local machine.
- Fix: default to localhost-only validation. If remote hosts are needed, require
  an explicit flag such as `--allow-remote-host` and record that choice in trace
  metadata.

### AK-D4: Memory-ID Citations Are Not Grounded

- Status: open
- Severity: medium
- Area: RAG grounding
- Found in:
  - `src/agent_k/scoring/citations.py`
  - Manatuabon bridge outputs that cite `[Memory #ID]`
- What was found: citation grounding detects quoted spans and URLs, but not
  bracketed memory references such as `[Memory #7]`.
- Why it matters: RAG systems commonly cite internal memory/document IDs. Agent
  K currently records `citation_grounding` as skipped for those references.
- Fix: add a conditional `memory_reference_grounding` dimension or extend
  citation grounding with explicit source-id metadata in the trace.

### AK-D5: OpenClaw Consistency Is a Neutral Pass

- Status: open
- Severity: low
- Area: scoring semantics
- Found in: `src/agent_k/openclaw_evaluator.py`
- What was found: OpenClaw traces do not carry structured assertions, so
  consistency returns a neutral `1.0` with a detail explaining it is not
  autonomously evaluated.
- Why it matters: the score can look stronger than the available evidence when
  a trace lacks structured assertions.
- Fix: consider making consistency conditional/skipped for OpenClaw traces, or
  introduce structured output assertions into the trace schema.

### AK-D6: Optional Bridge Runner Couples Agent K To Manatuabon

- Status: open
- Severity: low
- Area: architecture
- Found in:
  - `src/agent_k/agents/manatuabon.py`
  - `tests/test_manatuabon_runner.py`
- What was found: Agent K now has an optional Manatuabon-specific bridge runner.
- Why it matters: useful for local testing, but long-term integrations should
  prefer generic trace ingestion so Agent K does not grow one runner per app.
- Fix: keep the runner optional, then prioritize `eval-trace` so Manatuabon can
  integrate by emitting OpenClaw traces instead of requiring custom Agent K code.

### AK-D7: Generated Reports And Temporary Artifacts Need Hygiene Review

- Status: open
- Severity: low
- Area: repository hygiene
- Found in:
  - `reports/`
  - `.tmp/`
- What was found: the repository contains generated evaluation reports and a
  temporary OpenClaw artifact directory.
- Why it matters: generated artifacts can make diffs noisy and may contain
  sensitive prompts or model outputs in future runs.
- Fix: decide which demo reports are intentionally versioned, ignore ephemeral
  outputs, and document report-retention expectations.

## Closed Items From This Audit

### AK-C1: Stale README Test Count

- Status: fixed
- Severity: low
- Area: documentation
- What was found: README still said `58 tests` after the Manatuabon bridge
  runner tests raised the suite to `61 tests`.
- Why it mattered: stale test counts make audit status ambiguous.
- Fix applied: updated README test count and added the Manatuabon runner test
  row.

### AK-C2: Missing Generic `eval-trace` CLI

- Status: fixed
- Severity: medium
- Area: usability / integration boundary
- What was found: Agent K could score traces through the Python API, but there
  was no first-class command like `python -m agent_k eval-trace trace.json`.
- Why it mattered: every external system needed a custom runner or adapter even
  when it could already emit an OpenClaw trace.
- Fix applied: added `eval-trace`, default adjacent JSON/Markdown outputs,
  clean malformed-trace errors, README docs, and CLI regression tests.
