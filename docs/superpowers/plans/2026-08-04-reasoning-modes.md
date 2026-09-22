# V12.3 Reasoning Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add configurable Nemotron reasoning modes while preserving structured JSON reliability and handling transient NVIDIA saturation.

**Architecture:** The CLI selects a named reasoning profile and passes it to `NvidiaChatClient`. The client translates that profile into NVIDIA `chat_template_kwargs`, `reasoning_budget`, and an adjusted token ceiling. Response normalization remains deterministic in Python, including tolerant catalogue page parsing.

**Tech Stack:** Python 3.11+, argparse, requests, pytest, NVIDIA OpenAI-compatible chat completions.

## Global Constraints

- Default profile: `normal`.
- Profiles: `rapide`, `normal`, `approfondi`.
- Normal budget: 2048 reasoning tokens.
- Deep budget: 4096 reasoning tokens.
- Preserve V12.2 behavior and output compatibility.

---

### Task 1: Reasoning profile contract
- [x] Write failing client and CLI tests.
- [x] Implement profile validation and payload generation.
- [x] Run focused tests.

### Task 2: Robust response and transient retry
- [x] Write failing tests for scalar/page-label `catalogue_pages` and ResourceExhausted retry.
- [x] Implement tolerant page normalization and bounded exponential backoff.
- [x] Run focused tests.

### Task 3: Documentation and release
- [x] Update README, changelog, version, architecture, and verification.
- [x] Run full tests and compilation.
- [x] Build and verify ZIP archive.
