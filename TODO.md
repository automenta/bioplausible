# Bioplausible UI Redesign - Complete Development Plan

> **Single Source of Truth**: This document contains the complete, verified plan for redesigning `bioplausible_ui/` into a production-ready, extensible dual-app architecture.

---

## Executive Summary

**Goal**: Replace ~8000 LOC of duplicated UI code with ~1750 LOC of schema-driven, metaprogramming-based architecture while maintaining 100% feature coverage and enabling rapid scientific iteration.

**Approach**: 
- Dual apps: `biopl` (production) + `biopl-lab` (research)
- Schema-driven UI generation via metaclasses
- Registry pattern for automatic tool discovery
- Event-driven backend pipeline

**Outcome**: 78% code reduction, 100% feature coverage, extensible in 1-3 steps

---

## Implementation Checklist

### ✅ Phase 1: Backend Pipeline
- [x] `bioplausible/pipeline/session.py` - TrainingSession, SessionState
- [x] `bioplausible/pipeline/config.py` - TrainingConfig
- [x] `bioplausible/pipeline/events.py` - Event types
- [x] `bioplausible/models/registry.py` - Add family, capabilities
- [x] Backend integration tests

### ✅ Phase 2: Core Abstractions
- [x] `core/schema.py` - WidgetDef, ActionDef, LayoutDef, TabSchema
- [x] `core/base.py` - TabMeta, BaseTab, base widgets
- [x] `core/bridge.py` - SessionBridge, TrainingWorker
- [x] `core/widgets/` - 8 base component implementations
- [x] Component tests (pytest-qt)

### ✅ Phase 3: Main App (biopl)
- [x] `app/main.py` - Entry point
- [x] `app/window.py` - AppMainWindow
- [x] `app/schemas/` - 8 tab schemas
- [x] `app/tabs/` - 8 tab implementations
- [x] UI integration tests

### ✅ Phase 4: Lab App (biopl-lab)
- [x] `lab/main.py` - Entry point + CLI
- [x] `lab/window.py` - LabMainWindow with auto-discovery
- [x] `lab/registry.py` - ToolRegistry
- [x] `lab/tools/base.py` - BaseTool
- [x] `lab/tools/` - 7 tool implementations
- [x] Tool tests

### ✅ Phase 5: Migration
- [x] Update `pyproject.toml` entry points
- [x] Archive old code to `bioplausible_ui_old/`
- [x] Update README with new commands
- [x] Full regression test (all 51 validation tracks)
- [x] Documentation update

---

**Status**: ✅ All Phases Complete.
