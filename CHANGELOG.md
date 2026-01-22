# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-23

### Added
- **Equilibrium Propagation Framework**: Core implementation of EqProp with spectral normalization.
- **Bioplausible UI**: Comprehensive dashboard with 10 tabs (Home, Train, Compare, Search, Results, Benchmarks, Deploy, Community, Console, Settings).
- **Bioplausible Lab**: Advanced analysis tools for researching model dynamics (DeepDream, Microscope, CubeViz).
- **Validation Suite**: 51 verification tracks to ensure scientific rigor.
- **Triton Kernels**: Optimized kernels for accelerated EqProp dynamics (optional dependency).
- **P2P Training**: Decentralized training capabilities via Kademlia DHT.
- **Experiment Tracking**: Integration with ResultsManager for saving and exporting runs.
- **Documentation**: Initial README, development roadmap (TODO.md), and contribution guidelines.

### Changed
- Standardized project structure under `bioplausible/` and `bioplausible_ui/`.
- Updated dependencies to include `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `wandb`.
- Improved test suite stability and coverage.

### Removed
- Deprecated legacy UI components (`bioplausible_ui_old`).
