# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2020-06-18

### Added

- Wrapper for training a gensim word2vec model [#13](https://github.com/iomega/spec2vec/tree/13-gensim-wrapper)
- Basic logger for word2vec model training [#11](https://github.com/iomega/spec2vec/issues/11)

### Changed

- Extend spec2vec similarity calculation to handle missing words [#9](https://github.com/iomega/spec2vec/issues/9)
- Extend documentation and given code examples [#15](https://github.com/iomega/spec2vec/issues/15)
- Updated the integration test to work with matchms 0.4.0 [#7](https://github.com/iomega/spec2vec/issues/7)

## [0.1.0] - 2020-06-02

### Added

- Matchms as dependency [#4](https://github.com/iomega/spec2vec/pull/4)
- Bump2version config

### Changed

- Splitted spec2vec from [matchms]. See (https://github.com/matchms/matchms) [#1](https://github.com/iomega/spec2vec/pull/1) [#4](https://github.com/iomega/spec2vec/pull/4)
  - Updated packaging related configuration
  - Update the GH Actions workflows
  - Updated the documentation
  - Updated the badges
  - Updated the integration and unit tests
  - Zenodo metadata
  
### Fixed

### Removed

- Fossa configuration
- Flowchart

[Unreleased]: https://github.com/iomega/spec2vec/compare/0.2.0...HEAD
[0.2.0]: https://github.com/iomega/spec2vec/releases/tag/0.2.0
[0.1.0]: https://github.com/iomega/spec2vec/releases/tag/0.1.0
