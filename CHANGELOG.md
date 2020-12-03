# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Add optional progress bar for spec2vec.matrix() calculations (default is False) [#43](https://github.com/iomega/spec2vec/pull/43)

## [0.3.1] - 2020-09-23

### Changed

- Implement faster, numba-based cosine similarity function [#29](https://github.com/iomega/spec2vec/pull/29)

## [0.3.0] - 2020-09-16

### Added

- Support for Python 3.8 [#35](https://github.com/iomega/spec2vec/pull/35)

### Changed

- Refactored Spec2Vec class to provide .pair() and .matrix() methods [#35](https://github.com/iomega/spec2vec/pull/35)

### Removed

- Spec2VecParallel (is now included as Spec2Vec.matrix()) [#35](https://github.com/iomega/spec2vec/pull/35)

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

[Unreleased]: https://github.com/iomega/spec2vec/compare/0.3.1...HEAD
[0.3.1]: https://github.com/iomega/spec2vec/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/iomega/spec2vec/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/iomega/spec2vec/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/iomega/spec2vec/releases/tag/0.1.0
