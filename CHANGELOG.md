# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2022-01-03

## Added

- Logging (replacing former print statements) including options to write logs to file [#73](https://github.com/iomega/spec2vec/pull/73)
- Now supports Python 3.9 (including CI test runs) [#40](https://github.com/iomega/spec2vec/issues/40)

## Changed

- missing words percentage above the `allowed_missing_percentage` no longer causes an expection but only leads to raising a warning [#73](https://github.com/iomega/spec2vec/pull/73)
- default setting for `allowed_missing_percentage` to 10.0 to be less strict on model coverage [#72](https://github.com/iomega/spec2vec/pull/72)

## [0.5.0] - 2021-06-18

## Changed

- Spec2Vec is now using gensim >= 4.0.0 [#62](https://github.com/iomega/spec2vec/pull/62)

## [0.4.0] - 2021-02-10

## Changed

- refactored `Spec2Vec` to now accept `Spectrum` or `SpectrumDocument` as input [#51](https://github.com/iomega/spec2vec/issues/51)

## Fixed

- updated and fixed code examples  [#51](https://github.com/iomega/spec2vec/issues/51)
- updated and fixed attribute typing [#51](https://github.com/iomega/spec2vec/issues/51)

## [0.3.4] - 2021-02-10

### Changed

- update required numba version to >=0.51 to avoid issues between numba and numpy [#55](https://github.com/iomega/spec2vec/pull/55)

## [0.3.3] - 2021-02-09

### Added

- Metadata getter method for `SpectrumDocument` [#50](https://github.com/iomega/spec2vec/pull/50)
- Implement `is_symmetric=True` option for `Spec2Vec.matrix` method [#53](https://github.com/iomega/spec2vec/pull/53)

### Changed

- Change default for `n_decimals` parameter from 1 to 2 [#50](https://github.com/iomega/spec2vec/pull/50)

## [0.3.2] - 2020-12-03

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

[Unreleased]: https://github.com/iomega/spec2vec/compare/0.6.0...HEAD
[0.6.0]: https://github.com/iomega/spec2vec/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/iomega/spec2vec/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/iomega/spec2vec/compare/0.3.4...0.4.0
[0.3.4]: https://github.com/iomega/spec2vec/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/iomega/spec2vec/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/iomega/spec2vec/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/iomega/spec2vec/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/iomega/spec2vec/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/iomega/spec2vec/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/iomega/spec2vec/releases/tag/0.1.0
