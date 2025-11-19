# Git Hook

introduce git hook

<!--more-->

[TOC]

## pre-commit

The Comprehensive Guide to `pre-commit`

> generally you want to use [pre-commit.ci](https://pre-commit.ci/) which is faster and has more features.

### What is `pre-commit`

[`pre-commit`](https://pre-commit.com/) is an open-source framework for managing and maintaining multi-language pre-commit hooks. In software development, **Git hooks** are scripts that run automatically on specific Git events (e.g., commit, push). `pre-commit` makes it easy to install and run hooks for code quality, security, style, and more—before code gets committed.

It is language-agnostic and supports everything from Python, JavaScript, Go, and Shell scripts, to Rust, Ruby, and more.

### Why use

- **Automate Code Quality**: Catch formatting errors, linting issues, and security flaws *before* they reach your repo.
- **Consistency Across Teams**: Ensures everyone runs the same checks, reducing “works on my machine” problems.
- **Developer Productivity**: Automates repetitive checks, freeing up dev time.
- **Multi-language Support**: Works across codebases with multiple languages and ecosystems.
- **Easy CI/CD Integration**: Hooks can be run manually or in CI to ensure consistency.

### How to use

1. insall pre-commit ：

   - python user

      ```
      pip install pre-commit
      ```

   - linux/macOS

     ```
     sudo/brew apt install pre-commit
     ```

   **创建或编辑 `.pre-commit-config.yaml`** ： 将Example配置粘贴到 `.pre-commit-config.yaml` 文件中。

2. install hooks ：

   ```
   pre-commit install
   ```

3. run hooks ：

   ```
   pre-commit run --all-files
   ```

4. update hooks ：

   ```
   pre-commit autoupdate
   ```

5. add new hook

   ```
   pre-commit sample-config > .pre-commit-config.yaml
   ```

6. add github actionhttps://github.com/pre-commit/action

通过这些步骤，你可以确保在提交代码时，各种语言的代码都会经过相应的检查和格式化，从而提高代码质量。





### Example

```yaml
repos:
- repo: https://github.com/psf/black.git
  rev: 22.3.0
  hooks:
    - id: black
      language_version: python3

- repo: https://github.com/PyCQA/flake8.git
  rev: 4.0.1
  hooks:
    - id: flake8

- repo: https://github.com/pre-commit/mirrors-mypy.git
  rev: v0.971
  hooks:
    - id: mypy

- repo: https://github.com/pre-commit/mirrors-cppcheck.git
  rev: v2.5
  hooks:
    - id: cppcheck
      args: [--enable=all, --inconclusive, --force]

- repo: https://github.com/pre-commit/mirrors-clang-format.git
  rev: v12.0.1
  hooks:
    - id: clang-format
      args: [-i]

- repo: https://github.com/pre-commit/mirrors-yamllint.git
  rev: v1.26.3
  hooks:
    - id: yamllint

- repo: https://github.com/pre-commit/mirrors-xmlstarlet.git
  rev: v1.6.1
  hooks:
    - id: xmlstarlet
      args: [val, -e]

- repo: https://github.com/pre-commit/mirrors-pylint.git
  rev: v2.11.1
  hooks:
    - id: pylint

- repo: https://github.com/pre-commit/mirrors-pytest.git
  rev: v6.2.4
  hooks:
    - id: pytest

- repo: https://github.com/pre-commit/mirrors-shellcheck.git
  rev: v0.8.0
  hooks:
    - id: shellcheck

- repo: https://github.com/pre-commit/mirrors-dockerfile-linter.git
  rev: v1.0.0
  hooks:
    - id: dockerfile-linter

- repo: https://github.com/pre-commit/mirrors-hadolint.git
  rev: v2.8.0
  hooks:
    - id: hadolint

- repo: https://github.com/pre-commit/mirrors-checkstyle.git
  rev: v8.42.1
  hooks:
    - id: checkstyle
      args: [-c, .checkstyle/checkstyle.xml]

- repo: https://github.com/pre-commit/mirrors-google-java-format.git
  rev: v1.10.0
  hooks:
    - id: google-java-format
      args: [-a]

- repo: https://github.com/pre-commit/mirrors-golangci-lint.git
  rev: v1.45.2
  hooks:
    - id: golangci-lint
      args: [run, --fix]

- repo: https://github.com/pre-commit/mirrors-rustfmt.git
  rev: v1.4.32
  hooks:
    - id: rustfmt

- repo: https://github.com/pre-commit/mirrors-swagger-cli.git
  rev: v2.1.1
  hooks:
    - id: swagger-cli
      args: [validate]
```

## Integrating with CI/CD

- **Why**: Ensures code quality checks run in CI, not just on developer machines.

- **How**:

  - In your CI job, install `pre-commit` and run:

    ```
    pre-commit run --all-files --show-diff-on-failure
    ```

  - For some CIs (e.g., GitHub Actions), there are pre-built actions: [pre-commit/action](https://github.com/pre-commit/action)

- **Fail the build** if hooks fail.







# References

- [The Comprehensive Guide to `pre-commit`](https://gist.github.com/MangaD/6a85ee73dd19c833270524269159ed6e)
- https://git-scm.com/docs/githooks
- https://github.com/pre-commit/action
- https://pre-commit.ci/
