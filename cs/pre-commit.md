# Main Takeaway

pre-commit

## 使用步骤

1. **安装 pre-commit** ：

   ```bash
   pip install pre-commit
   ```

   **创建或编辑 `.pre-commit-config.yaml`** ： 将上述配置粘贴到 `.pre-commit-config.yaml` 文件中。

2. **安装 hooks** ：

   ```
   pre-commit install
   ```

3. **运行 hooks** ：

   ```
   pre-commit run --all-files
   ```

4. **更新 hooks** ：

   ```
   pre-commit autoupdate
   ```

通过这些步骤，你可以确保在提交代码时，各种语言的代码都会经过相应的检查和格式化，从而提高代码质量。





# Example

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

