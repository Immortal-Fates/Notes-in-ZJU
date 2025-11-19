# Main Takeaway

介绍CI/CD

<!--more-->

# CI/CD Intro

## 什么是CI/CD

CI/CD是现代软件开发中的核心实践，包含以下三个概念：

### CI (Continuous Integration) - 持续集成

- **定义**：开发人员频繁地将代码变更集成到主分支，每次集成都通过自动化构建和测试来验证
- **核心原则**：
  - 频繁提交代码（至少每天一次）
  - 自动化构建和测试
  - 快速反馈机制
  - 保持主分支随时可发布

### CD (Continuous Delivery/Deployment) - 持续交付/部署

#### Continuous Delivery - 持续交付

- **定义**：确保代码随时处于可发布状态，但需要人工决定何时部署到生产环境
- **特点**：
  - 自动化测试和构建
  - 自动部署到预生产环境
  - 人工审批生产部署

#### Continuous Deployment - 持续部署

- **定义**：在持续交付基础上，自动将通过测试的代码部署到生产环境
- **特点**：
  - 完全自动化的部署流程
  - 无需人工干预
  - 更快的功能交付

## CI/CD的优势

1. 提高开发效率

   - **快速反馈**：及时发现和修复问题

   - **减少集成问题**：避免"集成地狱"

   - **自动化流程**：减少重复性手工操作

2. 提升代码质量

   - **自动化测试**：确保代码变更不会破坏现有功能

   - **代码审查**：强制代码review流程

   - **质量门禁**：不符合标准的代码无法合并

3. 降低风险

   - **小批量发布**：降低单次发布的风险

   - **快速回滚**：出现问题可快速恢复

   - **环境一致性**：确保开发、测试、生产环境一致

4. 加快上市时间

   - **自动化部署**：减少部署时间和人为错误

   - **并行开发**：多个功能可并行开发和集成

   - **快速迭代**：支持敏捷开发模式

## CI/CD流水线阶段

1. 源码管理 (Source Control)
   - 版本控制系统（Git、SVN等）

2. 构建阶段 (Build)

3. 测试阶段 (Test)

4. 部署阶段 (Deploy)

5. 监控阶段 (Monitor)

## 常用CI/CD工具

1. CI/CD平台

   - **Jenkins**：开源自动化服务器

   - **GitLab CI/CD**：GitLab内置CI/CD

   - **GitHub Actions**：GitHub原生CI/CD

2. 容器化工具

   - **Docker**：容器化平台

   - **Kubernetes**：容器编排平台

   - **Docker Compose**：容器组合工具

3. 基础设施即代码

   - **Terraform**：基础设施管理

   - **Ansible**：配置管理

   - **CloudFormation**：AWS基础设施管理

# Github Action

GitHub Actions是GitHub提供的CI/CD解决方案，它允许您直接在GitHub仓库中自动化软件开发工作流程。

## Quick Start

1. 创建`.github/workflows/***.yml`，推到github上，github会自动识别yaml文件

### 示例：简单的Node.js项目CI流程

```yaml
name: Node.js CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [14.x, 16.x, 18.x]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linting
      run: npm run lint

    - name: Run tests
      run: npm test

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

## 常用Actions

1. 基础Actions

   - **actions/checkout**：检出代码

   - **actions/setup-node**：设置Node.js环境

   - **actions/setup-python**：设置Python环境

   - **actions/setup-java**：设置Java环境

   - **actions/cache**：缓存依赖

2. 部署Actions

   - **peaceiris/actions-gh-pages**：部署到GitHub Pages

   - **aws-actions/configure-aws-credentials**：配置AWS凭证

   - **azure/webapps-deploy**：部署到Azure Web Apps

3. 通知Actions

   - **8398a7/action-slack**：发送Slack通知

   - **appleboy/telegram-action**：发送Telegram通知
