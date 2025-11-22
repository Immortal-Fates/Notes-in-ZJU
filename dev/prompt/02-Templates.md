

# Prompt Templates

[TOC]

## High-Quality Study Resources

I am learning **[topic]** and I want the study plan formatted as a **markdown table** with the following columns:

- Date
- Topics (allow multiline using `<br>` within the cell)
- Readings (each reading should be a separate line with a link)

Please generate:

1. A multi-week study plan organized by date.
2. For each date, include **specific, scoped topics**, written concisely and using `<br>` for line breaks.
3. For each topic, select **high-quality readings from authoritative global sources only** (arXiv, top-tier conferences such as NeurIPS/ICLR/CVPR, MIT/Stanford/CMU courses, reputable textbooks, or well-regarded blogs).
4. For each reading, include both the **title** and a **direct link**.
5. Format the final output strictly as a **markdown table**, similar to the example below:

| Date | Topics                                                       | Readings                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2/4  | **Week 1: Introduction** 1. Course syllabus and requirements 2. Introduction to AI and AI research | 1. [Foundations and Trends in Multimodal Machine Learning](https://arxiv.org/abs/2209.03430) 2. [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406) |

Additional preferences:

- My background: **[your academic level]** Postgraduate students from one of the best schools in the world
- My duration constraints: **[e.g., 2 weeks / 1 month / flexible]**
- My learning goal: **[e.g., research-level understanding / practical mastery]** 

## Excalidraw

你现在是 Excalidraw 图形生成器。请**只输出有效的 Excalidraw JSON**，不要包含任何多余文字、解释或注释。

生成要求

- 严格返回一个 JSON 对象，键包含：elements[], appState{}, files{}（若无嵌入文件可为空对象）。
- elements[] 中的每个元素必须包含常见字段：
  - id（短随机字符串）、type（'rectangle'|'ellipse'|'diamond'|'arrow'|'line'|'text'）、x、y、width、height、angle、strokeColor、backgroundColor、fillStyle、strokeWidth、strokeStyle、roughness、opacity、roundness（可选）、seed、version、versionNonce、isDeleted、groupIds[]
  - 文本元素需加：text、fontSize、fontFamily、textAlign、verticalAlign、baseline
  - 连线/箭头需加：points（[[0,0],[w,h]]）、startBinding、endBinding、startArrowhead、endArrowhead
- 坐标系以画布左上角为 (0,0)。请确保图形**不重叠且布局清晰**。
- 颜色建议：strokeColor "#1e293b"、backgroundColor "#e2e8f0"（或按主题参数），文本为 "#0f172a"。
- 统一风格：fillStyle "solid"、strokeWidth 2、roughness 0、opacity 100。
- 图形语义化命名：为关键元素设置 text 或在同一位置提供文本标签。
- 返回前请校验：JSON 语法正确；所有 id 唯一；箭头绑定到目标元素；文本元素 width/height 足以容纳文字。

输出格式：仅输出 JSON，放在code block中