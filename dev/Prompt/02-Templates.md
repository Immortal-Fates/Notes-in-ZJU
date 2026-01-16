

# Prompt Templates

[TOC]

## High-Quality Study Resources

I am learning **[topic]** and I want the study plan formatted as a **Typora-optimized markdown table** with the following columns:

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

> Tips: Anything else you want, remember to chat more
>
> - Consider your level: you want a course or just some papers?
> - Whether it's theory or practical operation

## Deep Understanding of a Specific Section

I am studying the section **[Section Title or Topic]** from my study plan.
 Please provide:

1. A clear, graduate-level explanation of the concepts, including definitions, key ideas, and intuitive interpretations.
2. A structured breakdown of the topic into its essential components.
3. Concrete examples to illustrate each major concept.
4. If the topic is technical, include relevant formulas, diagrams (mermaid if needed), or pseudocode.
5. A comparison to related concepts when helpful.
6. The typical pitfalls or misunderstandings students have about this topic.
7. Questions I should be able to answer after studying this section (self-assessment checklist).
8. If applicable, mini-exercises that strengthen understanding.
9. Optional: advanced or research-oriented extensions for deeper study.

Answer in a clear, professional, and concise structure suitable for a graduate-level learner.

## Notebook

introduce the [section] in the following way - Takeaway - Prior - Core Mechanism - Pipeline - Pros - Cons

summarize they in English and lay they in typora format like this

- Takeaway

- Prior

...

and show out the math formula

## Discussion Roles

Scientific Peer Reviewer. The paper has not been published yet and is currently submitted to a top conference where you’ve been assigned as a peer reviewer. Complete a full review of the paper answering all prompts of the official review form of the top venue in this research area (e.g., NeurIPS). This includes recommending whether to accept or reject the paper

Academic Researcher. You’re a researcher who is working on a new project in this area. Propose an imaginary follow-up project not just based on the current but only possible due to the existence and success of the current paper.

Hacker. You’re a hacker who needs a demo of this paper ASAP. Implement a small part orsimplified version of the paper on a small dataset or toy problem. Prepare to share the core code of the algorithm to the class and demo your implementation. Do not simply download and run an existing implementation – though you are welcome to use (and give credit to) an existing implementation for “backbone” code.

## 给傻子（我）用的

我是一名智力低下的博士生，我想学习一下这篇论文/文献/资料，请用傻子都能懂的语言详细给我讲一下这篇论文/文献/资料怎么做的，特别是模型和实证方面

## Draw.io

> check the [github](https://github.com/DayuanJiang/next-ai-draw-io)

直接在Codex CLI中用mcp

