

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

## Read Paper

I am reading the paper: **[Insert Title / Abstract / or Full Paper Text Here]**. Please help me analyze it by providing:

- **High-level overview**

  - Research problem
  - Motivation
  - Why this problem matters
  - What gap the paper fills

- **Core contributions**

  - List each contribution explicitly
  - Clarify what is new vs. what is incremental

- **Method details (structured and rigorous)**

  - Key assumptions
  - Model/algorithm formulation
  - Important equations (explain each term)
  - Pipeline or architecture (diagram if needed)
  - Step-by-step breakdown

- **Experimental setup**

  - Data
  - Baselines
  - Metrics
  - Key decisions that influence results

- **Results and interpretation**

  - What the results actually prove
  - Comparative analysis
  - Any hidden limitations from the numbers

- **Strengths and weaknesses**

  - Technical strengths
  - Methodological weaknesses
  - Risks of overclaiming
  - Realistic impact

- **Connections**

  - How this relates to prior work
  - What category of approach this fits into
  - Similar ideas in other top-tier papers

- **If code is available**

  - How the implementation corresponds to the paper
  - What parts deserve more attention

- **Deep understanding tools**

  - Intuitive analogies
  - Concrete examples
  - Simplified explanation for core mechanisms
  - Potential exam/interview questions based on this paper

- **Future directions**

  - What follow-up research is possible

  - What is missing

  - What this paper enables for the field

> Tips: choose the prompts to integrated.



