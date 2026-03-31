# Beyond the Code

Being a good software engineer isn’t just about writing code that works. It’s about writing code that others (including future you) can understand, maintain, and build upon.

## One-way communication

Your goal is to capture and convey the *why*, not just the *what*.

Code comments: *why* something is done a particular way, not *how* it works (which is what the code shows).

Types of comments that are nearly always worthwhile:

- TODOs
- References
- **Correctness arguments**: Explain *why* non-trivial code produces correct results. The code shows the steps; a comment explains why those steps work.
- **Hard-learned lessons**
- **Rationale for constants**: Magic numbers deserve explanation.
- **Load-bearing choices**: If correctness depends on a seemingly-innocent implementation detail (e.g., “must be a BTreeSet because iteration order matters below”), call it out explicitly.
- “Why not”s

### READMEs

A good one answers four questions immediately: What does this do? Why should I care? How do I use it? How do I install it? In that order.

Structure it like a funnel: a one-liner and maybe a visual demo at the top so someone can decide in seconds if this solves their problem, then progressively add depth.

### Commit messages

We can use the LLM to help us in writing messages but LLM will only have access to the *what*, not the *why* so a useful trick is to specifically tell the LLM you’d like a commit message focused on the “why”

## Collaboration

### Contributing

用户数量通常远多于贡献者，贡献者数量也比维护者多一个数量级。因此必须确保你的贡献具有高信噪比，并且值得维护者投入时间

错误报告：

- **Environment**: OS, version numbers, relevant configuration
- **What you expected** vs **what actually happened**
- **Steps to reproduce**: Be specific. “Click the button” is less useful than “Click the Submit button on the /settings page while logged in as an admin.”
- **What you’ve already tried**: This prevents duplicate suggestions and shows you’ve done some investigation

If you’re looking to make a code contribution, you’ll also want to familiarize yourself with the contribution guidelines. Many projects have a `CONTRIBUTING.md` — follow it.

### Reviewing

代码审查

Review is also one of the fastest ways to learn.

## Education

Asking good questions is a skill that makes you better at learning from anyone, not just perfect explainers.

 “[How to ask good questions](https://jvns.ca/blog/good-questions/)” and “[How to get useful answers to your questions](https://jvns.ca/blog/2021/10/21/how-to-get-useful-answers-to-your-questions/)” that are worth reading.



## AI etiquette(礼仪)

> [!TIP]
>
> 如果你的目标是学习作为你正在做的工作的一部分，请记住，如果让人工智能完成大部分或全部工作，可能会自我挫败;你更可能学到更多关于提示（甚至复习 AI 输出）的知识，而不是任务本身。尤其是在学习阶段，重点可能是旅程本身，而非终点，因此用人工智能“快速找到解决方案”反而是反目标。

## Refernces

- https://missing.csail.mit.edu/2026/beyond-code/