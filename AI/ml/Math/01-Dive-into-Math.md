# Explore the essence

- __Convolutional neural networks with low-rank regularization.__ *Cheng Tai et al.* __arXiv: Learning, 2015__ [(Arxiv)](https://arxiv.org/abs/1511.06067) [(S2)](https://www.semanticscholar.org/paper/d5b4721c8188269b120d3d06149a04435753e755) (Citations __483__)



## Hungarian Bipartite Matching

- Takeaway: Hungarian bipartite matching (also called the **Hungarian Algorithm** or **Kuhn–Munkres algorithm**) finds an **optimal one-to-one assignment** between two sets (e.g., predicted objects and ground-truth objects) that **minimizes total matching cost**.

- Problem Statement: Given two sets:

  - Set A: predictions (N items)
  - Set B: ground truths (M items)

  We want to assign each ground truth to **exactly one** prediction (or to a “no-object” prediction if unmatched). This is the classic **linear assignment problem**, represented as a cost matrix:

  C[i, j] = cost of assigning predicted item i to ground truth item j

  The goal: Find a matching M that minimizes total cost: Sum over (i, j) in M of C[i, j]

## Inductive Bias

- Takeaway: Since we only ever see a limited number of examples, the model must “guess” how to behave on new inputs. The rules it uses to make that guess are its inductive bias.

- Intuition:

  - There are infinitely many functions that agree with your training data but behave differently on unseen points.

  - A learning algorithm must **prefer** some of these functions over others (e.g., “simpler functions,” “smooth functions,” “locality,” “translation invariance,” etc.).

    > Examples: 
    >
    > - Linear regression assumes the relationship is *approximately linear*.
    > - K-NN assumes *points close in input space have similar outputs*.

  That **preference** is exactly the inductive bias.

## Energy and Policy Considerations for Deep Learning in NLP

- __Energy and Policy Considerations for Deep Learning in NLP.__ *Emma Strubell et al.* __ArXiv, 2019__ [(Arxiv)](https://arxiv.org/abs/1906.02243) [(S2)](https://www.semanticscholar.org/paper/d6a083dad7114f3a39adc65c09bfbb6cf3fee9ea) (Citations __5983__)
