# Object Detection Recipe

introduce the modern recipe of object detection

## Loss

| Task                | Typical Loss                |
| ------------------- | --------------------------- |
| Classification      | Focal Loss / Varifocal Loss |
| Localization        | CIoU / DIoU / SIoU          |
| Objectness          | BCE / IoU‑aware confidences |
| Anchor‑free centers | Heatmap Focal / L1          |

- __Focal Loss for Dense Object Detection.__ *Tsung-Yi Lin et al.* __IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017__ [(Arxiv)](https://arxiv.org/abs/1708.02002) [(S2)](https://www.semanticscholar.org/paper/1a857da1a8ce47b2aa185b91b5cb215ddef24de7) (Citations __3109__)

  - Takeaway: Focal loss was proposed to handle the class imbalance problem in **dense object detection**.

  - Motivation: extreme imbalance between foreground and background classes

  - Core Mechanism: a modification of the standard **Cross-Entropy Loss** using a **modulating factor**
    $$
    FL(p_t) = -\alpha_t(1 - p_t)^\gamma \log(p_t)
    $$

    - $p_t$ is the model's estimated probability for each class.
    - $\alpha_t$ is a balancing factor that adjusts the weight of the positive and negative classes (used to handle class imbalance).
    - $\gamma$ is a **focusing parameter** that reduces the loss for well-classified examples and focuses more on hard examples.

  - Pros

    - Better Performance on Dense Datasets

  - Cons

    - Hyperparameter Sensitivity
    - **Computational Overhead**: Although the focal loss term is computationally lightweight, the model might require additional training time to adjust to the new loss function.

- __Generalized Focal Loss: Towards Efficient Representation Learning for Dense Object Detection.__ *Xiang Li et al.* __IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022__ [(Link)](https://doi.org/10.1109/TPAMI.2022.3180392) [(S2)](https://www.semanticscholar.org/paper/4b457ca003b4d05b2fd6af22c283fe8b4e211a2f) (Citations __257__)

  - Takeaway: designed to improve performance in object detection tasks by extending the idea of **class imbalance handling** to more general cases

  - Core Mechanism:
    $$
    \text{GFL} = \text{CE Loss} + \text{IoU Loss} + \text{Center-ness Loss}
    $$

    - **IoU Loss**: It directly incorporates **IoU (Intersection over Union)** into the loss, making it more robust for object detection tasks.

    - **Center-ness Loss**: This term addresses issues related to object localization by focusing on the central regions of an object.

    - **CE Loss**: Standard Cross-Entropy loss.

