# Object Detection

# Papers

- **Object Detection YOLO Algorithms and Their Industrial Applications: Overview and Comparative Analysis**. Kang Shizhao et.al. **Electronics**, **2025-3-11**,([pdf](..\..\papers\Object_Detection\yolo_review.pdf)) ([link](https://doi.org/10.3390/electronics14061104)). 
  
  - YOLOv1 to YOLOv7

    ![b0385e635dd60c899e0d3901aa6c18c5](./assets/Object_Detection.assets/b0385e635dd60c899e0d3901aa6c18c5.png)
  
  - YOLO在工业中的应用与改进
  
    ![04ca9a0a2c18e3f6a1ba5a9573199e1b](./assets/Object_Detection.assets/04ca9a0a2c18e3f6a1ba5a9573199e1b.png)
  
  - Sec4 可以看看Practical Adaptations of YOLO Series Algorithms in Industrial Fields  
  
- **YOLO advances to its genesis: a decadal and comprehensive review of the You Only Look Once (YOLO) series**. Ranjan Sapkota et.al. **Artif Intell Rev 2025**([pdf](..\..\papers\Object_Detection\YOLO_advances_to_its_genesis.pdf))([link](https://doi.org/10.1007/s10462-025-11253-3))

- **Surveying You Only Look Once (YOLO) Multispectral Object Detection Advancements, Applications, and Challenges**. Gallagher James E. et.al. **IEEE Access**, **2025**, ([pdf](..\..\papers\Object_Detection\Surveying_You_Only_Look_Once_(YOLO)_Multispectral_Object_Detection_Advancements,_Applications,_and_Challenges.pdf))([link](https://doi.org/10.1109/access.2025.3526458)).

- **SSD: Single Shot MultiBox Detector**. Liu Wei et.al. **No journal**, **2016**, ([pdf](..\..\papers\Object_Detection\SSD.pdf))([link](https://doi.org/10.1007/978-3-319-46448-0_2)). 

- **A Benchmark Review of YOLO Algorithm Developments for Object Detection**. Hua Zhengmao et.al. **IEEE Access**, **2025**, ([pdf](..\..\papers\Object_Detection\A_Benchmark_Review_of_YOLO_Algorithm_Developments_for_Object_Detection.pdf))([link](https://doi.org/10.1109/access.2025.3586673)).



# basic fundamentals

- IoU（Intersection over Union） 衡量两个区域重合度的指标：
  $$
  \text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
  $$
  
- Anchors Base

  ![image-20251106135409183](./assets/Object_Detection.assets/image-20251106135409183.png)

  **在特征图的每个网格位置预先放置若干“模板框”**（不同尺度、长宽比）。网络只需**预测这些模板的偏移量（Δx, Δy, Δw, Δh）以及类别/置信度**，就能得到最终目标框。它把“在整幅图里到处找框”变成“从一组模板微调”。

- Anchors Free

  **Anchor-free**（如 FCOS、CenterNet、YOLOX、PP-YOLOE、Ultralytics YOLOv8 的默认检测头等）不再放模板框，而是让**像素/点/中心**直接回归框的四边距离，并用动态分配做正负样本选择。优点是**更简洁、少超参、泛化好**；在很多场景表现与 anchor-based 相当或更优。





# YOLO





# Cmd

```
set PYTHONUTF8=1
autoliter -i ./Object_Detection.md -o ../../papers/Object_Detection
```

