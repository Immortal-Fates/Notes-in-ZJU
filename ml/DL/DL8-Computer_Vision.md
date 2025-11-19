# Computer Vision

Computer Vision (CV) focuses on enabling machines to interpret and understand visual information such as images and videos. In deep learning, CV tasks are primarily solved using neural networks — especially *Convolutional Neural Networks (CNNs)* — which can automatically learn spatial hierarchies of features.

# Image Augmentation

Image augmentation generates random images based on existing training data to improve the generalization ability of models.

- Flipping and Cropping

- change colors: brightness, contrast, saturation, and hue

# Fine-Tuning

- *transfer learning* to transfer the knowledge learned from the *source dataset* to the *target dataset*.

- *fine-tuning*: a common technique in transfer learning

  - steps:

    ![finetune](./assets/DL8-Computer_Vision.assets/finetune.svg)

    1. Pretrain a neural network model, i.e., the *source model*, on a source dataset.
    2. Create a new neural network model, i.e., the *target model*. This copies all model designs and their parameters on the source model except the output layer.
       - We assume that these model parameters contain the knowledge learned from the source dataset and this knowledge will also be applicable to the target dataset.
       - We also assume that the output layer of the source model is closely related to the labels of the source dataset; thus it is not used in the target model.
    3. Add an output layer to the target model, whose number of outputs is the number of categories in the target dataset. Then randomly initialize the model parameters of this layer.
    4. Train the target model on the target dataset, such as a chair dataset. The output layer will be trained from scratch, while the parameters of all the other layers are fine-tuned based on the parameters of the source model.

```
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

- Generally, fine-tuning parameters uses a smaller learning rate, while training the output layer from scratch can use a larger learning rate.

# Object Detection and Bounding Boxes
