# Transformer Overview and Core Ideas



## Transformers

There are different kinds of transformer models. Broadly, they can be grouped into three categories:

- GPT-like (also called *auto-regressive* Transformer models)
- BERT-like (also called *auto-encoding* Transformer models)
- T5-like (also called *sequence-to-sequence* Transformer models)

### How do Transformers work?

- Transformers are language models: This means they have been trained on large amounts of raw text in a self-supervised fashion.

  > [!NOTE]
  >
  > Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model.
  >
  > That means that humans are not needed to label the data!

  For specific task, the general pretrained model then goes through a process called *transfer learning* or *fine-tuning*.

- Transformers are big models
