# Exploring Transfer Learning Approaches in Brain MRI Classification: A Comparative Analysis of CNN Architectures

## Overview

This project focuses on the classification of brain MRI images into different tumour types using various Convolutional Neural Network (CNN) architectures. The models explored include a custom-built VGG-16 architecture, as well as advanced pretrained models such as DenseNet and EfficientNet. The project aims to compare the performance of these models and evaluate the impact of data augmentation techniques, including traditional methods and Generative Adversarial Networks (GANs), on the classification accuracy.

## Dataset

The dataset used in this project is the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) provided by Masoud Nickparvar on Kaggle. The dataset is a combination of three datasets: FigShare, SARTAJ, and Br35H, containing 7023 images of human brain MRI images classified into four categories: glioma, meningioma, no tumour, and pituitary. The dataset was chosen due to its relevance and adequacy for addressing the project's research questions.

### Dataset Distribution

The dataset is divided into training and testing sets, each containing images from the four tumour categories. Please refer to the table below for a detailed distribution of images across these categories:

| Tumour Type | Training Set | Testing Set |
|-------------|--------------|-------------|
| Glioma      | 1321         | 300         |
| Meningioma  | 1339         | 306         |
| No Tumour   | 1595         | 405         |
| Pituitary   | 1457         | 300         |

## Model Architectures

### Custom VGG-16

The custom VGG-16 model follows the traditional architecture but has been adapted to handle single-channel (grayscale) MRI images. The model consists of four convolutional blocks followed by fully connected layers. Each block contains convolutional layers with ReLU activation and max-pooling. The final layers are fully connected, leading to the softmax output for classification.

### Pretrained Models

- **DenseNet**: Utilised DenseNet121, a densely connected network, which facilitates better gradient flow and efficient feature reuse, contributing to its superior performance on this dataset.
- **EfficientNet**: Employed EfficientNet-B0, known for its efficiency in scaling networks. It provided robust classification performance while maintaining computational efficiency.

## Training Procedure

The models were trained using the Adam optimiser with a learning rate of 0.0001. The training process involved 50 epochs, with early stopping implemented to prevent overfitting. The dataset was split into training and validation sets, and accuracy was used as the primary metric for evaluation.

## Data Augmentation & GAN

Traditional data augmentation techniques, including random rotations, flips, and crops, were employed to double the size of the training dataset. Although GANs were initially considered for further data augmentation, the quality of GAN-generated images did not meet the required standards for inclusion in the training process. As a result, GAN-generated images were excluded from the final model training.

## Results and Discussion

### Model Performance Metrics

The DenseNet and EfficientNet models outperformed the custom VGG-16 model, achieving higher accuracy and better generalisation on the test set. These results suggest that more advanced models are better suited for the complex task of brain tumour classification.

### Comparative Analysis

A comparative analysis was conducted to evaluate the performance of the different models. The advanced pretrained models demonstrated superior performance compared to the custom VGG-16 architecture, with DenseNet and EfficientNet achieving higher accuracy.

### Challenges and Limitations

The project faced challenges, particularly in training the custom VGG-16 model, which struggled to achieve high accuracy. The quality of GAN-generated images was also a significant limitation, leading to their exclusion from the final training process. These challenges highlight the complexity of medical image classification and the importance of selecting the right model architecture and data augmentation strategies.

## Future Work

Future research could explore more sophisticated data augmentation techniques and the integration of explainability methods within the model architectures to improve transparency and trust in AI-driven diagnostics. Additionally, further exploration of transfer learning from models pretrained on larger datasets could potentially enhance model performance.

## Conclusion

This project demonstrates the effectiveness of advanced CNN architectures like DenseNet and EfficientNet in brain tumour classification. The study highlights the importance of selecting appropriate model architectures and data augmentation strategies for improving classification accuracy in medical imaging tasks.

## References

- Filatov, D. and Ahmad Hassan Yar, G.N. (2022) ‘Brain tumor diagnosis and classification via pre-trained convolutional Neural Networks’, medRxiv. (Available at: https://arxiv.org/abs/2208.00768)
- Kang, J., Ullah, Z. and Gwak, J. (2021) ‘MRI-based brain tumor classification using ensemble of deep features and Machine Learning Classifiers’, Sensors, 21(6). (Available at: https://doi.org/10.3390/s21062222)
- Nickparvar, M. (2021), Brain Tumor MRI Dataset, [Online]. (Available at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [Accessed 18th August 2024].
- Özkaraca, O. et al. (2023) ‘Multiple brain tumor classification with dense CNN architecture using brain MRI images’, Life, 13(2). (Available at: https://doi.org/10.3390/life13020349)
- Ravinder, M. et al. (2023) ‘Enhanced brain tumor classification using graph convolutional neural network architecture’, Scientific Reports, 13(1). (Available at: https://doi.org/10.1038/s41598-023-41407-8).
- Saad, M.M., O’Reilly, R. and Rehmani, M.H. (2024) ‘A survey on training challenges in generative adversarial networks for Biomedical Image Analysis’, Artificial Intelligence Review, 57(2). (Available at: https://doi.org/10.1007/s10462-023-10624-y).
- Simonyan, K. and Zisserman, A. (2015), VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION, Oxford: University of Oxford (Available at: https://arxiv.org/abs/1409.1556)
- Younis, A. et al. (2022) ‘Brain tumor analysis using deep learning and VGG-16 ensembling learning approaches’, Applied Sciences, 12(14). (Available at: https://doi.org/10.3390/app12147282)
