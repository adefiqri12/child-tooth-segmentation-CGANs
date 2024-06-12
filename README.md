# Child Tooth Segmentation with CGANs

**Panoramic Image Segmentation of Children's Teeth Using Conditional Generative Adversarial Networks and a Modified U-Net Model**

This project focuses on developing and optimizing a Conditional Generative Adversarial Networks (CGANs) model to diagnose dental caries in children.

## Dataset
We utilize a pediatric dental panoramic radiography dataset created by Zhang et al., 2023. The dataset includes 193 pairs of images and masks. For more details, you can refer to the [original publication](https://www.nature.com/articles/s41597-023-02237-5).

## Methodology
### Model
Our approach employs CGANs with a modified U-Net architecture as the generator for image segmentation. In this setup, the training mask data serve as labels to guide the model, transforming this machine learning approach into supervised learning.

### Data Augmentation
Out of the 193 pairs, we use 163 pairs for training. The data is augmented with horizontal flips, random brightness adjustments, and random contrast changes to improve the model's robustness.

### Training
The model is trained for 2000 epochs. The performance metrics on the test data are as follows:
- **IoU (Intersection over Union):** 90.75%
- **Dice Coefficient:** 91.76%
- **Accuracy:** 98%
