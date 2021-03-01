# Crowd Counting using Convolution Neural Network

### Archit Jain

### Master of Science


## Abstract

### Crowd Counting using

### Convolution Neural Network

### Archit Jain

Crowd Counting has grown popular in recent years due to its vast applications. Crowd
counting is useful in surveillance and safety, attendance at events, managing social distanc-
ing, infrastructure, and commute planning. The crowd counting done via image processing
is quite challenging due to the low lighting, distortion, and occlusion present in the images
where the crowd is dense.
This paper implements a crowd counting method using Convolutional Neural Network
(CNN). Convolutional Neural networks are computationally heavy and take a lot of time
during the training. They also don’t work very well in a highly congested crowd image.
The method proposed in the report aims to resolve two issues of getting a better estimate
of a highly dense crowd and reducing the network’s complexity making model train easier.
The trained model produces the input image’s density map containing the total count.
Crowd counting has numerous applications in the modern world. With the rise in Ma-
chine Learning algorithms and techniques, a more inclusive model can be developed to
predict crowd behavior and tendencies using video input. With high accuracy and robust
crowd counting models, many businesses can rely on them for decision-making and managing the crowd. In normal and critical times, having an open-sourced crowd counting
application performing on the states of the art level will be beneficial to professionals in the
industries and the academic community of computer science.


## Contents


- Abstract
- 1 Introduction
- 2 Related Work
- 3 Problem Statement
- 4 Methodology
- 5 Timeline
- 6 Conclusion
   - 6.1 Limitations
   - 6.2 Future Work
- References
- 1 Images illustrating crowd List of Figures
- 2 Basic CNN model
- 3 Pooling Layer working
- 4 Stride example
- 5 Graphical representation of ReLU layer
- 6 CNN Architecture (Z. Li et al., 2019)
- 7 Project Timeline


## 1 Introduction

Crowd counting is an activity/task of counting the number of people in a crowd’s image.
Counting people in a single frame manually can repetitive, and complex job as many people
can be packed in a single picture, as shown in Figure 1. Machines can be automated to do
this task more accurately and precisely.

```
![Images illustrating crowd](https://github.com/Archit-Jain/Crowd-Counting-CNN/blob/main/crowdCounting.jpg)

Figure 1. Images illustrating crowd
```
Crowd counting has become quite popular among researchers in recent years. The pro-
cess has evolved over the years. With the human population on the rise, and the crowd turns
up of social events like political gatherings, sports, and public meetings are increasing. It
has become essential to develop the tools to monitor and analyze crowd behavior (Sindagi
& Patel, 2018). Crowd Counting is also beneficial for surveillance and security. They can
estimate the peak hours at retail or public place like airports. They can also be employed for
Infrastructure planning to estimate the project’s scale. Crowd Counting can also be adopted
to plan commute. Taking the transit less crowded also enables social distancing.
Convolutional Neural Network(CNN) is a Deep Neural Network design based on the
human biological model. They are the artificial neural network, an immensely popular
technique used mainly on object recognition and image analysis. CNN's are specialized in
detecting some sort of pattern from the input data. This pattern detection makes CNN so
useful in the image analysis. CNN has a network of hidden layers known as convolutional
layers (Pan et al., 2016). A convolutional layer takes the input data transforms it and gives
its output as an input to the next convolutional layer as can be seen in Figure 2.
CNN has many types of layers.

**Filter** : A filter is a feature learning matrix containing some numerals. A filter is gen-
erated after the model is trained to detect a certain type of object. A filter is nothing but
a matrix containing the numeric values. A dot product is taken from the filter and input
image. If the filter matches exactly as the object it needs to detect in the image, the dot
product becomes 1. The dot product’s value becomes closer to 1 as the object detected is
closer to the feature exacter in the filter.


```
Figure 2. Basic CNN model
```
A **Pooling layer** is used to reduce the dimensions of the image. For an image 1900x720,
a neural network with 6 million neurons will be needed. With the pooling layer, the object
in the matrix can be replaced by a single value of the maximum number (L. Zhang et al.,
2018) as can be seen in Figure 3 (savyakhosla, 2019). For a grid of 3x3, each input will be
halved in size. The pooling layer of type max is the best suited for the problem of crowd
counting.

```
Figure 3. Pooling Layer working
```
**Stride** is the number of spaces a filter moves after its operation. For a stride of 1, the
filter will move by 1 column. And for a stride of 2, the filter moves two spaces, as shown in
Figure 4 (NeuralNet, 2019). Higher the stride, lesser the computation. Increasing the stride
results in the loss of certain data. Implementing the stride has a certain trade-off between
the computation speed and accuracy, which needs to be analyzed before selecting the right
stride value.
A **rectified linear unit** or **ReLU** is a layer that turns the negative values present in the
matrix to zero as per Figure 5 (Liu, 2020). This layer also reduces the computational cost
by removing unnecessary data from the network (L. Zhang et al., 2018). Also, it easier to
process the convolutional layer by reducing the network’s complexity.


```
Figure 4. Stride example
```
```
Figure 5. Graphical representation of ReLU layer
```
## 2 Related Work

Crowd Counting is achieved through various methods with continuous improvement in the
accuracy and its applications over time. It all started in 2013 by Cheng Change as he is used
the detection method as the first technique for counting crowds Loy et al. (2013). However,
this technique only could count the number of people who are clearly visible in the image.
The traditional approaches are classified into three categories as per Loy et al. (2013) de-
tection, regression, and density-based. The detection-based technique does not work well
in high-density crowds or low lighting conditions.
The regression technique came is one of the most researched techniques and came to
be used around 2016 Huang et al. (2016). Y Wang uses the perspective map using the
regression model. This approach separates the background from subjects and can give an
accurate count for the same location with the same perspective. The drawback is its only
works for trained backgrounds.
While these earlier methods were successful and provided the accurate count, the so-
lutions could not have been applied to problems of spatial information Pham et al. (2015)
proposed to use a density map using the local features, thereby incorporating the spatial in-
formation. He used the random forest approach to create the density mapping (Pham et al.,
2015).
Recently researchers have focused their attention on deep learning approaches to get


more desired results. Shou uses the basic Convolutional Neural Network (CNN) for the
crowd counting techniques in 2016 Y. Zhang et al. (2016). CNN is the most recent approach
to crowd counting and is being continually being experimented upon for the best results. The
CNN approach does not face any issue with the dense crowd, cluttering, collusion, and also
includes the spatial data in the image. The perspective map for training does not restrict it.
CNN currently provides the best accuracy lowest MSE and is the state of the art approach
to the crowd counting problem.
The CNNs are classified into four different categories based on their properties (Sindagi
& Patel, 2018). Basic CNNs, Scale aware models, Context-aware models, and Multitask
framework.
The approach focused on this paper is mostly related to scale aware models. Scale Aware
models take into consideration different head sizes of the crowd for outputting the count.
Differing from the other CNN approaches where classification and regression are used in
combination with Neural Network (Pan et al., 2016), Scale adaptive approaches use the
multi-resolution or multiscale architectures (L. Zhang et al., 2018). Lu Zhang uses the
same approach, uses fixed small receptive fields, and outputs the result in the same format.
Similar works are present in S. D. Khan’s works with scale-specific detectors combined
with the CNN. He also optimized parameters from end to end, differentiating from Zhang’s
work. Zhang used the residual channel and weighing unit to enhance the network’s feature
learning capabilities. However, it is a computationally costly approach.

## 3 Problem Statement

It is quite challenging to estimate the number of people in a highly dense crowd from an
image due to lighting, occlusions, congested scenes, and difficulty training the counting
model.

## 4 Methodology

The research and development are divided into three phases: Algorithm and Model Design,
Development, and Analysis.
Phase I Algorithm and Model design: The Convolutional Neural Network is a great
technique to handle object detection. In the case where people in the image can be of various
shapes and sizes, they may be completely or partially visible the traditional approaches to
this problem fails here. The approach of head detection is the best-suited approach to the
problem. The CNN model will detect the heads in the image and output the number of
counts from the result. The model will be trained to detect the head, and the generated filter
will be used to convolve over the image resulting in the headcounts. The CNN architecture
is illustrated in the Figure 6 (Z. Li et al., 2019)
The challenge to the approach is the head in the image can be of different shapes and
sizes. To tackle this problem the proposed CNN model will have filters of various sizes.


```
Figure 6. CNN Architecture (Z. Li et al., 2019)
```
3x3, 5x5, and 9x9. Thus, using this approach, higher accuracy and precision can be achieved
while detecting heads in the image.
Phase II Model Development Once the algorithm and model are designed, development
and testing can be started following the software development life-cycle (Jain, 2019). The
dataset containing crowd images are used to train the model. The model is trained using
the PyTorch library. Once the model is trained using the training data from the dataset.
The generated filter is used for the inference model to get the people’s count in the image.
As the filter convolves over the whole image, the matrix of the result will contain positive
numeric values wherever the head is detected. After each convolution, a max-pooling layer
is applied to down-sample the input, followed by the rectified linear layer. The filters of
different sizes are applied to capture the head of diverse sizes. This matrix, in the end, is
then flattened, and output is released. Another approach for counting is using the dilated
CNN can also be used to replace the pooling layers. And using a Deep CNN to get the
density map (X. Li & Chen, n.d.)
Phase III Analysis and Improvement In the final phase, the analysis of the model and
Accuracy and MSE is calculated to compare it with the other benchmark model. Fine-tuning
is done on the model to get rid of anomalies. Usage of the max-pooling layer automatically
avoids the over-fitting of the model. The Mean Squared Error and the Mean Absolute error
is calculated and is analyzed amongst the best model for crowd counting.

## 5 Timeline

The timeline is designed based on the software development life cycle keeping testing and
updating the model in mind.


```
Figure 7. Project Timeline
```
## 6 Conclusion

The paper present here proposes an implementation of a Convolutional Neural Network to
improve the computational speed and increase the accuracy of a crowd counting model.
The model is tested and analyzed with benchmark CNN approaches to evaluate the model.
While the detection, regression, and density-based approach provide accurate results. The
CNN approach is more robust and provides accurate results.

### 6.1 Limitations

The CNN model does not consider the object’s orientation and will require more filters in a
different orientation to detect the object. The adding of filters can be computationally costly
and will slow the model.

### 6.2 Future Work

Future work can include implementing CNN for live video and giving output in real-time.
Hosting the model on the cloud can increase the computational speed by tenfold. As crowd
counting has numerous industrial and academic applications, the idea can be further ex-
tended to count different objects like fishes to track and control the fishing in an area or
other similar approach. The crowd counting on humans can be further expanded to under-
stand, predict, and analyze crowd movement patterns. The scope of crowd counting is vast,
and this area of field needs continuous research and development.


## References

Huang, X., Zou, Y., & Wang, Y. (2016). Cost-sensitive sparse linear regression for crowd

```
counting with imbalanced training data. 2016 IEEE International Conference on
Multimedia and Expo (ICME , 1–6.
```
Jain, A. (2019). Crowd counting accurately using image recognition backed by scale-adaptive

```
convolutional neural network [Retrieved Scholarship in ISTE].
```
Li, X., & Chen, D. (n.d.). Csrnet: Dilated convolutional neural networks for understanding

```
the highly congested scenes [arXiv:1802.10062 [cs], Apr. 2018, Accessed: Jan. 21,
```
2021. [Online]. Available:]. [http://arxiv.org/abs/1802.10062.](http://arxiv.org/abs/1802.10062.)

Li, Z., Li, J., Xie, L., & Liu, J. (2019). Scale-adaptive cnn based crowd counting and dy-

```
namic supervision. 2019 IEEE International Conference on Multimedia & Expo
Workshops (ICMEW , 408–413.
```
Liu, D. (2020). https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f

Loy, C., Gong, S., Chen, K., & Xiang, T. (2013). Crowd counting and profiling: Methodol-

```
ogy and evaluation. Modeling, Simulation and Visual Analysis of Crowds : A Mul-
tidisciplinary Perspective , (Generic), 347–382.
```
NeuralNet, M. (2019). [http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-](http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-)

```
output-size-of-convolutions.html
```
Pan, S.-Y., Guo, J., & Huang, Z. (2016). An improved convolutional neural network on

```
crowd density estimation. ITM Web of Conferences , 7 , 5009.
```
Pham, V.-Q., Kozakaya, T., Yamaguchi, O., & Okada, R. (2015). Count forest: Co-voting

```
uncertain number of targets using random forest for crowd density estimation. 2015
IEEE International Conference on Computer Vision (ICCV , 3253–3261. https://doi.
org/10.1109/ICCV.2015.372.
```
savyakhosla. (2019). https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/

Sindagi, V., & Patel, V. (2018). A survey of recent advances in cnn-based single image

```
crowd counting and density estimation. Pattern Recognition Letters , 107 , 3–16.
```
Zhang, L., Shi, M., & Chen, Q. (2018). Crowd counting via scale-adaptive convolutional

```
neural network. 2018 IEEE Winter Conference on Applications of Computer Vision
(WACV , 1113–1121.
```
Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-image crowd counting via

```
multi-column convolutional neural network. 2016 IEEE Conference on Computer
Vision and Pattern Recognition (CVPR , 589–597.
```


