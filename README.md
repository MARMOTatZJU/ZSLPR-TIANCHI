# ZSLPR-TIANCHI
This is the report repositroy for [**ZhejiangLab Cup Global Artificial Intelligence Competition 2018**](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.7b964899Om4fqt&raceId=231677) (Zero-shot Learning Picture Recognition track) within which we have ranked at **35/3224**  (team *AILAB-ZJU*, classification accuracy=**0.1900** during the semi-final).<br>
This competition has been completed in collaboration with the teammate from **Student AI Association of Zhejiang University**: [L. Hong](https://github.com/lanhongvp), [Y. Gu](https://github.com/shaoniangu), [X. Chen](https://github.com/XavierCHEN34), [L. Hu](https://github.com/rainofmine). In addition, [L. Hao](https://github.com/michuanhaohao)(asso. president), [Y. Li](https://github.com/wxzs5) and [Z. Tian](https://github.com/ZichenTian) have also given useful advices. Thanks to all of them.<br>
All the codes are written in Python and within PyTorch framework.

# Competition Description
## ZSL Problem
**Zero shot learning (ZSL)** is one of the semi-supervised recognition methods. Simply put, it is to identify the data categories that have never been seen before, that is, the trained classifier can not only identify the existing data categories in the training set, but also distinguish the data from the unseen categories. The ZSL problem is usually addressed with the aid of category level annotations as input, e.g. class name embedding/class annotations.

## Dataset
Around 30k images have been utilised from the raw images set(DatasetA+DatasetB+DatasetC+DatasetD) which originally contained around 150k images in total, after a manual cleaning process in order to address the noisy dataset problem. The original images are resized under different shape, and we unified them at 96x96.<br>

As for category level annotations, the pretrained **GloVe-300d word vectors** of the class names have been provided. The **class attributes annotated by competition organizer**  (e.g. "is animal", "is blue") have also been given. However, after a rapid check with aid of t-SNE visualization, we decided to only use the class name vectors as our category leval annotations for the semantic correlation attribute annotations was not shown enough salient in terms of t-SNE result.

# Related Work
ZSL problem is generally addressed in an *embedding-based* scheme within which the essential element is how to learn an effective embedding from visual feature space to the classification space. During the paper reading stage, several works have inspired us, including the following ones.<br>
Chen et al. proposed a [**semantics-preserving adverserial embedding networks (SP-AEN)**](https://arxiv.org/abs/1712.01928) aimed to address semantic loss problem by introducing an branch of image reconstructor and adding a related adverserial loss to preserve the semantic information preserving ability of the embedding space. However, this framework is heavy to implement and the reconstructor needs pretrained parameters that are forbidden by the competition.<br>
Wang et al. addressed the ZSL problem with the [**graph convolutional networks (GCN)**](https://arxiv.org/abs/1803.08035). After having  trained a classical image clssification model, its classifiers corresponding to seen categories are put into an GCN (constructed with class embedding and relationships) for ZSL generalization. Nevertheless, the reliable class relationships have been found hard to obtain with the provided attribute annotations.<br>
Xian et al. proposed a GAN-based method in which a [**feature generating WGAN model**](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2709.pdf) has been conditioned on the class word vector. The output of generator is set to be the corresponding visual feature for the classfier training. The model worked well on small dataset, but was hard to be trained on the finally merged dataset, according to our experience. <br>
Li et al. introduced a [**latent discriminative (LDF) feature**](https://arxiv.org/pdf/1803.06731) embedding in parallel with the original semantic embedding (class name word vector) in order to improve the model's discriminability. In practice, the discriminative embedding has been found hard to learn, but the semantic embedding has worked. We finally adopted the latter one for our main embedding method.

# Our Approach
In consideration of the actual competition dataset and the results of some early stage experiments, we finally adopted the following settings:
* [**SE-ResNet**](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1287.pdf) used for visual feature extraction
* **LDF baseline** (without latent space embedding) adopted for  zero-shot embedding
* **Cosine similarity** instead of the simple dot product (to be compatible with the GloVe space)
* [**TriHard loss**](https://arxiv.org/abs/1809.05864) added in the total loss (in order to form clusters in the embedding space). We use the implementation in [this repository](https://github.com/lyakaap/NetVLAD-pytorch).

# Final Settings
## Model Configuration
### Front-end networks
According to our experiment result, we adopted the SE-ResNet in the following self-designed setting. To be precise, we chose [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) BasicBlock as the basic block and transform it into the SE style. We found in practice that this self-designed work is more compatible with the competition image quality than the original setting.

| Stage | Type | Configuration | Repeat | Output dim |
|---|---|---|---|---|
|stage1| conv2d+BN2d+relu | 3x3, s=2 | 1 | 64 |
|stage2| SEBasicBlock | s=2 | 3 | 128 |
|stage3| SEBasicBlock | s=2 | 4 | 256 |
|stage4| SEBasicBlock | s=2 | 3 | 512 |


## Data Augmentation
During training phase, we employed the image augmentation modules of [imgaug](https://github.com/aleju/imgaug). We adopted *image flip*, *image rotation*, *contrast normalization* as our data augmentation techniques after a series of experiments. Specifically, we found the *contrast normalization* had an stable and significant positive effect (1-2%) in terms of the online result.<br>
During prediction phase, we also generated the submission result with the test data augmentation technique. For each test image, we made several forward propagations with different duplicates after the image augmentation on producing the probabilities of all the candidate class and chose the largest sum of probability's class as the predicted label of this test image.

## Training Method
The training details can be consulted from the `Src/confg/optimizer.py`.<br>
We adopted a *quasi-linear learning rate decrement* for 200 epochs in total, starting from a base learning rate of 7e-2 and with a weight decay (L2 regulraization) of 5e-3. We found this learning rate scheduling could avoid the overfitting phenomenom in practice.<br>
We use two different batch size, the normal size (smaller) and the size for TriHard loss (larger). The first half epochs was done with the normal size while the second half epochs with the size for TriHard loss. In our practice, we found that the normal batch size offered an acceptable model optimization during the first training stage and the size for TriHard assured the matches of TriHard loss during the second stage.

# TODO for Further Try
The limitation of time and usability of computational servers has constrain our idea verification. However, we reserve the following points for further try :
* try to perform ZSL on a different dataset with a better image quality (e.g. [LAD dataset](https://github.com/lyakaap/NetVLAD-pytorch))
* exploration with different word embeddings and attribute annotations
* exploration of the feature generating method by a better WGAN training method
* exploration of the GCN model with more reliable class relationships (e.g. from WordNet)
* etc.

# Reference
For reasons of MarkDown's clarity, all the references have been given in form of the link directed to the corresponding article/code addresses (arxiv version, github repo, etc.) in the body text.
