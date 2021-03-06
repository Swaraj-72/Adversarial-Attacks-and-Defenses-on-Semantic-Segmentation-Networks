# Adversarial Attacks and Defenses on Semantic Segmentation Networks
Welcome to the repo of Adversarial attacks and defenses on Semantic Segmentation. Adversarial attacks prove to be the biggest challenge for deep neural networks. Even many state-of-the-art architectures are vulnerable to these attacks. Although many journals have tried to explain the defense mechanisms, we introduce other such methods focussing specially on Semantic Segmentation models and datasets like MS-COCO, PascalVOC, Cityscapes. 

## Abstract
The project explores and analyses various types of adversarial attacks, untargeted and targeted, used to fool the deep neural networks. Specially, we focus on the semantic segmentation networks with state-of-the-art pretrained models trained on two popular datasets, namely PascalVOC and Cityscapes.We restrict our scope only to white box attacks where the attacker has access to the model parameters.
 We first show how an imperceptible noise added to the images makes the network predict incorrect classes or segmentation maps. Next, we describe a method of mutli-scaling to reduce the risk of such attacks. Then we experiment similar attack now with a particular image among the batch as target. We plan the attack such that the network outputs the desired class or targeted image's segmentation map for all the images in the batch. We then explore methods of image reconstruction to lessen the damage caused by such attacks.
For the entire experimentation, we make use of mean Intersection over Union (mIoU) and Pixel Accuracy as the metrics to measure how effective the attacks and defense mechanisms are on the semantic segmentation networks. Our findings pave way for robust semantic segmentation models which could be potentially implemented in safety critical applications.

## Plan of Attack
The main idea is to perform adversarial attacks based on Projected Gradient Descent(PGD). There are two different types of attacks in particular we deal with. All our attacks are performed on the state-of-the-art pre-trained networks trained on PascalVOC and Cityscapes Datasets. We explore how these attacks fool the network to output a completely different segmentation map compared to that of the ground truth.
Especially when the attack is targeted, we observe that the resulting segmentation map looks similar to that of the desired image. This way the attacks are performed to expose the vulnerability of segmentation networks. 
We also propose a defense mechanism namely multi-scaling for both targeted and untargeted attacks. This way it is observed that the mIoU and Accuracy values are being improved. We also discuss about the transferability of such attacks wherein a perturbated image obtained using one network is passed through another network to check the effect of perturbation.

## Results
### Untargeted Attacks
The results shown below clearly indicate a unqiue segmentation map is produced for each image which is different from the original prediction of the network.
![untargeted_orig](https://user-images.githubusercontent.com/65396498/129730846-deb31b88-de11-4a90-8ba0-1b6b50693adf.jpg)
![untargeted](https://user-images.githubusercontent.com/65396498/129730860-f2e60f01-1e6e-4927-bf03-859c9c413c01.jpg)

In the case of untargeted attacks, our goal is to maximize the loss simultaneously ensuring the perturbations are not visible to the naked eye.
### Targeted Attacks
Here the second image in the batch of 4 images is targeted and it is observed that all other images now output similar segmentation map as the one that is targeted.
![Tar_Org](https://user-images.githubusercontent.com/65396498/129731350-108b56c9-20e7-487d-8c40-287cab9d86c0.jpg)
![Tar_adv](https://user-images.githubusercontent.com/65396498/129731372-541d3553-98f5-4375-b500-69f54d5017bf.jpg)
 Here, the goal is to minimize the loss between the targeted segmentation map and the other images' segmentation maps. Here too the perturbations shouldn't be visible to the naked eye.
 
 ### Image Scaling Attacks
 Here we use an target image to perturbate in a way that the original image predicts actual segmentation map when sent in with original size but when the image is scaled, then it outputs the targeted segmentation map. This is very good area of research because in many of the applications of deep learning, images are always resized to be input to the network. Sample images to be updated soon.

The scaling part of this attack is completely taken from the repository by Erwin Quiring (https://github.com/EQuiw/2019-scalingattack) where they have described a novel method to illustrate the image scaling attack. We have applied his concept to our targeted segmentation map generation. 


