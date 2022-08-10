
# Rotated object Detection

**Problem:**
A data synthesizer generates images and labels. The goal is to train a model with at most 4.5 million trainable parameters which determines whether each image has a star and, if so, finds a rotated bounding box that bounds the star.

More precisely, the labels contain the following five numbers, which your model should predict:
* the x and y coordinates of the center
* yaw
* width and height.

If there is no star, the label consists of 5 `np.nan`s. The height of the star is always noticeably larger than its width, and the yaw points in one of the height directions. The yaw is always in the interval `[0, 2 * pi)`, oriented counter-clockwise and with zero corresponding to the upward direction.

**Important Points**
1. ResNet18 inspired Network for rotated object detection using Modulated Rotation Loss
2. Architecture: ResNet18 with classification and regression heads
3. Loss Function: Modulated Rotation Loss
4. Model Summary: modelSummary.txt
5. Final Score: score.txt
6. Final Model weights: model.pickle
7. Version requirements: requirements.txt
8. Configuration for teh model: config.yaml

**Method**
1. Generated a fixed number of images (defined in the config file) for train and validation dataloader with high noise and rotated objects. The probability of star being present in any image is set to 0.8
2. Normalize the values such that the range of the labels is [0, 1]
3. Add a classification label for every image based on the values of the generated labels
4. The images are passed through a ResNet18 inspired model with seprate classification and regression heads
5. Implement modulated loss function, to bridge the discontinuity in the values of rotation angle (yaw)

**Opportunities to Improve:**
1. Add a more complex network (Upsampling from the bottleneck or FPN) for the regression head
2. Improvise the loss function to take into account the imbalance between the classes
