# Kaggle-Catheter-and-Line-Position-Challenge
Serious complications can occur as a result of malpositioned lines and tubes in patients. Checking these can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity. The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists can be busy reporting other scans.

In this competition, the goal was to detect the presence and position of catheters and lines on  40,000 chest x-rays to categorize a tube into 11 classes. Now these samples were multi-labeled as each x-ray image can have 1 or more classes due to positioning of multiple tubes in the chest.

While throughout the challenge, Google's masterpiece EfficientNets showed very good results, I tried to improvise on the existing Tensorflow's InceptionV3 architecture. During training, the Inception model was achieving very high training AUC score (>0.99) withing 15-20 epochs but it's score on validation set was limited to 0.92. 

Implementing noisy layers (Gaussian Noise) with std. dev of 1 worked out to be best which bettered the AUC score of the baseline Inception model (~0.95). Also adding L1 and L2 regularizers to the model layers boosted the score upto 0.94. 

Overall it was observed that improvising on the Inception network proved to work well wrt the EfficientNet B6 and B7 models which achieved slightly better scores but with much higher training time (>1.5x time) which matters when running the model for high no. of epochs.
