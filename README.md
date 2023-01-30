# Reduced Volume of diabetic Pancreatic Islets in Rodents Detected by Synchrotron X-ray Phase-contrast Microtomography and Deep Learning Network

This is the code of AA-Net proposed in the paper **"Reduced volume of diabetic pancreatic islets in rodents detected by synchrotron X-ray phase-contrast microtomography and deep learning network"** published in Heliyon, 2023. 

## About AA-Net
AA-Net is a deep-learning based method to segment islet images and, therefore, help visualize the islets *in situ* in diabetic rodents by the synchrotron radiation X-ray phase-contrast microtomography (SRμCT) at the ID17 station of the European Synchrotron Radiation Facility. 

AA-Net takes the traditional encoder-decoder structure where AAM is adopted to enhance the bottleneck features and the twin-block based encoder is included to improve feature abstraction performance.
![network](AA-Net.pdf)

## How to Use
The code is based on: Python 3.6, and pytorch 1.5.0, which includes both the training and testing parts. Please check Requirements.txt for the details on how to setup the code.
- Run /Project_AANet/test.py for testing.
- Run /Projext_AANet/main.py for training.
- Run Certainy_analysis/uncertainy_map.py and  Certainy_analysis/confindence.py for Certainty analysis. 
- Run size.py for estimating the size of islet. 

## Paper Information
Please check the following paper for more details:

Qingqing Guo, Abdulla AlKendi, Xiaoping Jiang, Alberto Mittone, Linbo Wang, Emanuel Larsson, Alberto Bravin, Erik Renström, Xianyong Fang, Enming Zhang. *Reduced volume of diabetic pancreatic islets in rodents detected by synchrotron X-ray phase-contrast microtomography and deep learning network*, Heliyon, vol. 9, no.2, E13081, 2023. DOI:https://doi.org/10.1016/j.heliyon.2023.e13081
