# Reduced Volume of diabetic Pancreatic Islets in Rodents Detected by Synchrotron X-ray Phase-contrast Microtomography and Deep Learning Network

This is the code on the AA-Net proposed in the paper **"Reduced volume of diabetic pancreatic islets in rodents detected by synchrotron X-ray phase-contrast microtomography and deep learning network"** published in Heliyon, 2023. 

In short, AA-Net is a deep-learning based method to segment islet images and help visualizing the islets in situ in diabetic rodents by the synchrotron radiation X-ray phase-contrast microtomography (SRμCT) at the ID17 station of the European Synchrotron Radiation Facility. 

- The code is based on: Python 3.6, and pytorch 1.5.0, which includes both the training and testing parts. Please check Requirements.txt for the details on how to setup the code.
- Run /Project_AANet/test.py for testing.
- Run /Projext_AANet/main.py for training.
- Run Certainy_analysis/uncertainy_map.py and  Certainy_analysis/confindence.py for Certainty analysis. 
- Run size.py for estimating the size of islet. 


Please check our paper for more details:
Qingqing Guo, Abdulla AlKendi, Xiaoping Jiang, Alberto Mittone, Linbo Wang, Emanuel Larsson, Alberto Bravin, Erik Renström, Xianyong Fang, Enming Zhang. Reduced volume of diabetic pancreatic islets in rodents detected by synchrotron X-ray phase-contrast microtomography and deep learning network, Heliyon, 2023. DOI:https://doi.org/10.1016/j.heliyon.2023.e13081

Thanks!
