# Ensemble Classification
This toolbox provides some combination methods to fuse an ensemble of classifiers. These methods include: 
- Majority Vote
- Maximum (MAX)
- Sum (SUM)
- Minimum (MIN)
- Average (AVR)
- Product (PRO)
- Bayes
- Decision Template
- Dempster-Sahfer fusion method
- Behavior Knowledge Space (BKS)

For better understanding of the toolbox, reading the following references is strongly recommended 
[1] R. Polikar, "Ensemble based systems in decision making," IEEE Circuits and Systems Magazine, vol. 6, pp. 21-45, 2006. 
[2] L. I. Kuncheva, Combining Pattern Classifiers: Methods and Algorithms. New York, NY: Wiley, 2004.
 
## How to use 
%% You should run the Main_Ensemble.m file

### Dataset 
*** You can use any of the included datasets (taken from UCI ) or your own dataset; 
*** To use your dataset, you should put the dataset in the Datasets folder 
*** The data should be stored in one text file. You can save data in "Excel > Save as Text (Tab Delimited)" 
*** The dataset should be in a Matrix format; 
*** number of rows is equal to the number of samples 
*** Number of columns is equal to the number of features + 1 
*** The last column is the Target (Label) column of samples

## Citation
Please cite at least one of the following papers if you like to use this toolbox: 
+ M.A. Bagheri, Gh. Montazer, and E. Kabir, “A Subspace Approach to Error-Correcting Output Coding”, 
Pattern Recognition Letters, vol. 34, pp. 176–184, 2013 
+ M.A. Bagheri, Q. Gao and S. Escalera, “Rough Set Subspace Error Correcting Output Codes”, 
in Proc. IEEE International Conf on Data Mining, Brussels, Belgium, 2012
