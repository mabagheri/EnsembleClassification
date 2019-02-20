%%  Introduction
% This toolbox provides some useful methods for combinition of ensemble of classifiers
clear; clc; close all

%% Identify dataset, ensemble type, base classifiers, and other parameteres
dataset='Glass';  % the following UCI datasets are also included in the package
% Abalone Balance Car Ecoli Glass iris Vehicle Wine Yeast Zoo, ...

%--- General Settings
N_runs=3;                           % Number of runs (replication)
Reorder=1;                          % change order of samples in each run or not? (1:Yes, 0:No); 
scale=1;                              % scale the data to be in [0 1]? (1:Yes, 0:No)

%--- Setting of Ensemble Classifiers
N_classifiers=3;                   % Number of base classifiers (ensemble size)
ClassifierTypes= [1 1 1] ;   % Identify base classifiers: 1=MLP 2=KNN 3=LIBSVM
Params.MLP.Fnc='tansig';     % Function (purelin, tansig, hardlim,logsig)
Params.MLP.N_nodes=10;
Params.KNN.K=1;
Params.KNN.distance='euclidean';           %(euclidean,cityblock,cosine,correlation)
Params.SVM_Params='-t 2 -c 100 -g 10  -q';          % -t= 0:linear; 1:polynomial; 2:RBF 3:sigmoid

%%% Validation Parameters
Pmethod='Kfold'; N_folds=5;  % Partitioning (evaluation) method is either "Holdout" or "Kfold"
if strcmp(Pmethod,'Holdout'), N_folds=1; end
TrainPercent=0.5;  % Percent of train samples (If P_Method is Holdout)

CombinitionMethods=[1,2,3,4,5,6,7,8,9];  % ***Note: Many combination methods are not suitable for KNN classifiers. 
% 1:Majority voting  2:Maximum  3:SUM  4:Min 5:Average 6:Product 7:Bayes 8:Decision Template  9:Dempster-Shafer
N_CombinitionMethods=length(CombinitionMethods);

FS_Type='All';                      % Feature Selection: All | Ranodm ...

  
%% Do some error checking
if length(ClassifierTypes) ~= N_classifiers
  error ('ClassifierTypes should be equal to number of classifiers');
end

%% Start_Classification Module
%--- Load Data
[Patterns,Targets,V_Targets]=LoadData(dataset,scale);
N_Samples=size(Patterns,1); N_features=size(Patterns,2);N_class=size(V_Targets,1);

Accuracy_Ensemble=zeros(N_runs,N_CombinitionMethods);
Accuracy_SingleClassifiers=zeros(N_runs,N_classifiers);

%--- Main Module
for run=1:N_runs %number of replications
  if Reorder==1, y=randperm(N_Samples); else y=(1:N_Samples); end
  Patterns=Patterns(y,:);Targets=Targets(y);V_Targets=V_Targets(:,y);
  
  Accuracy_fold_Ensemble=zeros(N_folds,N_CombinitionMethods);
  Accuracy_fold_SingleClassifiers=zeros(N_folds, N_classifiers);

  for fold_N = 1:N_folds
    [TrainPatterns,TrainTargets,Train_V_Targets,TestPatterns,TestTargets,Test_V_Targets]=...
      DataPartitioning(Patterns,Targets,V_Targets,Pmethod,fold_N,N_folds,TrainPercent);
    N_test=length(TestTargets);
    N_train=length(TrainTargets);
    
    %-- Create_Ensemble; %
    FSS  % Feature Subset Selection 

    for ii=1:N_classifiers
      Features=FeatureSubsets(ii,:);
      TrainPatts=TrainPatterns(:,Features);
      TestPatts=TestPatterns(:,Features);

      ClassifierType=ClassifierTypes(ii);

      [TrainedCLF,CLF_Train_output(ii)]=...
        TrainClassifier_Complete(TrainPatts,TrainTargets,Train_V_Targets,ClassifierType,Params);

      [Predicted_class,CLF_Test_output(ii)]=...
        ClassifyTestSamples_Complete(TrainedCLF,TrainPatts, TrainTargets, TestPatts,TestTargets,ClassifierType,Params); 

    Accuracy_fold_SingleClassifiers(fold_N,ii)=sum(Predicted_class==TestTargets)/N_test;

    end %for ii
    
    %%% Combine Trained Classifiers using different Combinition methods
    for C=1:length(CombinitionMethods);
      CombinitionMethod=CombinitionMethods(C);
      Ensemble_decisions=CombineCLFs(CombinitionMethod,...
        CLF_Train_output,CLF_Test_output,N_classifiers,N_test,N_train,N_class,TrainTargets);
      Accuracy_fold_Ensemble(fold_N,C)=sum(Ensemble_decisions'==TestTargets)/N_test;
    end
 
  end %for K'th fold

  Accuracy_Ensemble(run,:)= mean(Accuracy_fold_Ensemble,1);      %N_runs * N_combinationMethods 
  Accuracy_SingleClassifiers(run,:)= mean(Accuracy_fold_SingleClassifiers,1);
end   % run loop

clc;
disp('The average accuracy of Single Classifiers:');
Avg_Accuracy_SingleClassifiers=mean(Accuracy_SingleClassifiers,1)

disp('The average accuracy of Ensemble System with different Combination Methods is:');
Avg_Accuracy_Ensemble=mean(Accuracy_Ensemble,1)
