function [TrainedCLF,CLF_Train_output]=...
  TrainClassifier_Complete(TrainIn,TrainTarget,Train_V_Targets,ClassifierType,Params)

switch (ClassifierType)

  case {1} % Multi layer perceptron  (MLP)
    CLF = newff(TrainIn',Train_V_Targets,Params.MLP.N_nodes,{Params.MLP.Fnc},'trainlm');
    CLF.trainParam.showWindow = false;
    CLF.trainParam.showCommandLine = false;
    TrainedCLF=train(CLF,TrainIn',Train_V_Targets);
    net_TrainOut=sim(TrainedCLF,TrainIn');

    % Produce 3 types of label output
    DP=mapminmax(net_TrainOut',0,1); % measurment level output (Decision profile)
    [temp,class]=max(net_TrainOut); %  the abstract level output
    [temp,Ranked_class]=sort(net_TrainOut,'descend'); % the rank level output

    [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,net_TrainOut);
    accuracy=1-TrainError;
  case {2}  %KNN
    % Do Nothing!
    TrainedCLF=[];
    class=[];
    Ranked_class=[];
    DP=[];
    ConfusionMatrix=[];
    CM_per=[];

  case {3}  %LIB SVM
    Params.SVM_Params=[Params.SVM_Params ' -b 1'];
    TrainedCLF = svmtrain2(TrainTarget, TrainIn,Params.SVM_Params);
    [class,accuracy,prob_estimates]=svmpredict2(TrainTarget,TrainIn,TrainedCLF,'-b 1');
    DP(:,TrainedCLF.Label)=prob_estimates;
 
    [tmp,Ranked_class]=sort(DP,2,'descend'); % the rank level output
    [Train_error,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');

end

%%%% Train Outputs
CLF_Train_output = struct('Abstract_level_output'     , class,...
  'Rank_level_output'         , Ranked_class, ...
  'Measurment_level_output'   , DP, ...
  'Train_Recognition_rate'      , accuracy, ...
  'ConfusionMatrix'           , ConfusionMatrix,...
  'ConfusionMatri_Percentage' , CM_per);
end