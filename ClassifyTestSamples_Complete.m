function [Predicted_class,CLF_Test_output]= ... %[Predicted_class,Ranked_class,DP]=...
  ClassifyTestSamples_Complete(TrainedCLF,TrainIn, TrainTargets, TestIn,TestTargets,ClassifierType,Params)

switch (ClassifierType)
  case {1} %MLP
    net_Testout=sim(TrainedCLF,TestIn');
    [tmp,Predicted_class]=max(net_Testout);   
    Predicted_class=Predicted_class';

    % Produce 3 types of label output
    DP=mapminmax(net_Testout',0,1); % measurment level output (Decision profile)
    [tmp,Ranked_class]=sort(DP,2,'descend'); % the rank level output
    %[Train_error,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,net_TrainOut);	
    test_accuracy=sum(TestTargets==Predicted_class)/length(TestTargets);
  case {2} %KNN (1NN)
    [Predicted_class,DP] = myknnclassify(TestIn, TrainIn, TrainTargets,Params.KNN.K,Params.KNN.distance,'nearest');
    [temp,Ranked_class]=sort(DP,2,'descend'); % the rank level output

  case {3}  % LIB SVM
    [Predicted_class,test_accuracy,prob_est]=svmpredict2(TestTargets,TestIn,TrainedCLF,'-b 1');
    DP(:,TrainedCLF.Label)=prob_est;
    [tmp,Ranked_class]=sort(DP,2,'descend'); % the rank level output

end

CLF_Test_output = struct('Abstract_level_output'     , Predicted_class,...
                         'Rank_level_output'         , Ranked_class, ...
                         'Measurment_level_output'   , DP, ...
                         'CLF_test_accuracy'         , test_accuracy);
%                          'ConfusionMatrix'           , ConfusionMatrix,...
%                          'ConfusionMatri_Percentage' , CM_per);

end