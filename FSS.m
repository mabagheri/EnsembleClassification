%Feature Subsets Selection (FSS)
switch (FS_Type)
  case {'All'}
    FeatureSubsets=repmat(1:N_features,N_classifiers,1);

  case {'Random'}
    featureSubsets=round(rand(N_classifiers,N_features));
    FeatureSubsets=featureSubsets==1;
    for n=1:N_classifiers
      %FeatureSubsets(n,:)=find(featureubsets(n,:));
      if FeatureSubsets(n,:)==0,
        FeatureSubsets(n,:)=round(rand(1,N_features));
      end
    end

  case {'GA'}
    [BestCost,FeatureSubsets]=GA_FST(TrainInput,TrainTarget,TrainGroup,...
      N_classifiers,FeatureSubsetEvaluator,Param1,Param2);

  case {'GA_V1'}
    [FeatureSubsets,SubsetsAccuracy,EnsembleAccuracy]=...
      GA_V1(TrainInput,TrainTarget,TrainGroup,N_classifiers,mutrate,popsize,FSE);

  case {'GA_V3'}
    [BestCost,FeatureSubsets]=GA_V3(TrainInput,TrainTarget,TrainGroup,N_classifiers);

  case {'AB'}
    [FeatureSubsets,SubsetsAccuracy]=...
      AB(TrainInput,TrainTarget,TrainGroup,N_classifiers,popsize,FSE,ValPercent);

  case {'MAB'}
    [FeatureSubsets,SubsetsAccuracy]= MAB(TrainInput,TrainTarget,TrainGroup,...
      N_classifiers,popsize,FSE,Alpha,ValPercent,DIV_Measure);

  case {'PSO'}
    [BestCost,FeatureSubsets]= BPSO_FS(TrainInput,TrainTarget,TrainGroup,...
      N_classifiers,FeatureSubsetEvaluator,Param1,Param2);

  case {'Hclustering'}
    No_of_clusters=round(N_features/1.6);
    %         Input2=std(Input,0,2);
    FeatureSubsets=Hclustering(Input,No_of_clusters,N_classifiers,N_features);
    FeatureSubsets=logical(FeatureSubsets);
end