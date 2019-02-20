function [TrainPatterns,TrainTargets,Train_V_Targets,TestPatterns,TestTargets,Test_V_Targets]=...
  DataPartitioning(Patterns,Target,V_Targets,Pmethod,fold_N,N_folds,TrainPercent)

N=size(Patterns,1); %N=Numeber of Samples;
switch (Pmethod)
  case {'Resubs'}
    TrainPatterns=Patterns;
    TrainTargets=Target;
    Train_V_Targets=V_Targets;

    TestPatterns=Patterns;
    TestTargets=Target;
    Test_V_Targets=V_Targets;
    
  case {'Holdout'}
    Train_point_cut=floor(TrainPercent*N);

    TrainPatterns=Patterns(1:Train_point_cut,:);
    TrainTargets=Target(1:Train_point_cut);
    Train_V_Targets=V_Targets(:,1:Train_point_cut);

    TestPatterns=Patterns(Train_point_cut+1:end,:);
    TestTargets=Target(Train_point_cut+1:end);
    Test_V_Targets=V_Targets(:,Train_point_cut+1:end);

  case {'Kfold'}
    %N=size(Patterns,2); %N=Numeber of Samples; %N_folds=Number of partitions
    n1=ceil(N/N_folds); % find the size of the testing data sets
    last=N-(N_folds-1)*n1;% find the size of the last set (if any)
    if last==0,
      last=n1; % N_folds divides N, all pieces are the same
    end
    if last<n1/2, % if the last piece is smaller than
      % half of the size of the others, % then issue a warning
      fprintf('%s\n','Warning: imbalanced testing sets')
    end
    v=[]; % construct indicator-labels for the N_folds subsets
    for i=1:N_folds-1;
      v=[v;ones(n1,1)*i];
    end
    v=[v;ones(last,1)*N_folds];

    L=v==fold_N;

    %size(Patterns,2)
    %length(L)
    TrainPatterns=Patterns(~L,:); % training data
    TrainTargets=Target(~L,:); % training labels
    Train_V_Targets=V_Targets(:,~L);

    TestPatterns=Patterns(L,:); % test data
    TestTargets=Target(L,:);
    Test_V_Targets=V_Targets(:,L);
end
