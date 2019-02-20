function [Patterns,Targets,V_Targets]=LoadData(dataset,scale)
%-------- Load a dataset / Define Patterns and Target Data --------

dataset=[char(dataset),'.txt'];
RawData=load(strcat('Datasets\',dataset));
Patterns=RawData(:,1:end-1);
Targets=RawData(:,end);


% N_class=length(unique(Targets));
% if ~isempty(N_selected_class)  && N_selected_class~=0 %generate subset of datasets
%   tmp=randperm(N_class);
%   selected_classes=tmp(1:N_selected_class);
% 
%   selected_samples=[]; Target=[];
%   for ii=1:N_selected_class
%     class=selected_classes(ii);
%     b=find(Targets==class);
%     selected_samples=[selected_samples;b];
%     Target=[Target;repmat(ii,length(b),1)];
%   end
%   Patterns=Patterns(selected_samples,:);
%   Targets=Target;
% end

if min(Targets)==0, Targets=Targets+1; end
N_Samples=size(Patterns,1); N_class=max(Targets);

% Create Vector Target (V_Target)
% Vector Target are used in some classifiers such as MLP NN; V_Target is like
% Target;  For example if the Target= 2 and we have 3 classes;
% the V_Target=[0 1 0]
V_Targets=zeros(N_class,N_Samples);
temp=0:N_class:(N_Samples-1)*N_class;
V_Targets(Targets'+temp)=1;

%[M,N]=size(Patterns);
if scale==1,
  %Patterns=(Patterns - repmat(min(Patterns,[],1),M,1))* spdiags(1./(max(Patterns,[],1)-min(Patterns,[],1))',0,N,N);
  Patterns=mapminmax(Patterns'); Patterns=Patterns';
end;
Patterns=removeconstantrows(Patterns'); Patterns=Patterns';
end