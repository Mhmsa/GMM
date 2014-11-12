function [ Confusion_Mtx ] = GMM_Classifier(dataset,kfold,mixtures,Cov)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
data1=dataset(dataset(:,end)==1,:);l1=length(data1);
data2=dataset(dataset(:,end)==2,:);l2=length(data2);
data3=dataset(dataset(:,end)==3,:);l3=length(data3);
Confusion_Mtx=zeros(3);
for ij=1:kfold
    ij
    tic
    %% partitioning data for cross validation
    ind1=false(l1,1);ind2=false(l2,1);ind3=false(l3,1);
    ind1(1+round((ij-1)*l1/kfold):round(ij*l1/kfold))=1;
    ind2(1+round((ij-1)*l2/kfold):round(ij*l2/kfold))=1;
    ind3(1+round((ij-1)*l3/kfold):round(ij*l3/kfold))=1;
    test=[data1(ind1,1:end-1);data2(ind2,1:end-1);data3(ind3,1:end-1)];test_label=[data1(ind1,end);data2(ind2,end);data3(ind3,end)];
    %% This classifier will compose of 3 pairwise classifiers
    GMM_Models{1}=fitgmdist(data1(~ind1,1:end-1),mixtures,'CovType',Cov,'Regularize',0.001);
    GMM_Models{2}=fitgmdist(data2(~ind2,1:end-1),mixtures,'CovType',Cov,'Regularize',0.001);
    GMM_Models{3}=fitgmdist(data3(~ind3,1:end-1),mixtures,'CovType',Cov,'Regularize',0.001);
    %% testing
    Label=[];
    for ii=1:length(test)
   [p1,nlogl1]=posterior(GMM_Models{1},test);
   [p2,nlogl2]=posterior(GMM_Models{2},test);
   [p3,nlogl3]=posterior(GMM_Models{3},test);
   [~,lbl]=max([nlogl1 nlogl2 nlogl3]);
   Label(ii)=lbl;
    end
    test_time(GMM_Models,test,'GMM')
    cm=confusionmat(test_label,Label);
    Confusion_Mtx=Confusion_Mtx+cm;
    toc
end
end

