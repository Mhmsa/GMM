load('NormalizedFeaturesSet1');
load('NormalizedFeaturesSet2');
load('SubSetNormalizedFeaturesSet1');
load('SubSetNormalizedFeaturesSet2');
kfold=10;
mixtures=2:20;
Cov={'full','diagonal'};
Datasets={NormalizedFeaturesSet1,SubSetNormalizedFeaturesSet1,NormalizedFeaturesSet2,SubSetNormalizedFeaturesSet2};
% parpool(8)
for ii=1:4
    dataset=Datasets{ii};
    Confusion_Mtx=[];
    for jj=1:length(mixtures)
        for kk=1:2
            Confusion_Mtx(:,:,jj,kk)  = GMM_Classifier(dataset,kfold,mixtures(jj),Cov{kk});
        end
    end
end