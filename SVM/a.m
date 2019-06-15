tic
dir=('C:\Users\ding\Desktop\picture\train2000');
testdir=('C:\Users\ding\Desktop\picture\test');
trainingSet = imageSet(dir,'recursive');
testSet = imageSet(testdir,'recursive');

[trainingFeatures,trainingLabels,testFeatures,testLabels]=extractFeature(trainingSet,testSet);

% 
%训练一个svm分类器
%fitcecoc 使用1对1的方案
classifier = fitcecoc(trainingFeatures, trainingLabels);
save classifier.mat classifier;

predictedLabels = predict(classifier, testFeatures);

% 评估分类器
% 使用没有标签的图像数据进行测试，生成一个混淆矩阵表明分类效果
confMat=confusionmat(testLabels, predictedLabels)
accuracy=(confMat(1,1)/sum(confMat(1,:))+confMat(2,2)/sum(confMat(2,:)))/2
toc
