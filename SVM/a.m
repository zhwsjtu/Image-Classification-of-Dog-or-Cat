tic
dir=('C:\Users\ding\Desktop\picture\train2000');
testdir=('C:\Users\ding\Desktop\picture\test');
trainingSet = imageSet(dir,'recursive');
testSet = imageSet(testdir,'recursive');

[trainingFeatures,trainingLabels,testFeatures,testLabels]=extractFeature(trainingSet,testSet);

% 
%ѵ��һ��svm������
%fitcecoc ʹ��1��1�ķ���
classifier = fitcecoc(trainingFeatures, trainingLabels);
save classifier.mat classifier;

predictedLabels = predict(classifier, testFeatures);

% ����������
% ʹ��û�б�ǩ��ͼ�����ݽ��в��ԣ�����һ�����������������Ч��
confMat=confusionmat(testLabels, predictedLabels)
accuracy=(confMat(1,1)/sum(confMat(1,:))+confMat(2,2)/sum(confMat(2,:)))/2
toc
