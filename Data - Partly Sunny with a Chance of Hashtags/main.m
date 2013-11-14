clear all;

%% importing data
[trainData_f trainData_s] = xlsread('train.csv');
numTrainData = size(trainData_f, 1);
trainData_title = trainData_s(1,:);
trainData_s(1,:) = [];

[testData_f testData_s] = xlsread('test.csv');
numTestData = size(testData_f, 1);
trainData_title = testData_s(1,:);
testData_s(1,:) = [];


%% search for most frequent keywords
