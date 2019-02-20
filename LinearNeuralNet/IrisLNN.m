%% Linear Neural Network Homework
% Bryn Louise

% load data
load('/home/louisebw/Downloads/IrisDataX.mat')
% define target classes
T = [ones(1,50), zeros(1,100)
    zeros(1,50), ones(1,50), zeros(1,50)
    zeros(1,100), ones(1,50)]; 
% define parameters
alpha = 0.01; 
NumEpochs = 150; 
% train
[W, b, EpochErr] = WidHoff(X, T, alpha, NumEpochs); 

NumEVec = [0:NumEpochs-1];
x1 = NumEVec;
y1 = EpochErr;
figure
plot(x1,y1) 

figure
plotconfusion(T, W*X + b); 

