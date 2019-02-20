%% Final Exam Problem 3
% Bryn Louise

% Load Data 
X = load('mushrooms.mat'); 

% Define Targets
T = X.T; 

% Define Data 
X = X.X; 

% Divide data and targets into Training and Testing sets
[m,n] = size(X);

P = 0.70;
idx = randperm(n);
XTrain = X(:, idx(1:round(P*n))); 
XTest = X(:, idx(round(P*n)+1:end));

TTrain = T(:, idx(1:round(P*n)));  
TTest = T(:, idx(round(P*n)+1:end));

%%
% Find Weights Using Training Data

% Find 30 centers using Kmeans
[idx,C] = kmeans(XTrain',1000); 
centers = C'; 

% Find EDM
A = edm(XTrain', centers'); 

% Find Phi
Phi = rbf1(A,2,1); 

% Find Weights using SVD
[u,s,v] = svd(Phi, 'econ'); 
PhiDagger = v * inv(s) * u';  
W = TTrain * PhiDagger'; 

%%
% Find Error using Test Data
[idx,C] = kmeans(XTest',1000); 
centers = C'; 

A = edm(XTest', centers'); 

Phi = rbf1(A,2,1);  
Yout = W*Phi'; 

plotconfusion(TTest, Yout);

