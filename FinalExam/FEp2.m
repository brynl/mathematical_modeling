%% Final Exam Problem 2
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
% Train a feedforward Network

% Create a 22-10-2 network
net = feedforwardnet([10]);
% Train network
net = train(net, XTrain, TTrain); 
% Get outputs from test data
y = sim(net, XTest);
% Plot confusion matrix using testing set
plotconfusion(TTest, y); 
