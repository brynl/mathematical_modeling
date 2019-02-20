%% Breast Cancer Linear Neural Network Batch Training
% Bryn Louise

%Load Data
BreastData
X = double(X)'; 
%Preprocess data
m = mean(X, 2); 
s = std(X,0,2); 
Xm = (X -m) ./repmat(s,1,106); 

% Construct Xhat and calculate What
b = ones(1,106);
Xhat = [Xm
    b]; 

What = T/Xhat; 

% Find W and b 
W = What(:,1:9); 
b = What(:,10); 

plotconfusion(T, (W*Xm + b));