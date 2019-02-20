%% Breast Cancer Linear Neural Network Online Training
% Bryn Louise

%Load Data
BreastData
X = double(X)'; 
%Preprocess data
m = mean(X, 2); 
s = std(X,0,2); 
Xm = (X -m) ./repmat(s,1,106); 

%Set Parameters
alpha = 0.0001; 
NumEpochs = 10000; 
%Train Data
[W, b, EpochErr] = WidHoff(Xm, T, alpha, NumEpochs); 

%Plot Error
NumEVec = [0:NumEpochs-1];
x1 = NumEVec;
y1 = EpochErr;
figure
plot(x1,y1) 

%Plot Confusion Matrix 
figure
plotconfusion(T, W*Xm + b); 