%% Final Exam Problem 1
% Bryn Louise

% Load Data
X = load('Photos01.mat'); 
X = X.X; 
Xm = X - mean(X,2);  

%% 1)

% Compute SVD
[u,s,v] = svd(X, 'econ'); 
% Find Best basis in R4 
basis = [u(:,1), u(:,2), u(:,3), u(:,4)]; 

% Visualize Basis Vectors
figure(1) 
for j=1:4
subplot(2,2,j)
imagesc(reshape(basis(:,j),162,149));
axis equal; colormap(gray)
end

% Visualize Mean
figure(2) 
meanimage = mean(X,2); 
imagesc(reshape(meanimage, 162, 149)); 
axis equal; colormap(gray)

%Plot 26 photos
figure(3)
for j=1:26
subplot(5, 6, j)
imagesc(reshape(X(:,j), 162, 149));
axis equal; colormap(gray)
end
%% 2)

% Convert matrix into cell array
Xcell = cell(1,26);
for i = 1:26
    Xcell{i} = reshape(X(:, i), 162, 149); 
end


% Train an autoencoder with a hidden layer containing 4 neurons
hiddenSize = 4; 
autoenc = trainAutoencoder(Xcell, hiddenSize,...
    'L2WeightRegularization',0.004,...
    'SparsityRegularization', 4,...
    'SparsityProportion', 0.15); 

W=autoenc.EncoderWeights;

% Visualize rows of Weights
figure(4) 
for i = 1:4
    subplot(2,2,i)
    imagesc(reshape(W(i, :), 162, 149)); 
    axis equal; colormap(gray); 
end


%%




