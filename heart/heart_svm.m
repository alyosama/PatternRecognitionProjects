%% Pattern Recognition Class 
%% Statlog (Heart) Data Set 
%% Author : Aly Osama
%% Email : alyosamah@gmail.com
%% Support Vector Machine
%% ============================================================================

%% Initialization
clear ; close all; randn('seed',0); clc
%% ==================== Part 1: Loading Data from test.data ===================	
data=load('heart.dat');
X_data=data(:,1:end-1);	
y_data=data(:,end);

%% for SVM
y_data(y_data==2)=-1;

%% ==================== Part 2: Estimating Two Probabilities ==================
[N,m]=size(X_data);
m_size=length(find(y_data==1));
b_size=length(find(y_data==-1));
P=[m_size b_size]'./N;

%% ==================== Part 3: Random Data Split 75% to 25% 	===============
rndIdx=randperm(N);
training_range=floor(75*N/100);

X_train=X_data(1:training_range,:)';
y_train=y_data(1:training_range,:)';	

X_test=X_data(training_range+1:end,:)';
y_test=y_data(training_range+1:end,:)';

% %% ==================== Part 4: Estimating Mean and Variance ==================
malignant_data=X_train(:,find(y_train==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(malignant_data);
benign_data=X_train(:,find(y_train==-1));
[m2_hat, S2_hat]=Gaussian_ML_estimate(benign_data);

S_hat=cat(3,S1_hat,S2_hat);
m_hat=[m1_hat m2_hat];
% %% ==================== Part 4: Applying SVM ==================================

kernel='linear';
kpar1=0;
kpar2=0;
C=2;
tol=0.001;
steps=100000;
eps=10^(-10);
method=0;

[alpha, w0, w, evals, stp, glob] = SMO2(X_train', y_train', ...
kernel, kpar1, kpar2, C, tol, steps, eps, method);


err_svm=sum((2*(w*X_train-w0>0)-1).*y_train<0)/length(y_train);
err_svm_test=sum((2*(w*X_test-w0>0)-1).*y_test<0)/length(y_test);

sup_vec=sum(alpha>0);
marg=2/sqrt(sum(w.^2));
fprintf('\nLinear SVM\n');
fprintf('Number of Support vectors is %.0f \n',sup_vec);
fprintf('Margin is equals %.3f \n',marg);
fprintf('SVM classifier error is %.2f%% \n',err_svm*100);
fprintf('SVM classifier for test error is %.2f%% \n',err_svm_test*100);


%=================RBF SVM==================
kernel='rbf';
kpar1=0.1;
kpar2=0;
[alpha, w0, w, evals, stp, glob] = SMO2(X_train', y_train', ...
kernel, kpar1, kpar2, C, tol, steps, eps, method);


err_svm=sum((2*(w*X_train-w0>0)-1).*y_train<0)/length(y_train);
err_svm_test=sum((2*(w*X_test-w0>0)-1).*y_test<0)/length(y_test);

sup_vec=sum(alpha>0);
marg=2/sqrt(sum(w.^2));
fprintf('\nRadial Basis SVM\n');
fprintf('Number of Support vectors is %.0f \n',sup_vec);
fprintf('Margin is equals %.3f \n',marg);
fprintf('SVM classifier error is %.2f%% \n',err_svm*100);
fprintf('SVM classifier for test error is %.2f%% \n',err_svm_test*100);


%=================polynomial SVM==================
kernel='poly';
kpar1=1;
kpar2=3;
C=1;
[alpha, w0, w, evals, stp, glob] = SMO2(X_train', y_train', ...
kernel, kpar1, kpar2, C, tol, steps, eps, method);


err_svm=sum((2*(w*X_train-w0>0)-1).*y_train<0)/length(y_train);
err_svm_test=sum((2*(w*X_test-w0>0)-1).*y_test<0)/length(y_test);

sup_vec=sum(alpha>0);
marg=2/sqrt(sum(w.^2));
fprintf('\nPolynomial SVM\n');
fprintf('Number of Support vectors is %.0f \n',sup_vec);
fprintf('Margin is equals %.3f \n',marg);
fprintf('SVM classifier error is %.2f%% \n',err_svm*100);
fprintf('SVM classifier for test error is %.2f%% \n',err_svm_test*100);

