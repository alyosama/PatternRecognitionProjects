%% Pattern Recognition Class 
%% Breast Cancer Wisconsin (Diagnostic) Data Set 
%% Author : Aly Osama
%% Email : alyosamah@gmail.com
%% ============================================================================

%% Initialization
clear ; close all; randn('seed',0); 
%% ==================== Part 1: Loading Data from test.data ===================	
data=load('wdbc.data');
X_data=data(:,3:end);	
y_data=data(:,2);
m_features=30;

%% ==================== Part 2: Estimating Two Probabilities ==================
N=length(y_data);
m_size=length(find(y_data==1));
b_size=length(find(y_data==2));
P=[m_size b_size]'./N;

%% ==================== Part 3: Random Data Split 75% to 25% 	===============
rndIdx=randperm(N);
training_range=floor(75*N/100);

X_train=X_data(1:training_range,:)';
y_train=y_data(1:training_range,:)';	

X_test=X_data(training_range+1:end,:)';
y_test=y_data(training_range+1:end,:)';

%% ==================== Part 4: Estimating Mean and Variance ==================

malignant_data=X_train(:,find(y_train==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(malignant_data);
benign_data=X_train(:,find(y_train==2));
[m2_hat, S2_hat]=Gaussian_ML_estimate(benign_data);

S_hat=cat(3,S1_hat,S2_hat);
m_hat=[m1_hat m2_hat];

%% ==================== Part 5: Applying Bayesian Classifier  =================
y_bayesian=bayes_classifier(m_hat,S_hat,P,X_test);

%% ==================== Part 6: Estimating Error of Bayes  ====================
err_bayesian = (1-length(find(y_test==y_bayesian))/length(y_test));
fprintf('Bayesian classifier y_test error is %.3f%% \n',err_bayesian*100);

%% ==================== Part 7: Using PCA  ====================================
m=30;
[eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X_data',m);
figure(1),plot([1:length(eigenval)],eigenval,'.r');
hold on;
title ('Ploting the eigenvals');
text (1,eigenval(1),'Max eigenval');
xlabel ('No eigenval');
ylabel ('eigenvalue');
hold off;

m=30;
[eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X_data',m);
original_data=y_data';

a=Y(:,original_data==1);
b=Y(:,original_data==2);
figure(2),subplot (2, 1, 1),plot(a(1,:),a(2,:),'xb');
hold on;
title ('Plot 2 dim after PCA');
plot(b(1,:),b(2,:),'or');
hold off;

%% ==================== Part 8: Classify using Bayes after PCA 	===============

X_data_pca=Y';

X_train=X_data_pca(1:training_range,:)';
y_train=y_data(1:training_range,:)';	

X_test=X_data_pca(training_range+1:end,:)';
y_test=y_data(training_range+1:end,:)';

malignant_data=X_train(:,find(y_train==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(malignant_data);
benign_data=X_train(:,find(y_train==2));
[m2_hat, S2_hat]=Gaussian_ML_estimate(benign_data);

S_hat=cat(3,S1_hat,S2_hat);
m_hat=[m1_hat m2_hat];

y_bayesian=bayes_classifier(m_hat,S_hat,P,X_test);

err_bayesian = (1-length(find(y_test==y_bayesian))/length(y_test));
fprintf('After PCA Bayesian classifier error is %.3f%% \n',err_bayesian*100);

%% ==================== Part 9: Using LDA  ====================================
X2=X_data';
y2=y_data';
mv_est(:,1)=mean(X2(:,y2==1)')';
mv_est(:,2)=mean(X2(:,y2==2)')';
[Sw,Sb,Sm]=scatter_mat(X2,y2);
w=inv(Sw)*(mv_est(:,1)-mv_est(:,2));

%Computation of the projections
t1=w'*X2(:,y2==1);
t2=w'*X2(:,y2==2);
%Plot of the projections
subplot (2, 1, 2),plot(t1,'xb');
hold on;
title ('Plot data after LDA'); 
plot(t2,'.r');
hold off;


%% ==================== Part 9: Classify using Bayes after LDA 	===============

X_data_lda=w'*X2;
X_data_lda=X_data_lda';

X_train=X_data_lda(1:training_range,:)';
y_train=y_data(1:training_range,:)';	

X_test=X_data_lda(training_range+1:end,:)';
y_test=y_data(training_range+1:end,:)';

malignant_data=X_train(:,find(y_train==1));
[m1_hat, S1_hat]=Gaussian_ML_estimate(malignant_data);
benign_data=X_train(:,find(y_train==2));
[m2_hat, S2_hat]=Gaussian_ML_estimate(benign_data);

S_hat=cat(3,S1_hat,S2_hat);
m_hat=[m1_hat m2_hat];

y_bayesian=bayes_classifier(m_hat,S_hat,P,X_test);

err_bayesian = (1-length(find(y_test==y_bayesian))/length(y_test));
fprintf('After LDA Bayesian classifier error is %.3f%% \n',err_bayesian*100);

%% ============================================================================
