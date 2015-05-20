%% Pattern Recognition Class 
%% Statlog (Heart) Data Set 
%% Author : Aly Osama
%% Email : alyosamah@gmail.com
%% Neural Networks
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


% %% ==================== Part 4: Neural Networks ==================================
k=30;
code=1;
iter=1000;
lr=0.01;
par_vec=[lr 0 0 0 0];

[net,tr]=NN_training(X_train,y_train,k,code,iter,par_vec);
% net = patternnet(k);
% net=init(net);
%net.trainParam.epochs=iter;
%net.trainParam.lr=par_vec(1);
% [net,tr] = train(net,X_train,y_train);

pe_train=NN_evaluation(net,X_train,y_train);
pe_test=NN_evaluation(net,X_test,y_test);
fprintf('NN Train performance %.2f\n',pe_train*100);
fprintf('NN Test performance %.2f\n',pe_test*100);
figure(1), plot(tr.perf)

% %% ==================== Part 4: Adaptive NN ==================================
iter=1000;
code=3;
k=30;
lr=0.001;
lr_inc=1.05;
lr_dec=0.7;
max_perf_inc=1.04;
par_vec=[lr 0 lr_inc lr_dec max_perf_inc];

[net,tr]=NN_training(X_train,y_train,k,code,iter,par_vec);
%  net = patternnet(k);
%  net=init(net);
% net.trainParam.epochs=iter;
%  net.trainParam.lr=par_vec(1);
%  net.trainParam.lr_inc=par_vec(3);
%  net.trainParam.lr_dec=par_vec(4);
%  net.trainParam.max_perf_inc=par_vec(5);
% [net,tr] = train(net,X_train,y_train);

pe_train=NN_evaluation(net,X_train,y_train);
pe_test=NN_evaluation(net,X_test,y_test);
fprintf('NN Train performance %.2f\n',pe_train*100);
fprintf('NN Test performance %.2f\n',pe_test*100);
figure(2), plot(tr.perf)