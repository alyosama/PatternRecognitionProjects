%% Pattern Recognition Class 
%% Breast Cancer Data Set 
%% Author : Aly Osama
%% Email : alyosamah@gmail.com
%% Percepton Algorithm
%% ============================================================================

%% Initialization
clear ; close all; randn('seed',0); clc
%% ==================== Part 1: Loading Data from test.data ===================	
data=load('wdbc.data');
X_data=data(:,3:end);	
y_data=data(:,2);

%% for Percepton
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


% %% ==================== Part 4: Applying Percepton ==================================
rho=0.01;
w_ini=rand(m+1,1);

[w, iter, mis_clas] = perce([X_train ;ones(1,training_range)], y_train, w_ini, rho);

SSE_out=2*(w'*[X_test;ones(1,N-training_range)]>0)-1;
err_SSE=sum(SSE_out.*y_test<0)/(N-training_range);
fprintf('Percepton error is %.2f%% \n',err_SSE*100);

[w, iter, mis_clas] = perce_online([X_train ;ones(1,training_range)], y_train, w_ini, rho);

SSE_out=2*(w'*[X_test;ones(1,N-training_range)]>0)-1;
err_SSE=sum(SSE_out.*y_test<0)/(N-training_range);
fprintf('Online Percepton error is %.2f%% \n',err_SSE*100);
