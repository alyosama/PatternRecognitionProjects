%% Initialization
clear ; close all; clc;

%% ==================== Part 1: Loading Data from Iris .txt ===================	
data=load('iris.txt');
X=data';
[l,N]=size(X);
%%%%%%%% kmean
m=2;
theta_ini=rand(l,m);
[theta,bel,J]=k_means(X,theta_ini);

fprintf('Iris k=%d objective function value: %d\n',m,fitnessclustsse(theta(:),data));

%% ==================== Part 2: Loading Data from ruspini.txt ===================	
data=load('ruspini.txt');
X=data';
[l,N]=size(X);
% Plot of the data set
figure (1), plot(X(1,:),X(2,:),'.')
figure(1), axis equal

%%%%%%%% kmean
m=4;
theta_ini=rand(l,m);
[theta,bel,J]=k_means(X,theta_ini);

X3=X;
figure(2), hold on
figure(2), plot(X3(1,bel==1),X3(2,bel==1),'r.',...
X3(1,bel==2),X3(2,bel==2),'g*',X3(1,bel==3),X3(2,bel==3),'bo',...
X3(1,bel==4),X3(2,bel==4),'cx',X3(1,bel==5),X3(2,bel==5),'md',...
X3(1,bel==6),X3(2,bel==6),'yp',X3(1,bel==7),X3(2,bel==7),'ks')
figure(2), plot(theta(1,:),theta(2,:),'k+')
figure(2), axis equal

fprintf('Ruspini k=%d objective function value: %d\n',m,fitnessclustsse(theta(:),data));

%% ==================== Part 3: Loading Data from german_postal.txt ===================	
data=load('german_postal.txt');
X=data';
[l,N]=size(X);
% Plot of the data set
figure (3), scatter3(X(1,:),X(2,:),X(3,:));



%%%%%%%% kmean
m=2;
theta_ini=rand(l,m);
[theta,bel,J]=k_means(X,theta_ini);

%X3=X;
%figure(4), hold on
%figure(4), scatter3(X3(1,bel==1),X3(2,bel==1),X3(3,bel==1),'r.',...
%X3(1,bel==2),X3(2,bel==2),X3(3,bel==2),'g*',X3(1,bel==3),X3(2,bel==3),X3(3,bel==3),'bo',...
%X3(1,bel==4),X3(2,bel==4),X3(3,bel==4),'cx',X3(1,bel==5),X3(2,bel==5),X3(3,bel==5),'md',...
%X3(1,bel==6),X3(2,bel==6),X3(3,bel==6),'yp',X3(1,bel==7),X3(2,bel==7),X3(3,bel==7),'ks')
%figure(4), scatter3(theta(1,:),theta(2,:),theta(3,:),'k+')
fprintf('German Postal k=%d objective function value: %d\n',m,fitnessclustsse(theta(:),data));