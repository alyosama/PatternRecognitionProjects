%% Initialization
clear ; close all; clc
%% ==================== Part 1: Loading Data from test.data ===================	
data=load('ruspini.txt');
X=data';
[l,N]=size(X);
% Plot of the data set
figure (1), plot(X(1,:),X(2,:),'.')
figure(1), axis equal

%%%%%%%% BSAS
q=15; % maximum number of clusters
theta=2.5; % dissimilarity threshold
order=randperm(N);
[bel, repre]=BSAS(X,theta,q,order);

X1=X;
figure(10), hold on
figure(10), plot(X1(1,bel==1),X1(2,bel==1),'r.',...
X1(1,bel==2),X1(2,bel==2),'g*',X1(1,bel==3),X1(2,bel==3),'bo',...
X1(1,bel==4),X1(2,bel==4),'cx',X1(1,bel==5),X1(2,bel==5),'md',...
X1(1,bel==6),X1(2,bel==6),'yp',X1(1,bel==7),X1(2,bel==7),'ks')
figure(10), plot(repre(1,:),repre(2,:),'k+')

%%%%%%%% MSAS
X1=X;
[l,N]=size(X1);
% Determination of the distance matrix
dista=zeros(N,N);
for i=1:N
    for j=i+1:N
        dista(i,j)=sqrt(sum((X1(:,i)-X1(:,j)).^2));
        dista(j,i)=dista(i,j);
    end
end
true_maxi=max(max(dista));
true_mini=min(dista(~logical(eye(N))));

%Determine min, max, and s by typing
meani=(true_mini+true_maxi)/2;
theta_min=.25*meani;
theta_max=1.75*meani;
n_theta=50;
s=(theta_max-theta_min)/(n_theta-1);

% Run BSAS N times
q=N;
n_times=10;
m_tot=[];
for theta=theta_min:s:theta_max
    list_m=zeros(1,q);
    for stat=1:n_times %for each value of Theta BSAS runs n_times times
        order=randperm(N);
        [bel, m]=BSAS(X1,theta,q,order);
        list_m(size(m,2))=list_m(size(m,2))+1;
    end
    [q1,m_size]=max(list_m);
    m_tot=[m_tot m_size];
end

%Plot m versus theta
theta_tot=theta_min:s:theta_max;
figure(2), plot(theta_tot,m_tot)

% Determining the number of clusters
m_best=0;
theta_best=0;
siz=0;

for i=1:length(m_tot)
    if(m_tot(i)~=1) %Excluding the single-cluster clustering
        t=m_tot-m_tot(i);
        siz_temp=sum(t==0);
        if(siz<siz_temp)
            siz=siz_temp;
            theta_best=sum(theta_tot.*(t==0))/sum(t==0);
            m_best=m_tot(i);
        end
    end
end

%Check for the single-cluster clustering
if(sum(m_tot==m_best)<.1*n_theta)
    m_best=1;
    theta_best=sum(theta_tot.*(m_tot==1))/sum(m_tot==1);
end

%Run the BSAS algorithm for Best Theta
order=randperm(N);
[bel, repre]=BSAS(X1,theta_best,q,order);

figure(11), hold on
figure(11), plot(X1(1,bel==1),X1(2,bel==1),'r.',...
X1(1,bel==2),X1(2,bel==2),'g*',X1(1,bel==3),X1(2,bel==3),'bo',...
X1(1,bel==4),X1(2,bel==4),'cx',X1(1,bel==5),X1(2,bel==5),'md',...
X1(1,bel==6),X1(2,bel==6),'yp',X1(1,bel==7),X1(2,bel==7),'ks')
figure(11), plot(repre(1,:),repre(2,:),'k+')

%Compare
[bel,new_repre]=reassign(X1,repre,order);

