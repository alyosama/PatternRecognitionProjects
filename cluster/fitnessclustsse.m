
% Function fitnessclustsse
%
% Input: 
%   centers - a (kp x 1) candidate solution vector representing the positions of
%       k cluster centers.  That is, the dimensionality of the optimization
%       problem is kp.
%   data - an (n x p) dataset to be clustered (n data points of dimensionality p)

function [fval] = fitnessclustsse(centers,data)

%Need to convert "centers" from an kpx1 candidate solution vector to a 
% center coordinates - k x p

centers = reshape(centers,size(data,2),(size(centers,1)/size(data,2)));
centers = centers';

%Pairwise distances of all data points
D = pdist2(centers,data);

%Find which cluster center each data point is closest(owns) to (ind)
%---Note: this breaks if k=1, so don't do that!
[~,ind]=min(D);

%Calculate error value for current clustering
errorclust = 0;
for i=1:size(centers,1)
    errorclust = errorclust + sum( pdist2(centers(i,:),data(ind==i,:)).^2 );
end

fval = errorclust;