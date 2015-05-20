
%Script to demonstrate the application of CMA-ES on clustering problems

%Use a fixed seed for the random number generator if you want reproducable
%results - uncomment next line
%rng(124);

%Number of clusters
%kvals = 9;
kvals = 2:4;

%Initialize variables to store (best-so-far) results

mres_cma = zeros(size(kvals,2),1);
sres_cma = zeros(size(kvals,2),1);
bres_cma = zeros(size(kvals,2),1);
mresit_cma = zeros(size(kvals,2),1);
sresit_cma = zeros(size(kvals,2),1);

mres_cma2 = zeros(size(kvals,2),1);
sres_cma2 = zeros(size(kvals,2),1);
bres_cma2 = zeros(size(kvals,2),1);
mresit_cma2 = zeros(size(kvals,2),1);
sresit_cma2 = zeros(size(kvals,2),1);

%Number of trials/restarts to run for each algorithm
trials = 1;

%Get the data
load german_postal;

%The German towns/postal data (across all variables) is in the range
%[24.49,1306024]^3.  This serves as a reasonable box-constraint on the
%feasible search region for optimization.
germanrangemin = 24.49;
germanrange = 1306024 - 24.49;

%-----------------     
%Main loop

kcnt = 0;
for k=kvals
    kcnt = kcnt + 1;
    %Problem dimensionality
    dim = k*size(data,2);
    disp(['k = ',num2str(k),', dim = ', num2str(dim)])
    
    FMIN = zeros(trials,1);
    cmaits = zeros(trials,1);
    
    %Default-CMA-ES
    disp('Now running Default-CMA-ES:')
    for i=1:trials
        [XMIN,FMIN(i),ceval] = cmaes('fitnessclustsse', ((germanrange/2)*ones(dim,1))+germanrangemin, (germanrange/3), [], data);
        %[XMIN,FMIN(i)] = cmaes('fitnessclustd', 0.5*ones(12,1), 0.3);
        cmaits(i) = ceval;
    end
    mres_cma(kcnt) = mean(FMIN);
    sres_cma(kcnt) = std(FMIN);
    bres_cma(kcnt) = min(FMIN);
    mresit_cma(kcnt) = mean(cmaits);
    sresit_cma(kcnt) = std(cmaits);
    
    %Try a larger population CMA-ES
    opts.PopSize=500;
    
    disp(['Now running (',num2str(opts.PopSize/2), ',', num2str(opts.PopSize), ')-CMA-ES:'])
    for i=1:trials
        [XMIN,FMIN(i),ceval] = cmaes('fitnessclustsse', ((germanrange/2)*ones(dim,1))+germanrangemin, (germanrange/3), opts, data);
        %[XMIN,FMIN(i)] = cmaes('fitnessclustd', 0.5*ones(12,1), 0.3);
        cmaits(i) = ceval;
    end
    mres_cma2(kcnt) = mean(FMIN);
    sres_cma2(kcnt) = std(FMIN);
    bres_cma2(kcnt) = min(FMIN);
    mresit_cma2(kcnt) = mean(cmaits);
    sresit_cma2(kcnt) = std(cmaits); 

end
