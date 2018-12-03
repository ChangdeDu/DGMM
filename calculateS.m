
function S=calculateS(k,t)

load data
options = [];
% options.Metric = 'Cosine';
% options.WeightMode = 'Cosine';
options.Metric = 'Euclidean';
options.WeightMode = 'Binary';
options.WeightMode = 'HeatKernel';
options.NeighborMode = 'KNN';
options.k = k;  % nearest neighbor
options.t = t;
ntrn=size(Y_train,1);
ntest=size(Y_test,1);
for i=1:ntest
    fmri=[Y_train;Y_test(i,:)];
    temp = constructW(fmri,options);
    temp2(:,i)=temp(1:end-1,end);
end 
temp2=full(temp2);

S=zeros(ntrn,ntest);
[dump, idx] = sort(-temp2,1); % sort each row 
for i=1:ntest
selectidx=idx(2:k+1,i);
S(selectidx,i)=-dump(2:k+1,i);
end
end