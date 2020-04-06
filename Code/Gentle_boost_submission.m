Data_conversion
%%
n = size(X_train,1);
m = floor(log(n - 1)/log(3));
learnRate = 0.1:0.1:1;

maxNumSplits = 3.^(0:m);
numMNS = numel(maxNumSplits);

numTrees =1000;
ens = fitcensemble(X_Train,Labels_Train, 'Method', 'GentleBoost');

outputs_train = predict(ens,X_Train);

TP_train= sum(outputs_train==Labels_Train & outputs_train==1);
FP_train= sum(outputs_train~=Labels_Train & outputs_train==1);
TN_train= sum(outputs_train==Labels_Train & outputs_train==0);
FN_train= sum(outputs_train~=Labels_Train & outputs_train==0);

Precision_train= TP_train/(TP_train+FP_train);
Recall_train= TP_train/(TP_train+FN_train);
F1_train= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train))

outputs_test = predict(ens,X_Test);

%%
N_test=length(X_Test);
result(:,1)=[1:N_test];
result(:,2)=outputs_test;

%Construction of the csv file
Header={'Id', 'Label'};
textHeader = strjoin(Header, ',');

fid = fopen('result_GentleBoost.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);

dlmwrite('result_GentleBoost.csv', result, '-append');
