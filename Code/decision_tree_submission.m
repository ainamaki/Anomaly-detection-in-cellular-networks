Data_conversion

%%

alpha=9.0000e-05;
tree = fitctree(X_Train,Labels_Train, 'prune', 'off','SplitCriterion', 'deviance');
tree_alpha= prune(tree, 'alpha', alpha);

outputs_train = predict(tree_alpha,X_Train);

TP_train= sum(outputs_train==Labels_Train & outputs_train==1);
FP_train= sum(outputs_train~=Labels_Train & outputs_train==1);
TN_train= sum(outputs_train==Labels_Train & outputs_train==0);
FN_train= sum(outputs_train~=Labels_Train & outputs_train==0);

Precision_train= TP_train/(TP_train+FP_train);
Recall_train= TP_train/(TP_train+FN_train);
F1_train= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train))

outputs_test = predict(tree_alpha,X_Test);

%%
N_test=length(X_Test);
result(:,1)=[1:N_test];
result(:,2)=outputs_test;

%Construction of the csv file
Header={'Id', 'Label'};
textHeader = strjoin(Header, ',');

fid = fopen('result_decision_trees_deviance_alpha.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);

dlmwrite('result_decision_trees_deviance_alpha.csv', result, '-append');

