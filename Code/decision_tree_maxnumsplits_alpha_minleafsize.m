Data_conversion

N_classes = 2;
N_samp = length(X_Train);

% training, validation indices
rng('default'); % for reproducibility
P_train=0.7;
P_test=0;
P_val=1-P_train-P_test;
Index_train=[];
Index_val=[];


for i_class=0:N_classes-1
    index=find(Labels_Train==i_class);
    N_i_class=length(index);
    [I_train,I_val, I_test] = dividerand(N_i_class,P_train,P_val,P_test);
    Index_train=[Index_train;index(I_train)];
    Index_val=[Index_val;index(I_val)];
end

% Mixing of vectors not to have all belonging to a class together
Permutation=randperm(length(Index_train));
Index_train=Index_train(Permutation);
Permutation=randperm(length(Index_val));
Index_val=Index_val(Permutation);
clear Permutation i_class index N_i_class I_train I_val 

% generation of training, validation and test sets
X_train=X_Train(Index_train,:);
Labels_train=Labels_Train(Index_train);
X_val=X_Train(Index_val,:);
Labels_val=Labels_Train(Index_val);

%% For different values of Maximum number of splits
max_num_splits=1:length(X_Train)-1;
for i=1:max_num_splits(end);
 % Tree classifier design
 tree = fitctree(X_train,Labels_train, 'prune', 'off','MaxNumSplits', i);

 % Measure Train error
 outputs_train = predict(tree,X_train);
 Tree_Pe_train(i)=sum(Labels_train ~= outputs_train)/length(Labels_train);

 % Measure Val error
 outputs_val = predict(tree,X_val);
 Tree_Pe_val(i)=sum(Labels_val ~= outputs_val)/length(Labels_val);
 
 
TP_train= sum(outputs_train==Labels_train & outputs_train==1);
FP_train= sum(outputs_train~=Labels_train & outputs_train==1);
TN_train= sum(outputs_train==Labels_train & outputs_train==0);
FN_train= sum(outputs_train~=Labels_train & outputs_train==0);

Precision_train= TP_train/(TP_train+FP_train);
Recall_train= TP_train/(TP_train+FN_train);
F1_train(i)= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train));

TP_val= sum(outputs_val==Labels_val & outputs_val==1);
FP_val= sum(outputs_val~=Labels_val & outputs_val==1);
TN_val= sum(outputs_val==Labels_val & outputs_val==0);
FN_val= sum(outputs_val~=Labels_val & outputs_val==0);

Precision_val= TP_val/(TP_val+FP_val);
Recall_val= TP_val/(TP_val+FN_val);
F1_val(i)= 2*((Precision_val*Recall_val)/(Precision_val+Recall_val));

 % Measure Test error
%  outputs = predict(tree,X_test);
%  Tree_Pe_test(i)=sum(Labels_test ~= outputs)/length(Labels_test);
i
end
figure
plot (max_num_splits, Tree_Pe_train)
hold on
plot (max_num_splits, Tree_Pe_val)
xlabel('MaxNumSplits')
ylabel('Error')
legend('Training subset', 'Validation subset')

figure
plot (max_num_splits, F1_train)
hold on
plot (max_num_splits, F1_val)
xlabel('MaxNumSplits')
ylabel('F1')
legend('Training subset', 'Validation subset')
%% For different values of alpha

alpha = 0:0.00001:0.001;

for i=1:length(alpha);
 % Tree classifier design
 tree = fitctree(X_train,Labels_train, 'prune', 'off','SplitCriterion', 'deviance');
 tree_alpha= prune(tree, 'alpha', alpha(i));

 % Measure Train error
 outputs_train = predict(tree_alpha,X_train);
 Tree_Pe_train(i)=sum(Labels_train ~= outputs_train)/length(Labels_train);

 % Measure Val error
 outputs_val = predict(tree_alpha,X_val);
 Tree_Pe_val(i)=sum(Labels_val ~= outputs_val)/length(Labels_val);

 % Measure Test error
 TP_train= sum(outputs_train==Labels_train & outputs_train==1);
FP_train= sum(outputs_train~=Labels_train & outputs_train==1);
TN_train= sum(outputs_train==Labels_train & outputs_train==0);
FN_train= sum(outputs_train~=Labels_train & outputs_train==0);

Precision_train= TP_train/(TP_train+FP_train);
Recall_train= TP_train/(TP_train+FN_train);
F1_train(i)= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train));

TP_val= sum(outputs_val==Labels_val & outputs_val==1);
FP_val= sum(outputs_val~=Labels_val & outputs_val==1);
TN_val= sum(outputs_val==Labels_val & outputs_val==0);
FN_val= sum(outputs_val~=Labels_val & outputs_val==0);

Precision_val= TP_val/(TP_val+FP_val);
Recall_val= TP_val/(TP_val+FN_val);
F1_val(i)= 2*((Precision_val*Recall_val)/(Precision_val+Recall_val));
i
end
figure
plot (alpha, Tree_Pe_train)
hold on
plot (alpha, Tree_Pe_val)
xlabel('Alpha')
ylabel('Error')
legend('Training subset', 'Validation subset')

figure
plot (alpha, F1_train)
hold on
plot (alpha, F1_val)
xlabel('Alpha')
ylabel('F1')
legend('Training subset', 'Validation subset')

%% For different values of MinLeafSize

min_leaf=1:30;
for i=1:min_leaf(end);
 % Tree classifier design
 tree = fitctree(X_train,Labels_train, 'prune', 'off','MinLeafSize', i);

 % Measure Train error
 outputs_train = predict(tree,X_train);
 Tree_Pe_train(i)=sum(Labels_train ~= outputs_train)/length(Labels_train);

 % Measure Val error
 outputs_val = predict(tree,X_val);
 Tree_Pe_val(i)=sum(Labels_val ~= outputs_val)/length(Labels_val);
 
 
TP_train= sum(outputs_train==Labels_train & outputs_train==1);
FP_train= sum(outputs_train~=Labels_train & outputs_train==1);
TN_train= sum(outputs_train==Labels_train & outputs_train==0);
FN_train= sum(outputs_train~=Labels_train & outputs_train==0);

Precision_train= TP_train/(TP_train+FP_train);
Recall_train= TP_train/(TP_train+FN_train);
F1_train(i)= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train));

TP_val= sum(outputs_val==Labels_val & outputs_val==1);
FP_val= sum(outputs_val~=Labels_val & outputs_val==1);
TN_val= sum(outputs_val==Labels_val & outputs_val==0);
FN_val= sum(outputs_val~=Labels_val & outputs_val==0);

Precision_val= TP_val/(TP_val+FP_val);
Recall_val= TP_val/(TP_val+FN_val);
F1_val(i)= 2*((Precision_val*Recall_val)/(Precision_val+Recall_val));
i
end
figure
plot (min_leaf, Tree_Pe_train)
hold on
plot (min_leaf, Tree_Pe_val)
xlabel('MinLeafSize')
ylabel('Error')
legend('Training subset', 'Validation subset')

figure
plot (min_leaf, F1_train)
hold on
plot (min_leaf, F1_val)
xlabel('MinLeafSize')
ylabel('F1')
legend('Training subset', 'Validation subset')