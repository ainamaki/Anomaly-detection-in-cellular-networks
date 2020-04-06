Data_conversion
%%
N_classes = 2;
N_samp_train = length(X_Train);
N_samp_test = length(X_Test);

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
%%
% Mixing of vectors not to have all belonging to a class together
Permutation=randperm(length(Index_train));
Index_train=Index_train(Permutation);
Permutation=randperm(length(Index_val));
Index_val=Index_val(Permutation);
clear Permutation i_class index N_i_class I_train I_val 

% generation of training, validation and test sets
X_train=X_Train(Index_train,:);
N_train=length(X_train);
Labels_train=Labels_Train(Index_train);
X_val=X_Train(Index_val,:);
Labels_val=Labels_Train(Index_val);
%%
method= {'Bag', 'AdaBoostM1', 'GentleBoost', 'LogitBoost', 'LPBoost', 'RobustBoost', 'RUSBoost'};
%learners= {'discriminant', 'knn','tree'};
learners= {'tree'};
learning_rate= 0.1:0.1:1;
max_i=0;
max_j=0;
max_k=0;
F1_max=0;
for i=1:length(method)
    for j=1:length(learners)    
        for k=1:length(learning_rate)
            ens = fitcensemble(X_train,Labels_train, 'Method',char(method(i)), 'Learner', char(learners(j)),'nlearn',learning_rate(k));
            outputs_train=predict(ens,X_train);
            TP_train= sum(outputs_train==Labels_train & outputs_train==1);
            FP_train= sum(outputs_train~=Labels_train & outputs_train==1);
            TN_train= sum(outputs_train==Labels_train & outputs_train==0);
            FN_train= sum(outputs_train~=Labels_train & outputs_train==0);

            Precision_train= TP_train/(TP_train+FP_train);
            Recall_train= TP_train/(TP_train+FN_train);
            F1_train= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train))
            if(F1_train>F1_max)
                F1_max=F1_train;
                max_i=i;
                max_j=j;
                max_k=k;
            end
        end
    end
end

ens = fitcensemble(X_train,Labels_train, 'Method',char(method(max_i)), 'Learner', char(learners(max_j)),'nlearn',learning_rate(max_k));
outputs_val=predict(ens,X_val);
TP_val= sum(outputs_val==Labels_val & outputs_val==1);
FP_val= sum(outputs_val~=Labels_val & outputs_val==1);
TN_val= sum(outputs_val==Labels_val & outputs_val==0);
FN_val= sum(outputs_val~=Labels_val & outputs_val==0);

Precision_val= TP_val/(TP_val+FP_val);
Recall_val= TP_val/(TP_val+FN_val);
F1_val = 2*((Precision_val*Recall_val)/(Precision_val+Recall_val));