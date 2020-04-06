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
x = 1:100;
n_weak_trees = x(rem(x,2)==1); %take odd numbers of trees 
for i=1:length(n_weak_trees)

    alpha=9.0000e-05;
    for j=1:n_weak_trees(i)
        n_samples= randi([round(0.5*N_train) N_train-1]);
        R = [1 N_train];
        index= round(rand(n_samples,1)*range(R)+min(R));
        X_subtree_train= X_train(index,:);
        Labels_subtree_train= Labels_train(index);
        tree = fitctree(X_subtree_train,Labels_subtree_train, 'prune', 'off','SplitCriterion', 'deviance');
        tree_alpha= prune(tree, 'alpha', alpha);

        outputs_subtrain (:,j) = predict(tree_alpha,X_Train);

    end

    outputs_train=sum(outputs_subtrain,2);
    for j=1:length(outputs_train)
        if(outputs_train(j)>n_weak_trees(i)/2)
            outputs_train(j)=1;
        else
            outputs_train(j)=0;
        end
    end
    TP_train= sum(outputs_train==Labels_Train & outputs_train==1);
    FP_train= sum(outputs_train~=Labels_Train & outputs_train==1);
    TN_train= sum(outputs_train==Labels_Train & outputs_train==0);
    FN_train= sum(outputs_train~=Labels_Train & outputs_train==0);

    Precision_train= TP_train/(TP_train+FP_train);
    Recall_train= TP_train/(TP_train+FN_train);
    F1_train(i)= 2* ((Precision_train*Recall_train)/(Precision_train+Recall_train))
end

figure
plot (n_weak_trees, F1_train)
xlabel('Number of weak classifiers')
ylabel('F1')
legend('Training subset')




