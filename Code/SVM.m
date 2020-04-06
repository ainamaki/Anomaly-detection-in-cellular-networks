clear all;
close all;

% Load train data
train=readtable('ML-MATT-CompetitionQT1920_train.csv');
test=readtable('ML-MATT-CompetitionQT1920_test.csv');

X_train=train{1:end, 3:12};
Labels_train=train{1:end,14};

X_test=test{1:end, 3:12};

%% 2D projection
% Let's remove and ignore NaN samples

X_train1=X_train((all((~isnan(X_train)),2)),:);
Labels_train1=Labels_train((all((~isnan(X_train)),2)),:);
n_clases=2;

W_fc=mda_ml(X_train1,Labels_train1,n_clases);

X_train=X_train*W_fc;
X_test=X_test*W_fc;

%% Linear SVM Classifier 

P = 0.1;
Linear_model = fitcsvm(X_train, Labels_train);
Linear_out = predict(Linear_model, X_train);

Err_train=sum(Linear_out~=Labels_train)/length(Labels_train);
TP= sum(Linear_out==Labels_train & Linear_out==1);
FP= sum(Linear_out~=Labels_train & Linear_out==1);
TN= sum(Linear_out==Labels_train & Linear_out==0);
FN= sum(Linear_out~=Labels_train & Linear_out==0);

Precision= TP/(TP+FP);
Recall= TP/(TP+FN);
F1_l_SVM= 2* ((Precision*Recall)/(Precision+Recall));

%Linear_out = predict(Linear_model, X_test);

% Amb Linear SVM classifica tot a 0...

%% Linear classifier
P=0.5:0.5:5;

for i=1:length(P)
    Linear_model = fitcsvm(X_train, Labels_train,'BoxConstraint',P(i));
    Linear_out = predict(Linear_model, X_train);
    Err_train(i)=sum(Linear_out~=Labels_train)/length(Labels_train);
    Linear_out = predict(Linear_model, X_test);
    i
    clear Linear_out
end

figure
plot (P, Err_train)
title('Training and Validation error for Linear SVM')
xlabel('P')
ylabel('Error')
legend({'Train Error','Validation Error'})
clear i_lineal

%% Non-linear classifier, gaussian kernel

P = 4.8;
h=0.5;
Gauss_model = fitcsvm(X_train, Labels_train, 'BoxConstraint',P,...
    'KernelFunction','RBF','KernelScale',h);
Gauss_out = predict(Gauss_model, X_train);
Err_train=sum(Gauss_out~=Labels_train)/length(Labels_train);
TP= sum(Gauss_out==Labels_train & Gauss_out==1);
FP= sum(Gauss_out~=Labels_train & Gauss_out==1);
TN= sum(Gauss_out==Labels_train & Gauss_out==0);
FN= sum(Gauss_out~=Labels_train & Gauss_out==0);

Precision= TP/(TP+FP);
Recall= TP/(TP+FN);
F1= 2* ((Precision*Recall)/(Precision+Recall));

Gauss_out_test = predict(Gauss_model, X_Test_fs);
N_test=height(test);
result(:,1)=[1:N_test];
result(:,2)=Gauss_out_test;

%% Construction of the csv file
Header={'Id', 'Label'};
textHeader = strjoin(Header, ',');

fid = fopen('result_SVM.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);

dlmwrite('result_SVM.csv', result, '-append');
