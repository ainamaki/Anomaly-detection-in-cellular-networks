clear all;
close all;

% Load train data
data_train=readtable('ML-MATT-CompetitionQT1920_train.csv');
data_test=readtable('ML-MATT-CompetitionQT1920_test.csv');

% Let's identify NaN samples in order to remove them
X_subtrain=data_train{1:end, 3:12};
X_subtest=data_test{1:end, 3:13};

Labels_Train=data_train{1:end,14};

idx_train_all= ~isnan(X_subtrain);
idx_test_all= ~isnan(X_subtest);

idx_train= idx_train_all(:,1).*idx_train_all(:,2).*idx_train_all(:,3).*idx_train_all(:,4).*idx_train_all(:,5).*idx_train_all(:,6).*idx_train_all(:,7).*idx_train_all(:,8).*idx_train_all(:,9).*idx_train_all(:,10);
idx_test= idx_test_all(:,1).*idx_test_all(:,2).*idx_test_all(:,3).*idx_test_all(:,4).*idx_test_all(:,5).*idx_test_all(:,6).*idx_test_all(:,7).*idx_test_all(:,8).*idx_test_all(:,9).*idx_test_all(:,10).*idx_test_all(:,11);
clear idx_train_all idx_test_all

X_subtrain(isnan(X_subtrain))=0;
X_subtest(isnan(X_subtest))=0;



% Labels_Train=Labels_train_nan((all((~isnan(X_subtrain_nan)),2)),:);
% X_subtrain=X_subtrain_nan((all((~isnan(X_subtrain_nan)),2)),:);
% X_subtest=X_subtest_nan((all((~isnan(X_subtest_nan)),2)),:);

time_train= table2array(data_train(:, 1));
cellName_train= table2array(data_train(:, 2));
maxUE_UL_DL_train= table2array(data_train(:, 13));

time_test= table2array(data_test(:, 1));
cellName_test= table2array(data_test(:, 2));
j=1;
k=1;
for i=1:length(time_train)
    
        t=time_train{i};
        m_time= sscanf(t,'%2d:%2d');
        hours(i, 1)= m_time(1);
        minutes(i,1)= m_time(2);
        time_sin_train(i,1)= sin((2*pi*(hours(i,1)*60+minutes(i,1))*0.25)/360);
        time_cos_train(i,1)= cos((2*pi*(hours(i,1)*60+minutes(i,1))*0.25)/360);
        clear t m_time;

        cn=cellName_train{i};
        cn_n= strrep(cn, 'LTE', '');
        m_cellName= sscanf(cn_n,'%2d%s');
        number_train(i,1)=m_cellName(1);
        letter_train(i,1)=m_cellName(2);
        clear m_cellName cn cn_n;  
    if (idx_train(i)==0)
        maxUE_UL_DL_train{i}='0';
    end
        maxUE=maxUE_UL_DL_train{i};
        m_maxUE= sscanf(maxUE,'%2d');
        data_maxUE_train(i,1)=m_maxUE(1);
        clear maxUE;
       
        %j=j+1;
    %end
end
for i=1:length(time_test)
    %if (idx_test(i)~=0)
        t=time_test{i};
        m_time= sscanf(t,'%2d:%2d');
        hours(i, 1)= m_time(1);
        minutes(i,1)= m_time(2);
        time_sin_test(i,1)= sin((2*pi*(hours(i,1)*60+minutes(i,1))*0.25)/360);
        time_cos_test(i,1)= cos((2*pi*(hours(i,1)*60+minutes(i,1))*0.25)/360);
        clear t m_time;

        cn=cellName_test{i};
        cn_n= strrep(cn, 'LTE', '');
        m_cellName= sscanf(cn_n,'%2d%s');
        number_test(i,1)=m_cellName(1);
        letter_test(i,1)=m_cellName(2);
        clear m_cellName cn cn_n; 

end
clear cellName hours i j m_maxUE maxUE_UL_DL minutes time cellName_test cellName_train idx_train idx_test k Labels_train_nan maxUE_UL_DL_train time_test time_train X_subtest_nan X_subtrain_nan

X_Train= [time_sin_train time_cos_train number_train letter_train X_subtrain data_maxUE_train]; 
X_Test= [time_sin_test time_cos_test number_test letter_test X_subtest]; 
clear data_test
clear data_train
clear time_sin_train time_cos_train number_train letter_train X_subtrain data_maxUE_train
clear time_sin_test time_cos_test number_test letter_test X_subtest