training = xlsread('training_set_ALL_AML.xlsx');
test = xlsread('test_set_ALL_AML.xlsx');
training_labels = xlsread('training_labels.xlsx');
test_labels = xlsread('test_labels.xlsx');

training = training(:,3:40); % taking only number values 
training = manorm(transpose(training)); % microarray normalization
test = test(:,3:37); 
test = manorm(transpose(test));

[num_sv, beta, b0] = svmfit(training, training_labels, 'linear', 15)

predictions = svmpredict(training, training_labels, test, 'linear', beta, b0)

acc = accuracy(test_labels, predictions)
