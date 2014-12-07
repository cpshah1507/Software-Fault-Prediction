
data = csvread('final.arff');
inputs = data(:,1:end -1);
targets = data(:, end);

[inputs, targets] = SMOTE(inputs, targets);
targets = targets';
inputs = inputs';
targets_new = zeros(size(targets,2),2);
for i= 1 :size(targets,2)
    targets_new(i,targets(i) + 1) = 1; 
end
% Create a Fitting Network
hiddenLayerSize = 20;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
 
% Train the Network
[net,tr] = train(net,inputs,targets_new');
 
% Test the Network
outputs_new = net(inputs);

[temp, idx] = max(outputs_new, [], 1);
outputs = idx - 1;

stats = Evaluate(targets,outputs)
stats(1)
stats(6)
% accuracy sensitivity specificity precision recall f_measure gmean
% View the Network
view(net)
 
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
% figure, plotregression(targets,outputs)
% figure, ploterrhist(errors)