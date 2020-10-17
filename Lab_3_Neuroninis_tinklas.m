close all, clear

load Lab1Duomenys.txt
neuron_cnt = 40;
koef = 0.3; % coeeficent of what percentage is used for test data

% Divide data to training and testing
ind=randperm(size(Lab1Duomenys,1));             %Gives index in random order
dalinam=round(size(Lab1Duomenys,1)*koef);       %we change porporsion of train and test data 
TESTdata=Lab1Duomenys(ind(1:dalinam),:);        %from first till dalinam index
TRAINdata=Lab1Duomenys(ind(dalinam+1:end),:);

% Davide data to inputs (IN) and outputs (OUT)
INdataTEST=TESTdata(:,2:3)';                    %input to neuron net
OUTdataTEST=TESTdata(:,4)';
INdataTRAIN=TRAINdata(:,2:3)';
OUTdataTRAIN=TRAINdata(:,4)';
    
%Train sensor with given neuron count
net = feedforwardnet(neuron_cnt);
net = train(net,INdataTRAIN,OUTdataTRAIN);
%Testing with test data witch is never seen by net
y = net(INdataTEST);
perf = perform(net,y,OUTdataTEST) %calculate offset

%Get data of our neuron networks' parameters 
INw=net.IW{1}           % input layer weights
HLw=net.LW{2}           % output layer weights
B1w=net.b{1}            % bias input weight
B2w=net.b{2}            % bias juncture hidden layer weight
 
%tansing function is implemented as:
% a = tansig(n) = 2/(1+exp(-2*n))-1
 
[~,PSmapIn_1] = mapminmax(INdataTRAIN(1,:),-1,1);       %normalise input
[~,PSmapIn_2] = mapminmax(INdataTRAIN(2,:),-1,1);
[~,PSmapOut] = mapminmax(OUTdataTRAIN(1,:),-1,1);       %normalise output
Y1 = mapminmax('apply', TESTdata(:,2)', PSmapIn_1);     %apply our maps to testing data
Y2 = mapminmax('apply', TESTdata(:,3)', PSmapIn_2);
Input=[Y1; Y2];

for i=1:length(TESTdata')                       %go throught all inputs 
    in1=Input(1,i);
    in2=Input(2,i);
    for ii=1:length(INw)                        %  go thought weights of inouts
        in_tmp=in1*INw(ii,1)+in2*INw(ii,2)+B1w(ii);  
        inwin(ii)=2/(1+exp(-2*in_tmp))-1;       % using tansig function
    end
    for ih=1:length(HLw)                        % with weights of hidden layer 
        h_tmp=inwin(ih)*HLw(ih);
        inwhl(ih)=h_tmp;                        % our transmission function is linear
    end    
    Out(i)=sum(inwhl)+B2w;                      % add all up and don’t forget bias
end

Out_tikras=mapminmax('reverse',Out,PSmapOut);   % normalize outputs

%Draw figure with both outputs (1-made with net, 2- manually)
figure, plot(Out_tikras,'-ob'),hold on, plot(y,'-*r'),grid
legend('Atgamintas kodo eilute','net MATLAB struktura')