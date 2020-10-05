close all, clear

%rng(20);
load Lab1Duomenys.txt
neuron_cnt = 30;
koef = 0.2; % coeeficent of what percentage is used for test data

% Divide data training and testing
ind=randperm(size(Lab1Duomenys,1)); %Gives index in random order !!!!!CAN CHANGE SEED!!!
dalinam=round(size(Lab1Duomenys,1)*koef); %we change porporsion of train and test data 
TESTdata=Lab1Duomenys(ind(1:dalinam),:); %nuo pirmo iki dalinam indekso
TRAINdata=Lab1Duomenys(ind(dalinam+1:end),:);

% Paskirstome  iejimus (IN) ir isejimus (OUT)
INdataTEST=TESTdata(:,2:3)'; %iejimas i neuronini tinkla
OUTdataTEST=TESTdata(:,4)';
INdataTRAIN=TRAINdata(:,2:3)';
OUTdataTRAIN=TRAINdata(:,4)';
    
% Treniruojam jutikl? uzduodami pasl?pt? neuron? skai?i? 
net = feedforwardnet(neuron_cnt); % we change number of neurons
%net = feedforwardnet(neuron_cnt,'trainlm'); % we change number of neurons
net = train(net,INdataTRAIN,OUTdataTRAIN);
% Testuojame su testavimo imtimi t.y. niekada nematyta
y = net(INdataTEST);  %cia irgi
perf = perform(net,y,OUTdataTEST) % Suskai?iuojam tikslumo kriterij? / paklaid?

%Get data our Neuron networks' calculation 
INputWeigts=net.IW{1} % iejimo sluoksnio svoriai
HiddenLayerWeigts=net.LW{2} % išejimo sluoksnio svoriai
B1w=net.b{1} % biaso jungties iejimo sluoksniui svoris
B2w=net.b{2} % biaso jungties pasl?ptam sluoksniui svoris
 

% randam help dokumentacijoje, kad 
% a = tansig(n) = 2/(1+exp(-2*n))-1
% a - funkcijos tansig outputas
% n - funkcijos tansig inputas
 

 
[~,PSmap1] = mapminmax(TRAINdata(:,2)',-1,1);                     % atliekam iejimo kintam?j? normavim?
[~,PSmap2] = mapminmax(TRAINdata(:,3)',-1,1);                    % atliekam iejimo kintam?j? normavim?
Y1 = mapminmax('apply', TESTdata(:,2)', PSmap1);
Y2 = mapminmax('apply', TESTdata(:,3)', PSmap2);
Iejimas=[Y1; Y2];  %cia blogai?????
Iejimas = Iejimas';

    for i=1:length(TESTdata')                               % per ??jimus
        in=Iejimas(i,:);
        for ii=1:length(INputWeigts)                     % per svorius INputo sluoksnio ir bias nepamirstam
            in_tmp=in.*INputWeigts(ii,:)+B1w(ii);  
            inwin(ii,1)=2/(1+exp(-2*in_tmp(1)))-1;       % per tansig funkcija
            inwin(ii,2)=2/(1+exp(-2*in_tmp(2)))-1;
        end
        for ih=1:length(HiddenLayerWeigts)                        % per svorius pasl?pto sluoksnio
            h_tmp=inwin(ih,:)*HiddenLayerWeigts(ih);
            inwhl(ih,1)=h_tmp(1);                        % nes perdavimo funkcija tiesine
            inwhl(ih,2)=h_tmp(2); 
        end    
        Out1(i,1)=sum(inwhl(:,1))+B2w;                      % nes perdavimo funkcija tiesine viska sumuojam ir pridedam bias
        Out1(i,2)=sum(inwhl(:,2))+B2w;
    end
        Out_tikras1=mapminmax('reverse',Out1(:,1),PSmap1);         % atliekam išejimo kintamuju atnormavima
        Out_tikras2=mapminmax('reverse',Out1(:,2),PSmap2);         % atliekam išejimo kintamuju atnormavima


%Out_tikras1=mapminmax('reverse',Out1,PSmap1);         % atliekam išejimo kintamuju atnormavima
%Out_tikras2=mapminmax('reverse',Out2,PSmap2);         % atliekam išejimo kintamuju atnormavima


Out_tikras = Out_tikras1+ Out_tikras2;

% Br?žiam paveiksla , kur atvaizduojam abu atsakus
figure, plot(Out_tikras,'-ob'),hold on, plot(y,'-*r'),grid
legend('Atgamintas_kodo_eilute','net MATLAB struktura')
