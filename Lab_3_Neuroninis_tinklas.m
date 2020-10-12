close all, clear

load Lab1Duomenys.txt
neuron_cnt = 40;
koef = 0.3; % coeeficent of what percentage is used for test data

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
INw=net.IW{1} % iejimo sluoksnio svoriai
HLw=net.LW{2} % išejimo sluoksnio svoriai
B1w=net.b{1} % biaso jungties iejimo sluoksniui svoris
B2w=net.b{2} % biaso jungties pasl?ptam sluoksniui svoris
 
% randam help dokumentacijoje, kad 
% a = tansig(n) = 2/(1+exp(-2*n))-1
% a - funkcijos tansig outputas
% n - funkcijos tansig inputas
 

 
[~,PSmap1] = mapminmax(INdataTRAIN(1,:)',-1,1);                     % atliekam iejimo kintam?j? normavim?
[~,PSmap2] = mapminmax(INdataTRAIN(2,:)',-1,1);                    % atliekam iejimo kintam?j? normavim?
[~,PSmapOut] = mapminmax(OUTdataTRAIN(1,:)',-1,1);                    % atliekam iejimo kintam?j? normavim?
Y1 = mapminmax('apply', TESTdata(:,2)', PSmap1);
Y2 = mapminmax('apply', TESTdata(:,3)', PSmap2);
Iejimas=[Y1; Y2];  %cia blogai????? !!!!!!!!!!!

    for i=1:length(TESTdata')                               % per ??jimus
        in1=Iejimas(1,i);
        in2=Iejimas(2,i);
        for ii=1:neuron_cnt                      % per svorius INputo sluoksnio ir bias nepamirstam
            in_tmp=in1*INw(ii,1)+in1*INw(ii,2)+B1w(ii);  
            inwin(ii)=2/(1+exp(-2*in_tmp))-1;       % per tansig funkcija
        end
        for ih=1:length(HLw)                        % per svorius pasl?pto sluoksnio
            h_tmp=inwin(ih)*HLw(ih);
            inwhl(ih)=h_tmp;                        % nes perdavimo funkcija tiesine
        end    
        Out1(i)=sum(inwhl)+B2w;                      % nes perdavimo funkcija tiesine viska sumuojam ir pridedam bias
    end

Out_tikras=mapminmax('reverse',Out1,PSmapOut);         % atliekam išejimo kintamuju atnormavima
%Out_tikras2=mapminmax('reverse',Out2,PSmap2);         % atliekam išejimo kintamuju atnormavima


% Br?žiam paveiksla , kur atvaizduojam abu atsakus
figure, plot(Out_tikras,'-ob'),hold on, plot(y,'-*r'),grid
legend('Atgamintas_kodo_eilute','net MATLAB struktura')