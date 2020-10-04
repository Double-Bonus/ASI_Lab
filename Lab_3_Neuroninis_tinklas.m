close all, clear all
 
%============================ tik kai pirma darba darote ==================
[x,t] = simplefit_dataset; % užsikraunami duomenys testiniai
net = feedforwardnet([10]) % užduodama tinklo strukt?ra su vienu pasl?ptu sluoksniu ir 10 neuron? jame
net = train(net,x,t); % Atliekamas modelio treniravimas
% view(net) % perži?rim tinklo strukt?r?
y = net(x); % gaunam tinklo atsak? prie tam tikr? INput reikšmi?
perf = perform(net,y,t) % Suskai?iuojam tikslumo kriterij? / paklaid?
%==========================================================================
 
 
INw=net.IW{1} % iejimo sluoksnio svoriai
HLw=net.LW{2} % išejimo sluoksnio svoriai
B1w=net.b{1} % biaso jungties iejimo sluoksniui svoris
B2w=net.b{2} % biaso jungties pasl?ptam sluoksniui svoris
 
% randam help dokumentacijoje, kad 
% a = tansig(n) = 2/(1+exp(-2*n))-1
% a - funkcijos tansig outputas
% n - funkcijos tansig inputas
 
 
[~,PS] = mapminmax(t,-1,1);                     % atliekam iejimo kintam?j? normavim?
Y = mapminmax('apply', x, PS);
Iejimas=Y;  %cia blogai?????

for i=1:length(x)                               % per ??jimus
    in=Iejimas(i);
    for ii=1:length(INw)                        % per svorius INputo sluoksnio ir bias nepamirstam
        in_tmp=in*INw(ii)+B1w(ii);  
        inwin(ii)=2/(1+exp(-2*in_tmp))-1;       % per tansig funkcija
    end
    for ih=1:length(HLw)                        % per svorius pasl?pto sluoksnio
        h_tmp=inwin(ih)*HLw(ih);
        inwhl(ih)=h_tmp;                        % nes perdavimo funkcija tiesine
    end    
    Out(i)=sum(inwhl)+B2w;                      % nes perdavimo funkcija tiesine viska sumuojam ir pridedam bias
end
Out_tikras=mapminmax('reverse',Out,PS);         % atliekam išejimo kintamuju atnormavima


% Br?žiam paveiksla , kur atvaizduojam abu atsakus
figure, plot(Out_tikras,'-ob'),hold on, plot(y,'-*r'),grid
legend('Atgamintas_kodo_eilute','net MATLAB struktura')
