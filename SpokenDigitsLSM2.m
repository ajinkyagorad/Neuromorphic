%% Continue from param script file
addpath ../Replicate;
addpath ../SpeechDataset;

parfor res_iter=1:length(Wds(1,1,:))
  Wres = Wds(:,:,res_iter);
    %% LSM
    % GeneralLSM code function
    % can run in both direct and indirect method
    %{
%% Template for param script file
SAVE_VIDEO =0;
RESERVOIR = 1;
LOGTXT = 0;
SYNAPSE_ORDER= 2;
RESET_VAR_PER_SAMPLE =0;
LOAD_EXISTING_INPUT = 0;
W_INIT_RANDOM = 0;
MODIFIED_WT_UPDATE = 0;
PROBABILISTIC_WT_UPDATE = 1;
REVERSE_PRUNE = 2;
RESERVOIR_2DGRID=0; % if yes, then reservoir size must be 2D
RNG.option = 'default'; rng(RNG.option); % default/shuffle
RNG.rng = rng; % store rng object for reference
GET_DIST = 1; % overall distribution of spikes in reservoir( stored in param)
%DATASET= 'TI46-IF.mat';

DATASET= 'digitdata.mat';
%% save file names
savefilename = mfilename;

%% PARAM1
%Input Parameters
fs = 8E3; out_fs = 1E3; df = 8; earQ = 4; stepfactor =5/32; differ=1; agcf=1; tauf=2; max_AGC = 4E-4; BSAtau =4E-3; BSAtau2 = 1E-3;BSAfilterFac = 1;
appendS = 50;
% Neuron Parameters
tauV = 32E-3; tauC = 64E-3; tauv1e = 8E-3; tauv2e = 4E-3; tauv1i = 4E-3; tauv2i = 2E-3; RefracPeriod = 2E-3; Vth = 20;
% Reservoir Parameters %3D reservoir
resSize = [7 7 7]; Wres = [3.5 2; -4 -1]; r0 = 2; Kres = [0.45 0.3;0.6 0.15]; f_inhibit = 0.2; GinMag = 8; InputFanout = 10;
% Classifier Parameters
Nout = 10; dW0 = 0.01; Wlim = 8; Cth = 5; DeltaC= 3   ; dC =1; Iinf = 10000; Cfactor = 1; probN = 1/Nout; probP = (Nout-1)/Nout;
dt = 1E-3;
% Training Parameters
Nfold = 4; Nepochs = 200; samplesPerSpeakerPerClass=4; numSpeakers=5; samplesPerClass = numSpeakers*samplesPerSpeakerPerClass;
 fracResIn=1;
% END PARAM1s

    %}
    %% Add details of script here
    %% Updated : added weight update record spike statistics
    % activity of the network
    % total spikes of readout neuron
    % spikes of readout neuron when there is weight change
    % 2:32 AM ISTT{[1:5 14],'AccuracyTestAvg'}
    %% added modified weight update (2nd sept,18) 1:09AM IST
    %% added probabilistic wt update (3rd sept,18) 8:32PM IST
    %% v1.1 changed sample_i to sample_ii (13th sept,18)
    %% v1.2 Added random prune via prune method as 2 (25th Oct, 18) 10:00am IST
    %% v1.3 Added 2D grid LSM option.
    %    Added template of input file in this code(28th Oct, 18) 10:26pm IST
    %    Removed unused var 'T' from getNetwork
    %% v1.4 readout spikes added for last epoch
    %   appended robustness metric to results
    %   changed iter for total RO spike calc (!didn't logged, generate it
    %   separately from other  script file)
    %
    %% v1.5 added optional preprocessing
    %% store all network parameters in single object file
    versionLSM='v1.5';
%     if(exist('PARAM1','var')==1); clear PARAM1; end
%     if(exist('PARAM2','var')==1); clear PARAM2; end
%     if(exist('RESULT','var')==1); clear RESULT; end

    PARAM1=struct(  'BSAfilterFac' ,BSAfilterFac,...             
    'BSAtau'                       ,BSAtau,...                   
    'BSAtau2'                      ,BSAtau2,...                  
    'Cfactor'                      ,Cfactor,...                  
    'Cth'                          ,Cth,...                      
    'DATASET'                      ,DATASET,...                  
    'DeltaC'                       ,DeltaC,...                   
    'GET_DIST'                     ,GET_DIST,...                 
    'GinMag'                       ,GinMag,...                   
    'Iinf'                         ,Iinf,...                     
    'InputFanout'                  ,InputFanout,...              
    'Kres'                         ,Kres,...                     
    'LOAD_EXISTING_INPUT'          ,LOAD_EXISTING_INPUT,...      
    'LOGTXT'                       ,LOGTXT,...                   
    'MODIFIED_WT_UPDATE'           ,MODIFIED_WT_UPDATE,...       
    'Nepochs'                      ,Nepochs,...                  
    'Nfold'                        ,Nfold,...                    
    'Nout'                         ,Nout,...                     
    'PROBABILISTIC_WT_UPDATE'      ,PROBABILISTIC_WT_UPDATE,...  
    'RESERVOIR'                    ,RESERVOIR,...                
    'RESERVOIR_2DGRID'             ,RESERVOIR_GRID,...         
    'RESET_VAR_PER_SAMPLE'         ,RESET_VAR_PER_SAMPLE,...     
    'REVERSE_PRUNE'                ,REVERSE_PRUNE,...            
    'RNG'                          ,RNG,...                      
    'RefracPeriod'                 ,RefracPeriod,...             
    'SAVE_VIDEO'                   ,SAVE_VIDEO,...               
    'SYNAPSE_ORDER'                ,SYNAPSE_ORDER,...            
    'Vth'                          ,Vth,...                      
    'W_INIT_RANDOM'                ,W_INIT_RANDOM,...            
    'Wlim'                         ,Wlim,...                     
    'Wres'                         ,Wres,...                     
    'agcf'                         ,agcf,...                     
    'appendS'                      ,appendS,...                  
    'dC'                           ,dC,...                       
    'dW0'                          ,dW0,...                      
    'df'                           ,df,...                       
    'differ'                       ,differ,...                   
    'dt'                           ,dt,...                       
    'earQ'                         ,earQ,...                     
    'f_inhibit'                    ,f_inhibit,...                
    'fracResIn'                    ,fracResIn,...                
    'fs'                           ,fs,...                       
    'max_AGC'                      ,max_AGC,...                  
    'numSpeakers'                  ,numSpeakers,...  
    'OPTIMAL_WEIGHTS'              ,OPTIMAL_WEIGHTS,...
    'out_fs'                       ,out_fs,...                   
    'probN'                        ,probN,...                    
    'probP'                        ,probP,...                    
    'r0'                           ,r0,...                       
    'resSize'                      ,resSize,...                  
    'samplesPerClass'              ,samplesPerClass,...          
    'samplesPerSpeakerPerClass'    ,samplesPerSpeakerPerClass,...
    'savefilename'                 ,savefilename,...             
    'stepfactor'                   ,stepfactor,...               
    'tauC'                         ,tauC,...                     
    'tauV'                         ,tauV,...                     
    'tauf'                         ,tauf,...                     
    'tauv1e'                       ,tauv1e,...                   
    'tauv1i'                       ,tauv1i,...                   
    'tauv2e'                       ,tauv2e,...                   
    'tauv2i'                       ,tauv2i,...                   
    'versionLSM'                   ,versionLSM);
    
   %  PARAM1 = struct();

    fprintf('RESULT/LOG will be stored as %s.mat/LOG_*.txt',savefilename);
    %% CONFIG
    %addpath(['..' filesep 'MatlabInclude' filesep ]);
    fprintf('Running version %s\r\n',versionLSM);
  
    %% Load Data
    % Put AudioSpikes.m code here or any data generationn code
    addpath ../SpeechDataset/
    addpath ../SpeechDataset/AuditoryToolbox
    %Training Data
    dataset=load(DATASET); % load corresponding structure
    if(PREPROCESSING)
        field_dataset=fieldnames(dataset);field_dataset=field_dataset{1};
        if(isfield(dataset.(field_dataset),'class'));class='class'; info='info'; isTI46=1;else class='digit';info='subject';isTI46=0;end
        tempDATAfile = 'tempDatasetProc.mat'; % Change this if current data needs to be saved
        if(LOAD_EXISTING_INPUT)
            %load 0-9digits-10x16x4_new.mat
            load(tempDATAfile);
        else
            rng(RNG.rng.Seed); % for reproducibility
            DATA = struct([]);
            
            %fs = digitdata(1).Fs;  % assume 8KHz
            %dt = 1/out_fs;
            %tic();
            for i = 1:numel(dataset.(field_dataset))
                fprintf('%i',dataset.(field_dataset)(i).(class));
                %[filts,freqs] = DesignLyonFilters(fs);
                s = LyonPassiveEar(dataset.(field_dataset)(i).sig,fs,df,earQ,stepfactor,differ,agcf,tauf);
                s = min(s,max_AGC);
                s = s/max_AGC;
                S = BSA(s,BSAfilterFac*BSA_filter(out_fs,BSAtau,BSAtau2));
                DATA(i).type = dataset.(field_dataset)(i).(class);
                if(isTI46)
                    DATA(i).spk = str2num(dataset.(field_dataset)(i).(info){2}{12,2}(2));
                else
                    DATA(i).spk = dataset.(field_dataset)(i).subject;
                end
                DATA(i).S =sparse(logical(S));
                DATA(i).info = dataset.(field_dataset)(i).(info);
                %subplot(211);imagesc(s);title(DATA(i).type);
                %subplot(212);imagesc(S); drawnow;
            end
            %toc();
            %save(tempDATAfile,'DATA');
        end
        
        
        %dgt =[ DATA.type]; % sort accordingly
        %[~,id] = sort(dgt);
        %DATA = DATA(id);
        %DATA = reshape(DATA,samplesPerClass,Nout); % shape accordingly
        %DATA = DATA';
        %
        % if(0)
        % % show dataset speaker vs
        % spk=[DATA.spk];type=[DATA.type];
        % SMP=zeros(max(spk),max(type)+1);
        % end
        fprintf('\r\n... Data Preprocessed \r\n');
    else
        DATA=dataset.DATA;
    end
    %% PARAM2
    % General parameters
    Nstride = numel(DATA)/Nfold;  % Training
    Ke = 1/(tauv1e-tauv2e); Ke1 = 1/tauv1e;
    Ki = 1/(tauv1i-tauv2i); Ki1 = 1/tauv1i;
    alphaV = (1-dt/tauV); alphaC = (1-dt/tauC);
    alphav1e = (1-dt/tauv1e);alphav2e = (1-dt/tauv2e);
    alphav1i = (1-dt/tauv1i);alphav2i = (1-dt/tauv2i);
    RefracPeriodSamples = floor(RefracPeriod/dt);
    %nRP = ceil(RF/dt);
    %resting potential = 0, refractory period  = 2ms
    % END PARAM2
   
    %%  Reservoir
    rng(RNG.rng.Seed) % reset it for producing reservoir
    if(RESERVOIR_GRID)
        [X,Xn,G,R,E] = get3Dgrid(resSize,Wres,f_inhibit);
    else
        [X,Xn,~,G,R,E] = createNetworkM(resSize ,Wres,r0,Kres,f_inhibit,1E-3);
    end
    %[X,Xn,G,R,E] = get2DneighbourCrossConnectedNetwork(resSize,Wres,f_inhibit);
    Nres = length(E); Nsyn = length(X);
    
    alphav1=zeros(Nres,1); alphav1(E>0)=alphav1e; alphav1(E<0)=alphav1i;
    alphav2=zeros(Nres,1); alphav2(E>0)=alphav2e; alphav2(E<0)=alphav2i;
    K0 = zeros(Nres,1); K0(E>0) =Ke; K0(E<0) =Ki;
    K01 = zeros(Nres,1); K01(E>0) =Ke1; K01(E<0) =Ki1;
    G = sparse(X,Xn,G,Nres,Nres);
    Gres=G;
    
    % Input
    Nin = length(DATA(1).S(:,1)); % Input Neurons giving out spike trains
    Ain = double(sparse(applied_current_matrix(Nin,Nres,InputFanout)));
    Gin = GinMag*Ain.*sign(rand(size(Ain))-0.5);
    NsynIn = length(find(Ain~=0));
    
    PARAM2 = struct('G',G,'Nin',Nin,'Ain',Ain,'Gin',Gin,'E',E);
    
    % append gap in data
    if(appendS>0)
        for sample_i = 1:numel(DATA)
            DATA(sample_i).S = logical([DATA(sample_i).S zeros(Nin,appendS)]);
        end
        
    end
    %% RESERVOIR
    %SpikesRes.List = {};
    %SpikesRes.Info = {};
    if(RESERVOIR)
        %SpikesRes.List = {'RES','RES0'};
        %SpikesRes.Info = {'Reservoir','Reservoir-NoProp'};
        %SpikesRes.G{1} = G; SpikesRes.G{2}=G*1E-10;
        %SpikesRes.E{1} = E; SpikesRes.E{2} = E; % redundant in this scenario yet
        fprintf('Processing  reservoir spikes..\r\n');
        NinR =Nin;
        tic();
        spikeResVar = 'RES';%SpikesRes.List{iRes};
        %G = SpikesRes.G{iRes};
        %E = SpikesRes.E{iRes};
        %if(~exist('spikeResVar','var')); spikeResVar='RES'; end;
        DATA(1).(spikeResVar)=[];
        for sample_i = 1:numel(DATA)
            
            sample = DATA(sample_i);
            jmax = length(sample.S(1,:)); % time length of input
            sample_label = sample.type+1;
            fprintf('%i',sample.type);
            spikedLog = [];
            k = 0;
            % Res
            %if(RESET_VAR_PER_SAMPLE) % cannot be run in parfor loop
            I = zeros(Nres,2); % synaptic delay = 1;
            RP = -Inf*ones(Nres,1);
            V = zeros(Nres,1);
            v = zeros(Nres,SYNAPSE_ORDER);
            %Input
            vin = zeros(NinR,SYNAPSE_ORDER);
            %end
            for j = 1:jmax
                k = k+1;
                if(SYNAPSE_ORDER==0)
                    Iapp = sample.S(:,j)/dt;
                elseif(SYNAPSE_ORDER==1)
                    vin(:,1) = alphav1e*vin(:,1)+sample.S(:,j);
                    Iapp = Ke1*vin(:,1);
                else
                    vin(:,1) = alphav1e*vin(:,1)+sample.S(:,j);
                    vin(:,2) = alphav2e*vin(:,2)+sample.S(:,j);
                    Iapp = Ke*(vin(:,1)-vin(:,2));
                end
                
                if(RESERVOIR)
                    V = alphaV*V+G'*I(:,mod(k-1,2)+1)*dt+Gin*Iapp*dt;
                    V(k-RP<=RefracPeriodSamples) = 0;
                    spiked = V>Vth ; V(spiked)=0; V(V<0) = 0;
                    RP(spiked) = k;
                    if(SYNAPSE_ORDER==0)
                        I(:,mod(k,2)+1) = spiked/dt;
                    elseif(SYNAPSE_ORDER==1)
                        v(:,1) = alphav1.*v(:,1)+spiked;
                        I(:,mod(k,2)+1) = K01.*v(:,1);
                        
                    else
                        v(:,1) = alphav1.*v(:,1)+spiked;
                        v(:,2) = alphav2.*v(:,2)+spiked;
                        I(:,mod(k,2)+1) = K0.*(v(:,1)-v(:,2));
                    end
                    spikedLog(:,end+1) = spiked;
                end
            end
            %figure(2); subplot(5,2,sample_label); [a,b] = find(spikedLog); plot(b,a,'.'); drawnow;%xlabel(num2str(sample_label));drawnow;
            DATA(sample_i).(spikeResVar) = (logical(spikedLog));
            %figure(2); subplot(211); [a,b] = find(sample.S);plot(b,a,'.','MarkerSize',0.1);subplot(212); [a,b] = find(spikedLog);plot(b,a,'.','MarkerSize',0.1);title(num2str(sample_label));drawnow;
        end
        fprintf('\r\nReservoir Raster Mapping Done\r\n');
        toc();
    end
    
    %% Save DATA file
    saveDATA (savefilename,res_iter,DATA);
    %% Get Distribution
    % if(GET_DIST & 0)
    %
    %     [inDist,inSort,resDist,resSort]=showSpikesDist(DATA,SpikesRes.List,SpikesRes.Info,1); drawnow;
    %     DIST = struct('inDist',inDist,'inSort',inSort,'resDist',resDist,'resSort',resSort,'SpikesRes',SpikesRes);
    %     PARAM2.DIST = DIST;
    % end
    %% CLASSIFIER
    fprintf('Classifying reservoir spikes (Training and simultaneous testing of training and testing data)\r\n');
    k = 0;
    tic();
    RESULT = struct([]);
    
    if(RESERVOIR==0)
        NinC = Nin;
        spikeClassify = 'S';
        ActiveNeuronID=flip(inSort);
    else
        NinC = Nres; % input to the classifier
        spikeClassify = 'RES';
        %if(REVERSE_PRUNE~=0);     ActiveNeuronID=flip(resSort(:,1));end;
    end
    ConnectedNeuronMask = ones(NinC,1);
    % Deleted from here;
    PARAM2.ConnectedNeuronMask =ConnectedNeuronMask;
    
    for kfold = 1:Nfold
        %for kfold=Nfold:-1:1
        tic();
        
        
        % Input
        I = zeros(NinC,2); % synaptic delay = 1;
        RP = -Inf*ones(NinC,1);
        V = zeros(NinC,1);
        v = zeros(NinC,SYNAPSE_ORDER);
        
        % Output
        Vout = zeros(Nout,1);
        VoutR = zeros(Nout,1);
        Cout = zeros(Nout,1);
        Iteach = zeros(Nout,1);
        desired = zeros(Nout,1);
        Winit = zeros(Nout,NinC);
        Prob = zeros(Nout,NinC);
        W = Winit;
        WR = W; % for testing old data
        RPOut = -Inf*ones(Nout,1);
        %
        spikedLog=[];
        
        k =0;
        testOn = (kfold-1)*Nstride+(1:Nstride);
        trainOn  = setdiff(1:numel(DATA),testOn);
        typetrn = [DATA(trainOn).type]+1;
        typetest = [DATA(testOn).type]+1;
        [~,sortIdTrain] = sort(typetrn);
        [~,sortIdTest] = sort(typetest);
        Imerit = zeros(Nout,length([trainOn testOn]));
        robustnessMetric=[]; Rmetric = [];
        CoutLog = []; SpikeOutLog = []; IteachLog=[];
        wmean = []; accuracyTrain=[]; numCorrectTrain=[];
        accuracyTest=[]; numCorrectTest=[];
        numWchanges = 0; readoutSpikesTraining =0; readoutSpikesTestingAll=0; % total weight changes and spikes in each epoch
        
        for epoch = 1:Nepochs
            % script train & test epoch
            s = sprintf('Training epoch:%i (@t=%fs)',epoch,toc);
            fprintf('%s\t',s);if(LOGTXT);fprintf(LogFile,s);end;
            Imerit(:)=0;
            sample_ii=0;
            spikeSampleCountTest=zeros(Nout,numel(DATA));% for reading testing data
            fprintf('\t Training|Testing...\t');
            for sample_i = [trainOn testOn]
                sample_ii=sample_ii+1;
                sample = DATA(sample_i);
                jmax = length(sample.(spikeClassify)(1,:)); % time lenggth of input
                if(epoch==Nepochs);RO = zeros(Nout,jmax);end;
                sample_label = sample.type+1;
                if(sample_ii==length(trainOn)+1); fprintf('\t Testing:');end
                fprintf('%i',sample.type);
                if(RESET_VAR_PER_SAMPLE);   VoutR(:)=0; Vout(:)=0;v(:)=0;Cout(:)=0; end
                for j = 1:jmax
                    k = k+1;
                    
                    %if(SYNAPSE_ORDER==0)
                    %    Iapp = sample.(spikeClassify)(:,j)/dt;
                    %elseif(SYNAPSE_ORDER==1)
                    %    v(:,1) = alphav1e.*v(:,1)+sample.(spikeClassify)(:,j);
                    %    %vin(:,2) = alphav2e*vin(:,2)+sample.S(:,j);
                    %    Iapp = Ke1.*v(:,1);
                    %else
                    v(:,1) = alphav1e*v(:,1)+sample.(spikeClassify)(:,j);
                    v(:,2) = alphav2e*v(:,2)+sample.(spikeClassify)(:,j);
                    Iapp = Ke*(v(:,1)-v(:,2));
                    %end
                    
                    
                    if(MODIFIED_WT_UPDATE<2)
                        if(sample_ii<=length(trainOn))
                            Iteach(:) =0;
                            Iteach(Cout>Cth-dC) = -Iinf;
                            if(Cout(sample_label)<Cth+dC); Iteach(sample_label) = Iinf;
                            else Iteach(sample_label) = 0;
                            end
                        end
                    end
                    
                    Imerit(:,sample_ii) = Imerit(:,sample_ii)+W*Iapp;
                    Vout = alphaV*Vout+W*Iapp*dt+Iteach*dt;
                    VoutR = alphaV*VoutR+WR*Iapp*dt;
                    outputSpiked = Vout>Vth;Vout(Vout<0)=0;  Vout(outputSpiked) = 0;
                    outputSpikedR = VoutR>Vth;VoutR(VoutR<0)=0;  VoutR(outputSpikedR) = 0;
                    %readoutSpikesTestingAll = readoutSpikesTestingAll+ numel(find(outputSpikedR~=0));%LOG
                    if(epoch==Nepochs)% store RO spikes for last epoch for each sample
                        RO(:,j)=outputSpikedR;
                    end
                    if(sample_ii<=length(trainOn))
                        Cout = alphaC*Cout+Cfactor*outputSpiked;
                        signC = ((Cout-Cth)>0 & (Cout-Cth)<DeltaC)-((Cout-Cth)<0 & (Cout-Cth)>-DeltaC);
                        sample.spike = sample.(spikeClassify)(:,j).*ConnectedNeuronMask;
                        %if(MODIFIED_WT_UPDATE==0)
                        deltaW = dW0*(signC*sample.spike');
                        %elseif(MODIFIED_WT_UPDATE==1)
                        %    Cw = abs(signC).*((Cth-Cout)+signC.*DeltaC)/(DeltaC-dC);
                        %    deltaW = dW0*(Cw*sample.spike');
                        %elseif(MODIFIED_WT_UPDATE==2)
                        %    desired(:) = -1; desired(sample_label) = 1;
                        %    Cw = ((Cout<Cth+DeltaC).*(desired>0)-(Cout>Cth-DeltaC).*(desired<0));
                        %    deltaW = dW0*(Cw*sample.spike');
                        %else
                        %    desired(:) = -1; desired(sample_label) = 1;
                        %    deltaW=dW0*(desired*sample.spike'); % not dependent on concentration, greedy probabilistic approach
                        %end
                        
                        if(PROBABILISTIC_WT_UPDATE)
                            Prob(:)= probN; Prob(sample_label,:) = probP;
                            deltaW=deltaW.*(rand(size(deltaW))<Prob);
                        end
                        W_old=W;
                        W = W+deltaW;
                        W(W>Wlim) = Wlim;
                        W(W<-Wlim) = -Wlim;
                        %numWchanges = numWchanges+numel(find(abs(W_old-W)~=0)); % LOG
                        %readoutSpikesTraining = readoutSpikesTraining+numel(find(outputSpiked~=0)); %LOG
                        %    if(j==jmax)
                        %        wmean(:,end+1) = sum(W,2)/NinC;
                        %    end
                    end
                    spikeSampleCountTest(:,sample_ii) = spikeSampleCountTest(:,sample_ii)+outputSpikedR;
                    %CoutLog(:,k) = Cout;
                    %spikeOutLog(:,k) = outputSpiked;
                    %IteachLog(:,k) = Iteach;
                end
            end
            %% END of Trainind and testing of Train & Test data
            %{
        Imerit = Imerit/jmax;
        % Get robustness metric
        for i = 1:Nout
            Rmetric(:,i) = sum(Imerit(:,typetrn==i),2);
        end
        DRmetric = Inf*ones(Nout,Nout);
        for i =1:Nout; for j = 1:i-1
                metricDirn = zeros(Nout,1);metricDirn(i)=1;metricDirn(j)=-1;
                DRmetric(i,j) = sum((Rmetric(:,i)-Rmetric(:,j)).*metricDirn);
            end; end;
        DRmetricMin = min(DRmetric,[],2);
        robustnessMetric = sum(DRmetricMin(2:end));
        
        %if(Nout==3)
        %    set(h_Imerit,'XData',Imerit(1,:),'YData',Imerit(2,:),'ZData',Imerit(3,:));
        %end
            %}
            WR = W;
            [M,recognized] = max(spikeSampleCountTest);
            Y = spikeSampleCountTest./repmat(M,Nout,1);
            %recognizedTrain = recognized(trainOn);
            %recognizedTest = recognized(testOn);
            
            % TRAIN accuracy
            %numCorrectTrain = length(find((typetrn-recognizedTrain)==0));
            %accuracyTrain = numCorrectTrain*100/length(typetrn);
            Mn = sparse(1:length(trainOn),1+[DATA(trainOn).type],1,length(trainOn),Nout); % confusion matrix finder;
            MTest = sparse(1:length(testOn),1+[DATA(testOn).type],1,length(testOn),Nout); % confusion matrix finder;
            M  = blkdiag(Mn,MTest);
            Y = Y./repmat(max(Y),Nout,1); Y(Y~=1)=0;
            %check for correct classification only, remove no classification
            %and misclassification
            misClassifiedSamples = find(sum(Y,1)~=1);
            Y(:,misClassifiedSamples) = 0;
            CM = (Y*M); % confusion matrix
            
            accuracyTrain = 100*trace(CM(:,1:Nout))/length(trainOn);
            numCorrectTrain = trace(CM(:,1:Nout));
            accuracyTest = 100*trace(CM(:,Nout+(1:Nout)))/length(testOn);
            numCorrectTest = trace(CM(:,Nout+(1:Nout)));
            % TEST
            %numCorrectTest = length(find((typetest-recognizedTest)==0));
            %accuracyTest = numCorrectTest(end)*100/length(typetest);
            
            % **** LOG results *****
            RESULT(kfold).accTest(epoch) = accuracyTest;
            RESULT(kfold).accTrain(epoch) = accuracyTrain;
            %RESULT(kfold).numWchanges(epoch) = numWchanges;
            %RESULT(kfold).readoutSpikesTraining(epoch) = readoutSpikesTraining;
            %RESULT(kfold).readoutSpikesTestingAll(epoch) = readoutSpikesTestingAll;
            RESULT(kfold).W(:,:,epoch)= W;
            RESULT(kfold).CM(:,:,epoch) = CM;
            RESULT(kfold).spikeSCT(epoch).S=spikeSampleCountTest;
            RESULT(kfold).spikeSCT(epoch).Y=Y;
            %RESULT(kfold).spikeSCT(epoch).YS = spikeSampleCountTest*M;
            %RESULT(kfold).RM(epoch)=robustnessMetric;
            if(epoch==Nepochs);RESULT(kfold).RO=RO;end;
            
            s=sprintf('\r\n Accuracy : kFold(%i) Epoch(%i) Test %2.2f (%i/%i) Train:%2.2f (%i/%i) #dW : (%i) \t',kfold,epoch,accuracyTest,numCorrectTest,length(testOn),accuracyTrain,numCorrectTrain,length(trainOn),numWchanges);
            fprintf(s);
            if(LOGTXT);fprintf(LogFile,s);end
        end
        % Save Nfold Results for each
        
        RESULT(kfold).M=M;
        RESULT(kfold).CM = CM;
        RESULT(kfold).trainOn =  trainOn;
        RESULT(kfold).testOn= testOn;
    end
    
    %% Save params
   saveRES(savefilename,res_iter,PARAM1,PARAM2,RESULT);
    fprintf(['RESULT' num2str(res_iter)  'file saved \r\n']);
end


