addpath ../Replicate/;

%%
SAVE_VIDEO =0;
RESERVOIR = 1;
LOGTXT = 0;
SYNAPSE_ORDER= 2;
RESET_VAR_PER_SAMPLE =0; 
LOAD_EXISTING_INPUT = 0;
W_INIT_RANDOM = 0;
OPTIMAL_WEIGHTS =0;
MODIFIED_WT_UPDATE = 0;
PROBABILISTIC_WT_UPDATE = 1;
PREPROCESSING=1;
REVERSE_PRUNE = 2;% 1- Reverse Prune, 0 :forward prune, 2 Random Prune % change FracResIn for control
RESERVOIR_GRID=1; % if yes, then reservoir size must be 2D % 2 : 3D grid reservoir % old term : RESERVOIR_2DGRID
RNG.option = 'default'; rng(RNG.option); % default/shuffle
RNG.rng = rng; % store rng object for reference
GET_DIST = 1; % overall distribution of spikes in reservoir( stored in param)
DATASET= 'TI46-IF.mat';%DATASET= 'PoissonJittered_jitter16_lambda40.mat';%;%DATASET= 'digitdata.mat';
%% save file names
savefilename = mfilename;

%% PARAM1
%Input Parameters
fs = 12E3; out_fs = 1000; df = 24; earQ = 4; stepfactor =5/32; differ=1; agcf=1; tauf=2; max_AGC = 4E-4; BSAtau =4E-3; BSAtau2 = 1E-3;BSAfilterFac = 1;
appendS = 50;
% Neuron Parameters
tauV = 32E-3; tauC = 64E-3; tauv1e = 8E-3; tauv2e = 4E-3; tauv1i = 4E-3; tauv2i = 2E-3; RefracPeriod = 2E-3; Vth = 20;
% Reservoir Parameters %3D reservoir
resSize = [11 11 1]; Wres = [3 6; -2 -2]; r0 = 2; Kres = [0.45 0.3;0.6 0.15]; f_inhibit = 0.2; GinMag = 8; InputFanout = 4;
% Classifier Parameters
Nout = 10; dW0 = 0.01; Wlim = 8; Cth = 5; DeltaC= 3   ; dC =1; Iinf = 10000; Cfactor = 1; probN = 1/Nout; probP = (Nout-1)/Nout;
dt = 1E-3;
% Training Parameters
Nfold = 2; Nepochs = 50; samplesPerSpeakerPerClass=10; numSpeakers=5; samplesPerClass = numSpeakers*samplesPerSpeakerPerClass;
 fracResIn=1;
% END PARAM1s


%% Sweep Parameters
Wds=[];
Nsim=40;
alphaG=[1E-10 linspace(0.1,20,Nsim)];
%alphaG=[0.1:0.2:1 1.2:0.2:5 6:0.5:15];
%alphaG = 1E-10;
Nsim = numel(alphaG);
ALPHAG=repmat(reshape(alphaG,1,1,Nsim),2,2);
Wds=repmat([3 6;-2 -2],1,1,Nsim).*ALPHAG;

