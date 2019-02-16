%% Compare GridLSM results
% Poisson
srcpath = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\Replicate\res\2Dgrid\sweep1';tagname= {'LSM2DgridS','LSM3DgridS'};%,'LSM3DrandomS'};
% TI46
%srcpath = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\Replicate\res\2Dgrid\TIsweep1';tagname= {'TI46LSM2DgridS','TI46LSM3DgridS','TI46LSM3DrandomS'};
Nsim = 40;
alphaG=[1E-10 linspace(0.1,20,Nsim)]; NG=length(alphaG);
accTest=[];A=[];
for tagid=1:length(tagname)
    for iter=1:numel(alphaG)
    R=load([srcpath filesep tagname{tagid} '_' num2str(iter) '.mat']);%R=R.R;
    DATA=load([srcpath filesep tagname{tagid} '_' num2str(iter) '_DATA.mat']);DATA=DATA.DATA;
    for kfold =1:R.PARAM1.Nfold
        A(iter,tagid,:,kfold)=R.RESULT(kfold).accTest;
    end
    end
end
% accTest=[];
% for tagid=1:3
%     for iter=1:numel(alphaG)
%     R=load([srcpath filesep tagname{tagid} '_' num2str(iter) '.mat']);%R=R.R;
%     DATA=load([srcpath filesep tagname{tagid} '_' num2str(iter) '_DATA.mat']);DATA=DATA.DATA;
%      acc=R.RESULT(1).accTest;
%         for i = 2
%         acc=acc+R.RESULT(i).accTest;
%         end
%         accTest(:,iter,tagid)=acc/i;
%     end
% end

%% Include variance from last 20 epochs, plot avg accuracy over last 20 epochs and all kfold
Nfold=R.PARAM1.Nfold;
accTest=mean(mean(A(:,:,end-19:end,:),4),3);  accTestStd=std(reshape(A(:,:,end-19:end,:),NG,numel(tagname),20*Nfold),[],3);% variance over last 20 epochs

bandcolor=[0 0 1 0.3; 1 0 0 0.3; 1 0.64 0 0.3]; 
h=[];hold off; 
for tagid=1:length(tagname)
    x = alphaG; y =100-accTest(:,tagid)'; dy=accTestStd(:,tagid)'; 
    fill([x flip(x)],[y-dy flip(y+dy)],bandcolor(tagid,1:3),'FaceAlpha',bandcolor(tagid,4),'linestyle','none'); hold on;
    h(tagid)=line(x,y,'LineWidth',2,'Marker','o','Color',bandcolor(tagid,1:3));xlabel('\alpha_w');ylabel('Error(%)');
    set(gca,'YScale','log');set(gca,'XScale','linear');%ylabel('{\tau_M}(ms)'); ylabel('{k\cdot\lambda}');%ylim([-2 1]);
    
    legend(tagname,'location','sw');
    pbaspect([1 1 1]);
    set(findobj(gcf,'type','axes'),'FontName','Consolas','FontSize',14,'FontWeight','Bold', 'LineWidth', 1);
    col='w';set(gcf,'Color',col);set(gca, 'Color',col);set(findobj(gcf, 'Type', 'Legend'),'Color',col);
end
legend(h,{'2D grid LSM','3D grid LSM','3D LSM'});
drawnow;


