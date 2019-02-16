%% Compare GridLSM results

srcpath = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\Replicate\res\2Dgrid\comparison1';
%tagname= {'LSM2Dgrid','LSM3Dgrid','LSM3Drandom','LSMoptimal'};
tagname= {'LSM3Drandom','LSMoptimal'};

accTest=[];A=[];iter=''; NG=1;
for tagid=1:length(tagname)
    R=load([srcpath filesep tagname{tagid} '.mat']);%R=R.R;
    DATA=load([srcpath filesep tagname{tagid} '_DATA.mat']);DATA=DATA.DATA;
    for kfold =1:R.PARAM1.Nfold
        A(1,tagid,:,kfold)=R.RESULT(kfold).accTest;
    end
    
end
%% %% Plot accuracy vs epochs with variance on k fold
Nfold=R.PARAM1.Nfold;
accTest=mean(mean(A,4),4);  accTestStd=std(reshape(A,NG,numel(tagname),200,Nfold),[],4);% variance over last 20 epochs
accTest= squeeze(accTest)';accTestStd= squeeze(accTestStd)';
bandcolor=[0 0 1 0.3; 1 0 0 0.3; 1 0.64 0 0.3; 0 0.64 0.45 0.3]; 
bandcolor=[1 0.64 0 0.3; 0 0.64 0.45 0.3]; 
h=[];hold off; 
for tagid=1:length(tagname)
    x = 1:200; y =100-accTest(:,tagid)'; dy=accTestStd(:,tagid)'; 
    fill([x flip(x)],[y-dy flip(y+dy)],bandcolor(tagid,1:3),'FaceAlpha',bandcolor(tagid,4),'linestyle','none'); hold on;
    h(tagid)=line(x,y,'LineWidth',2,'Marker','o','Color',bandcolor(tagid,1:3));xlabel('\alpha_w');ylabel('Error(%)');
    set(gca,'YScale','log');set(gca,'XScale','log');%ylabel('{\tau_M}(ms)'); ylabel('{k\cdot\lambda}');%ylim([-2 1]);
    
    legend(tagname,'location','best');
    pbaspect([1 1 1]);
    set(findobj(gcf,'type','axes'),'FontName','Consolas','FontSize',14,'FontWeight','Bold', 'LineWidth', 1);
    col='none';set(gcf,'Color',col);set(gca, 'Color',col);set(findobj(gcf, 'Type', 'Legend'),'Color',col);
end
legend(h,{'2D grid LSM','3D grid LSM','3D LSM','optimal LSM'});
legend(h,{'Zero','Optimal'});title('Initial Weights');
drawnow;


