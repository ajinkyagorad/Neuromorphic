%% Compare GridLSM results

srcpath = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\Replicate\res\2Dgrid\TIsweep1';
tagname= {'TI46LSM2DgridS','TI46LSM3DgridS','TI46LSM3DrandomS'};
Nsim = 40;
alphaG=[1E-10 linspace(0.1,20,Nsim)]; NG=length(alphaG); alphaG=alphaG(1:end-1);
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
%% Show graph
tagid=3;iter=1;
R=load([srcpath filesep tagname{tagid} '_' num2str(iter) '.mat']);%R=R.R;
[X,Y,Z]=ndgrid(1:5);E=R.PARAM2.E;G=R.PARAM2.G';h=plot(digraph(G),'NodeColor',[E<0 zeros(size(E)) E>0],'XData',X(:),'YData',Y(:),'ZData',Z(:),'ArrowSize',8)
h.EdgeCData=G(G~=0);h.LineWidth=2;h.MarkerSize=5;h.LineWidth=1;
daspect([1 1 1]);axis off;caxis(2*[-2 3]*1E-10);colormap([cmap('Crimson',100,50,0);flipud(cmap('Navy',100,50,0))]);
set(findobj(gcf,'type','axes'),'FontName','Consolas','FontSize',14,'FontWeight','Bold', 'LineWidth', 1);
col='none';set(gcf,'Color',col);set(gca, 'Color',col);set(findobj(gcf, 'Type', 'Legend'),'Color',col);
drawnow;
%% Include mean and  variance from last 20 epochs
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
