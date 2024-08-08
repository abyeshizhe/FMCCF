clear all
clc

addpath(genpath('data'))
addpath(genpath('fans'))

%% 读入数据
load BBCSport.mat      

X = data;
Y = truelabel{1};


for nn = 1:length(X)
    [n,m] = size(X{nn});
    X{nn} = mapstd(X{nn}');
    X{nn}=double(X{nn});
end

%% 主程序
lambadlist = [1e-7];
RESULT=[];
OBJ_ALL = [];
for iter_lambad = 1:length(lambadlist)
    iter_lambad
    ACC=[];NMI=[];Purity=[];T=[];OBJ=[];ARI=[];FF=[];PP=[];RR=[];
    for average_iter = 1:1
        
        viewcell=X;
        clear X
        viewmat=[];
        K=length(viewcell);
        for i=1:K
            viewcell{i}=double(viewcell{i});
            viewmat = [viewmat viewcell{i}];  
        end
        [n,~] = size(viewmat);
        s=length(unique(Y'));  % s为数据类的个数
        m=s+25;
        tic
        %% k-means
        [label, cluster_centers] = litekmeans(viewmat, m);
        
        %% Get B
        Bcell=[];temp1=[];
        for i=1:K
            temp0=size(viewcell{i},2);
            temp1=[temp1,temp0];
        end
        cluster_centers_cell=mat2cell(cluster_centers,m,temp1);     % 此处将cluster_centers按照原始数据进行划分
        for ii=1:K
            Bcell{ii}=ConstructA_NP(viewcell{ii}',cluster_centers_cell{ii}');
            Bcell{ii}=sparse(Bcell{ii}');           
            S{ii}=sparse(Bcell{ii}*Bcell{ii}');          
            Diag_matrix{ii}=diag(sum(S{ii},2));
            Laplace{ii}=Diag_matrix{ii}-S{ii};
            Bcell{ii}=Bcell{ii}';
        end
        clear Diag_matrix S cluster_centers_cell viewcell
    
        %% 初始化
        Z=zeros(n,m);
        L=zeros(m,m);
        alphaV1=[];J=[];
        alphaV=1/K*ones(1,K);
        for i=1:K
            Z=Z+alphaV(i)*Bcell{i};
            L=L+alphaV(i)*Laplace{i};
        end
        F=randn(m,s);
        G=initializeG(m,s);    
        temp0=(Z-Z*F*G').^2;
        sigma=1e0*sqrt(sum(sum(temp0)/(2*m)));      
    %     sigma = 0.1;
        lambad = lambadlist(iter_lambad);
        OBJ = [];
        for iter = 1:30
            % 更新W
            temp1 = (Z-Z*F*G') .^2;
            temp2 = -(sum(temp1,2)) ./ (2*sigma^2);
            W = sparse(diag(exp(temp2) ./ (sigma^2)));
            
            % update F
             V = Z'*W*Z;
             F = G;
             F(F<0)=0;
             SSS = 2*(1e3*eye(m)-L)*G + (2/lambad)*V'*F;
             [UU,SS,WW] = svd(SSS,'econ');
             G = UU * WW';
            
            % update alpha
            AA = eye(m)-F*G';
            Btemp={};B0=[];L0=[];
            for i=1:K
                Btemp{i}=(sqrt(W)*Bcell{i}*AA)';
                Btemp=reshape(Btemp{i},m*n,1);
                Ltemp=reshape(Laplace{i},m*m,1);
                B0=[B0 Btemp];
                L0=[L0 Ltemp];
                clear Btemp;
            end
            B=B0'*B0;
            temp5 = reshape(G*G',m*m,1);
            b = lambad*temp5' * L0;
            [alphaV, val,p] = SimplexQP_ALM(B, b, 1e-3,1.05,1);
            
            %% compute objective function
            Z = zeros(n,m);
            L = zeros(m,m);
            for i=1:K
                Z=Z+alphaV(i)*Bcell{i};
                L=L+alphaV(i)*Laplace{i};
            end
            obj = trace((Z-Z*F*G')'*W*(Z-Z*F*G')) + lambad*trace(G'*L*G);
            OBJ = [OBJ,obj];
        end
        final = Z*F;
        [maxv,ind]=max(final,[],2);
        t=toc;
        out = ClusteringMeasure(Y', ind);
        ACC=[ACC, out(1)];
        NMI=[NMI, out(2)];
        Purity=[Purity, out(3)];
        ARI=[ARI,out(4)];FF=[FF,out(5)];PP=[PP,out(6)];RR=[RR,out(7)];
        T=[T t];
    end
    result=[mean(ACC),mean(NMI),mean(Purity),mean(ARI),mean(FF),mean(PP),mean(RR),mean(T);
            std(ACC),std(NMI),std(Purity),std(ARI),std(FF),std(PP),std(RR),std(T)];
    RESULT=[RESULT;result];
end



