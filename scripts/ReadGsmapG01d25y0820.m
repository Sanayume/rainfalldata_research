%% start
clear;

load mask.mat
filepath='H:\GSMap\GSMapV8\0820';

fileinfo=dirr(filepath);

for i = 1:length(fileinfo)
     filename=strcat(filepath,'\',fileinfo(i).name);
     fid=fopen(filename,'r');
     test1=fread(fid,[3600 1200],'float32','l');
     test=transpose(test1);
     gsmap(:,:,i)=test(61:420,721:1360);
     fclose(fid);
     disp(['Finish: ',fileinfo(i).name]);
     
end

gsmap=gsmap*24;
%% 将0.1°的数据分解成0.05°的数据
gsmap05=zeros(720,1280,length(fileinfo));
%temp05=zeros(720,1280);
for k=1:length(fileinfo)
for i = 1:360
    for j=1:640
        
        gsmap05((i-1)*2+1:(i-1)*2+2,(j-1)*2+1:(j-1)*2+2,k)=gsmap(i,j,k);
           
    end
end
disp(['stage-2:',num2str(k)])
end

%% 将0.05°的数据平均到0.25°的空间
%这里会有个问题：会把边界上好多点个给忽略掉，由于mean函数遇到nan就直接输出nan了。
gsmapd25=zeros(144,256,length(fileinfo));
gsmapdvn25=zeros(144,256,length(fileinfo));%4749days
%load mask.mat;
for k=1:length(fileinfo)
    for i = 1:144
        for j=1:256
            if mask(i,j)==1
                gsmapd25(i,j,k)=nanmean(nanmean(gsmap05((i-1)*5+1:(i-1)*5+5,(j-1)*5+1:(j-1)*5+5,k)));
            else   
                gsmapd25(i,j,k)=nan;
            end
        end
    end

    temp=gsmapd25(:,:,k);
    temp(mask==0)=nan;
    temp(temp<0.01)=0;
    gsmapdvn25(:,:,k)=temp;

    disp(['Finished: ',num2str(k)]);
end
for k=1:length(fileinfo)
    temp=gsmapd25(:,:,k);
    temp(mask==0)=nan;
    temp(temp<0.01)=0;
    gsmapdvn25(:,:,k)=temp;
    disp(['Finished: ',num2str(k)]);
end
gsmapd25y0820=gsmapdvn25;
save gsmapd25y0820.mat gsmapd25y0820;
