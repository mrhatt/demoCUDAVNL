function [] = DisplayOptimization(varargin)

if isempty(varargin)
    optimizer=1;
    x=50;
    y=50;
elseif length(varargin)==1
    optimizer=varargin{1};
    x=50;
    y=50;
elseif length(varargin)==2
    optimizer=varargin{1};
    x=varargin{2};
    y=50;
else 
    optimizer=varargin{1};
    x=varargin{2};
    y=varargin{3};
end

load I.mat;
fid = fopen('img.bin','w');fwrite(fid,I,'*single');fclose(fid);

%[status,result] = system('demo 1');  
[status,result] = system(['./demo ',num2str(optimizer),' ',num2str(x),' ',num2str(y)]);  %Amoeba
if status ~=0
    error('demo program had an error');
end
result

f = fopen('history.bin');
history = fread(f,inf,'*single');
fclose(f);

history = reshape(history,length(history)/3,3);

f = fopen('img.bin');
img = fread(f,512*512,'*single');
fclose(f);
img =reshape(img,512,512);
img = img';

figure(1)
for k=1:size(history,1)
    subplot(1,2,1)
    imshow(img,[]);
    hold on;
    plot(history(1:k,1),history(1:k,2),'r-.')
    plot(history(1,1),history(1,2),'rs')
    plot(history(k,1),history(k,2),'ro')
    hold off;
    subplot(1,2,2)
    plot(history(:,3),'-');
    hold on;
    plot(k,history(k,3),'o');
    hold off;
    set(gcf,'color','w')
    pause(.01)
    
end
