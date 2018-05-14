% Files=dir('*.*');
% for k=1:length(Files)
%   FileNames=Files(k).name
% end
% folderPath1 = actualFile folder
folderPath1 = 'E:\RUG\1-2\HandWriting Recognition\Segment';
cd(folderPath1); % path of the folder
% WriteDir = WriteFile Folder
WriteDir = 'E:\RUG\1-2\HandWriting Recognition\NewSeg';
files1 = dir('*.*')
files1(1:2) = [];
% totalFiles = numel(files1);
totalFiles = length(files1)
for i =1:totalFiles
    Fileaddress{i,1}=strcat(folderPath1,'\',files1(i).name);
    file{i} = imread(Fileaddress{i,1});
    % Edit the file
    file{i}(33:40,:)=0;
    file{i}(153:160,:)=0;
    cd(WriteDir) % go to dir where you want to save updated files
    writeFileName = strcat('TB_0',num2str(i),'.png');
    imwrite(file{i},writeFileName)
    cd(folderPath1) % return to actualFile folder
end