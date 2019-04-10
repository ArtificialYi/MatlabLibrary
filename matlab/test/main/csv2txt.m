function [fTrainTxt] = csv2txt(fileCsv, fileTxt, rowBegin)
%csv2txt 将csv文件转成txt文件

fTrainCsv = fopen(fileCsv);
fTrainTxt = fopen(fileTxt, 'w');
indexLine = 1;

while ~feof(fTrainCsv)
    line = fgetl(fTrainCsv);
    fprintf('第%d行读取成功\n', indexLine);
    if indexLine >= rowBegin
        fprintf(fTrainTxt, "%s\n", line);
        fprintf('第%d行写入成功\n', indexLine);
    end
    indexLine=indexLine+1;
end

fclose(fTrainTxt);
fclose(fTrainCsv);

end

