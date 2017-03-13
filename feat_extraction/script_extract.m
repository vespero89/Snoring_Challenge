%PNCC_EXTRACT SCRIPT

out_fold = ('../dataset/PNCC/');
wavname = dir('../wav/*.wav');
NumF = length(wavname(not([wavname.isdir])));

for i = 1:NumF
    filename = wavname(i).name
    input_name = [wavname(i).folder,'/',filename];
    outname = filename(1:end-4);
    outname = [out_fold,outname];
    PNCC_extract(input_name,outname);
end