from struct import unpack;
from numpy import reshape;

def readhtkfiles(filepaths):
    
    totFrames = 0;
    for filepath in filepaths:
        nframes, nfeat = readhtkdim(filepath);
        totFrames += nframes;
        
    features = numpy.zeros((totFrames, nfeat));
    
    startFrame = 0;
    for filepath in filepaths:
        feat = readhtk(filepath);
        features[startFrame:startFrame+feat.shape[0],:] = feat;
        startFrame = startFrame+feat.shape[0];


def readhtk(file):
    
    fid = open(file, "rb")
    
    nf = unpack(">i", fid.read(4))[0];
    fp = unpack(">i", fid.read(4))[0]*1e-7;
    by = unpack(">h", fid.read(2))[0];
    tc = unpack(">h", fid.read(2))[0];
    
    n_features = by/4;
    fmt = "f" * int(nf * n_features);
    fmt = ">" + fmt;
    feat = unpack(fmt, fid.read(int(n_features*nf*4)));
    
    fid.close();
    
    feat = reshape(feat, (nf, n_features));
    
    return feat;

def readhtkdim(file):
    
    fid = open(file, "rb")
    
    nf = unpack(">i", fid.read(4))[0];
    fp = unpack(">i", fid.read(4))[0]*1e-7;
    by = unpack(">h", fid.read(2))[0];
    tc = unpack(">h", fid.read(2))[0];
    
    n_features = by/4;
    
    return (nf, n_features);