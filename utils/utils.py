# -*- coding: utf-8 -*-     #aggiunto perche altrimenti sul cluster dava problemi di codifica del testo
import os;
import numpy;
from sklearn.externals import joblib;

def readfeatures(basePath, filenames):

    count = 0;
    for filename in filenames:
        feat = joblib.load(os.path.join(basePath, os.path.basename(filename)));
        if (count == 0):
            features = numpy.zeros((len(filenames), feat.shape[0] * feat.shape[1]));
        features[count] = feat.reshape((1, feat.shape[0] * feat.shape[1]));
        count += 1;

    return features;

def readlistfile(filename,labelling):#labellig : nclass-biclass
    classes   = dict();
    distances = dict();
    filepaths = list();
    classLabels = list();
    distLabels  = list();
    nClasses    = 0;
    nDistances  = 0;
    count=0; # mi serve per la classe bck (noise)

    with open(filename, "r") as f:
        for line in f:

            filename = os.path.basename(line);
            elements = filename.split("_");
            className = elements[0];
            occ = int(elements[len(elements)-1][0]); # occ = numero dell occorrenza
            if (className=="rndy"): #se il file contiene queste denominazioni significa che una classe di cadute "fall"
                className = "fall";
            else:
                if (labelling=="biclass"):#classificatore a 2 classi: tutto ciò che non è fall
                    className = "nofall";

            if (not className in classes): #conta il numero delle diverse classi (fall,fork,book,etc...)
                classes[className] = nClasses;
                #print(className);
                #print(nClasses)
                nClasses += 1;

            classLabel = classes[className];

            dist = "";

            #divido in 4 gruppi le features delle cadute dalla sedia. Lo faccio perchè esse hanno solo 2 "distazne" e non 4
            if(className=="fall" and elements[2]=="chair"):
                if (elements[1] == "d4st") and (occ%2==0):
                    dist="d1";
                elif (elements[1] == "d4st") and (occ%2==1):
                    dist="d2";
                if (elements[1] == "d6st") and (occ%2==0):
                    dist="d4";
                elif (elements[1] == "d6st") and (occ%2==1):
                    dist="d6";



            #divido in 4 gruppi le features delle cadute da in piedi. Lo faccio perchè esse hanno solo 3 "distazne" e non 4
            elif(className=="fall" and elements[2]!="chair"):
                if (occ==0):
                    dist="d1";
                elif elements[1] == "d2st":
                    dist="d2";
                elif elements[1] == "d4st":
                    dist="d4";
                elif elements[1] == "d6st":
                    dist="d6";

#            elif (className == "chair"):
#                if elements[1] == "d1h0" and (elements[2] == "back" or elements[2] == "front"):
#                    dist = "d1";
#                elif (elements[1] == "d1h0" and elements[2] == "side") or (elements[1] == "d2h0" and elements[2] == "back"):
#                    dist = "d2";
#                elif elements[1] == "d2h0" and (elements[2] == "front" or elements[2] == "side"):
#                    dist = "d4";
#                elif elements[1] == "d4h0":
#                    dist = "d6";

            elif(className=='noise'): #assegno distanze fittizie ai file noise_xx.htk anche se in realtà non anno questo label associato.
                if(count==0):
                    dist="d1";
                    count=1;
                if(count==1):
                    dist="d2";
                    count=2;
                if(count==2):
                    dist="d4";
                    count=3;
                if(count==3):
                    dist="d6";
                    count=0;


            else:
                dist = elements[1][0:2]; #i primi 2 caratteri di element[1]

            #commentato tanto non serve
#            if (elements[1] == "d3st"):
#                dist = "d4";
#            elif (elements[1] == "d4st"):
#                dist = "d6";

            if (not dist in distances):
                distances[dist] = nDistances;
                #print("dist: "+dist)
                nDistances += 1;


            distLabel = distances[dist];
            filepath = line.rstrip(); #leva gli spazi finali
            #print(filepath);
            #if (line[0] == "/"):
            #   filepath = "g:" + line.rstrip();
            filepaths.append(filepath);
            classLabels.append(classLabel);
            distLabels.append(distLabel);   #assego ad ogni file il label secondo la distanza (che in realtà non è la distanza perche faccio assegnamenti come mi fanno comodo vedi sopra) e creo un vettore lungo 288 (vedi righe di files_list_right_sphinx_deltas_cmn) che contiene in ogni elemento il label del file

    #for key in classes:
     #   print(key + "->" + str(classes[key]));


    return (filepaths, classLabels, distLabels);


def labToClass(a):
    b = int
    if a == 'V':
        b=0
    elif a == 'O':
        b=1
    elif a == 'T':
        b=2
    elif a == 'E':
        b=3

    return b