#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: andrea
"""

from os import walk, path


def create_result(eval_path, A, UAR, ConfMatrix, recall_report):
    result = path.join(eval_path, 'ComParE2017_Snore.result')
    with open(result, 'w') as f:
        line = "{:.1f}"
        line = line.format(A*100)
        f.write("Accuracy = " + line + "%\n")
        line = "{:.1f}"
        line = line.format(UAR*100)
        f.write("UAR = " + line + "%\n")
        line = "{:.0f}"
        line = line.format(recall_report[0]*100)
        f.write("Recall (V) = " + line + "%\n")
        line = "{:.0f}"
        line = line.format(recall_report[1]*100)
        f.write("Recall (O) = " + line + "%\n")
        line = "{:.0f}"
        line = line.format(recall_report[2]*100)
        f.write("Recall (T) = " + line + "%\n")
        line = "{:.0f}"
        line = line.format(recall_report[3]*100)
        f.write("Recall (E) = " + line + "%\n")
        f.write("Confusion matrix:\n")
        f.write("\t\t V\t O\t T\t E\n")
        f.write("\tV \t" + str(ConfMatrix[0, 0]) + "\t" + str(ConfMatrix[0, 1]) + "\t" + str(ConfMatrix[0, 2]) + "\t" + str(ConfMatrix[0, 3]) + "\n")
        f.write("\tO \t" + str(ConfMatrix[1, 0]) + "\t" + str(ConfMatrix[1, 1]) + "\t" + str(ConfMatrix[1, 2]) + "\t" + str(ConfMatrix[1, 3]) + "\n")
        f.write("\tT \t" + str(ConfMatrix[2, 0]) + "\t" + str(ConfMatrix[2, 1]) + "\t" + str(ConfMatrix[2, 2]) + "\t" + str(ConfMatrix[2, 3]) + "\n")
        f.write("\tE \t" + str(ConfMatrix[3, 0]) + "\t" + str(ConfMatrix[3, 1]) + "\t" + str(ConfMatrix[3, 2]) + "\t" + str(ConfMatrix[3, 3]) + "\n")

def create_pred(eval_path, y_devel_lab, y_devel_lit, output, class_pred):
    prediction = path.join(eval_path, 'ComParE2017_Snore.pred')
    with open(prediction, 'w') as f:
        f.write("\n\n=== Predictions on test data ===\n\n")
        f.write("\tinst#\tactual\tpredicted\terror\tdistribution\n")
        for o in range(len(output)):
            if class_pred[o] == 0:
                str_class_pred = 'V'
            elif class_pred[o] == 1:
                str_class_pred = 'O'
            elif class_pred[o] == 2:
                str_class_pred = 'T'
            elif class_pred[o] == 3:
                str_class_pred = 'E'
            f.write("\t"+str(o+1)+"\t"+str(int(y_devel_lab[o]))+":"+y_devel_lit[o]+"\t"+str(int(class_pred[o]))+":"+str_class_pred)
            stars = ['', '', '', '']
            if class_pred[o] != y_devel_lab[o]:
                f.write("\t\t+\t")
                stars[int(class_pred[o])] = '*'
            else:
                f.write("\t\t \t")
            line = stars[0]+"{:.3f},"+stars[1]+"{:.3f},"+stars[2]+"{:.3f},"+stars[3]+"{:.3f}"+"\n"
            line = line.format(output[o][0],output[o][1],output[o][2],output[o][3])
            f.write(line)
            stars = ['', '', '', '']

def create_arff(eval_path, name_list, output, class_pred):
    arff = path.join(eval_path, 'ComParE2017_Snore.arff')
    with open(arff, 'w') as f:
        f.write("@relation ComParE2017_Deception_Predictions_baseline\n")
        f.write("@attribute instance_name string\n")
        f.write("@attribute prediction { V, O, T, E }\n")
        f.write("@attribute score_V numeric\n")
        f.write("@attribute score_O numeric\n")
        f.write("@attribute score_T numeric\n")
        f.write("@attribute score_E numeric\n")
        f.write("@data\n")
        for o in range(len(output)):
            if class_pred[o] == 0:
                str_class_pred = 'V'
            elif class_pred[o] == 1:
                str_class_pred = 'O'
            elif class_pred[o] == 2:
                str_class_pred = 'T'
            elif class_pred[o] == 3:
                str_class_pred = 'E'
            f.write("'"+name_list[o]+"',"+str_class_pred+",")
            line = "{:.3f},{:.3f},{:.3f},{:.3f}"+"\n"
            line = line.format(output[o][0], output[o][1], output[o][2], output[o][3])
            f.write(line)
