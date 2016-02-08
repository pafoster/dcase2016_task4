# DCASE 2016::Domestic Audio Tagging / Baseline System
# Copyright (C) 2016 Peter Foster (p.a.foster@qmul.ac.uk) / QMUL
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn import metrics
import numpy
import csv

def compute_eer(result_filename, label, label_assignments):
    """Compute the equal error rate (EER) from the plot of the false negative rate 
        versus the false positive rate.
    
    Keyword arguments:
        result_filename -- The CSV file from which to read results.
            Each row in the file is of the form
    
                <filename>,<label>,<score>

            where <filename> is an audio file name, <label> is a label identifier 
            and where score is a classification score about the presence of
            <label> in <filename>.
        label -- The label identifier (as specified in result_filename) for which
        to compute the EER.
        label_assignments -- A dictionary whose keys are file names as contained 
        in result_filename and whose values are ground truth assignments about 
        the presence of the specified label.
    """
            
    results = []
    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter=','):
            if len(row[1]) != 1 or not row[1].isalpha():
                raise ValueError('The label identfier "' + row[1] + '" in row ' + str(row) + ' is not valid.')
            if row[1] == label:
                results.append((row[0], row[1], float(row[2])))
                
    if len(numpy.unique([r[0] for r in results])) != len(results):
        raise ValueError('File ' + result_filename + ' contains duplicate score assignments.')
    if len(set([r[0] for r in results]).symmetric_difference(set(label_assignments.keys()))) != 0:
        raise ValueError('One-to-one mapping between files listed in ' + result_filename + ' and ground truth assignments for label ' + label + ' not satisfied.')
    
    y_true = numpy.array([label_assignments[row[0]] for row in results])
    y_score = numpy.array([row[2] for row in results])
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_score,drop_intermediate=True)
    
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)        
    return EER
