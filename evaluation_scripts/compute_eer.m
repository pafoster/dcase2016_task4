function EER = compute_eer(result_filename, label, label_assignments, label_assignments_filelist)
% COMPUTE_EER   Compute the equal error rate (EER).
%  Compute the equal error rate (EER) from the plot of the false negative 
%  rate versus the false positive rate.
%
%  Arguments:
%  result_filename -- The CSV file from which to read results.
%    Each row in the file is of the form
%        <filename>,<label>,<score>
%    where <filename> is an audio file name, <label> is a label identifier 
%    and where score is a classification score about the presence of
%    <label> in <filename>.
%  label -- The label identifier (as specified in result_filename) for 
%    which to compute the EER.
%  label_assignments -- An vector whose entries indicate ground truth
%    information about the presence of the specified label.
%  label_assignments_filelist -- A cell array whose entries are the
%    corresponding file names of entries in label_assignments. Together
%    with the contents of result_filename and the label parameter, allows
%    scores to be mapped to label annotations.

% Copyright (C) 2016 Peter Foster (p.a.foster@qmul.ac.uk) / QMUL
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

results = struct();
results.files = {};
results.scores = [];

F = fopen(result_filename, 'r');
L = fgetl(F);
while ischar(L)
    S = strsplit(L,',');
    if length(S{2}) ~= 1 || ~ischar(S{2})
        error(['The label identfier "' S{2} + '" in row ' + S{2} + ' is not valid.']);
    end
    if strcmp(S{2}, label)
        results.files = cat(1, results.files, S{1});
        results.scores = cat(1, results.scores, str2double(S{3}));
    end
    L = fgetl(F);
end
fclose(F);

if length(unique(results.files)) ~= length(results.files)
    error(['File ' result_filename ' contains duplicate score assignments.']);
end
if length(label_assignments) ~= length(label_assignments_filelist)
    error('Lengths of label_assignments and label_assignments_filelist are not equal.');
end
if length(unique(label_assignments_filelist)) ~= length(label_assignments_filelist)
    error('label_assignments_filelist contains non-unique entries.');
end
if ~isempty(setxor(results.files, label_assignments_filelist))
    error(['One-to-one mapping between files listed in ' result_filename ' and ground truth assignments for label ' label ' not satisfied.']);
end

[label_assignments_filelist,I] = sort(label_assignments_filelist);
label_assignments = label_assignments(I);

[results.files, I] = sort(results.files);
results.scores = results.scores(I);

assert(all(strcmp(label_assignments_filelist, label_assignments_filelist)));

[fpr, tpr] = perfcurve(label_assignments, results.scores, 1);

eps = 1E-6;
Points = [0 0; fpr tpr];
i = find(Points(:,1) + eps >= 1 - Points(:,2), 1);
P1 = Points(i-1,:);
P2 = Points(i,:);

%Interpolate between P1 and P2
if abs(P2(1) - P1(1)) < eps
    EER = P1(1);
else
    m = (P2(2)-P1(2)) / (P2(1)-P1(1));
    c = P1(2) - m * P1(1);
    EER = (1-c) / (1+m);
end

