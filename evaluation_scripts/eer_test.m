result_filename = 'results_fold1.csv';

for label='bcfmopv'
    
    label_assignments = [];
    label_assignments_filelist = {};
    
    F = fopen('fold1_evaluate.csv', 'r');
    L = fgetl(F);
    while ischar(L)
        S = strsplit(L,',');
        label_assignments_filelist = cat(1,label_assignments_filelist, S{1});
        label_assignments = cat(1, label_assignments, any(strfind(S{2}, label)));
        L = fgetl(F);
    end
    fclose(F);    
    
    EER = compute_eer(result_filename, label, label_assignments, label_assignments_filelist);
    fprintf('Label %s: EER %f\n', label, EER);
end