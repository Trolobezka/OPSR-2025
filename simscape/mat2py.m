function mat2py(path, data)
    subformat = strjoin(repmat({'%.17g'}, 1, size(data, 2)), ', ');
    format = strjoin({'    [', subformat, '],\n'}, '');
    fileID = fopen(path, 'w');
    fprintf(fileID, '# type: ignore\nimport numpy as np\n\ndata = np.array([\n');
    fprintf(fileID, format, data');
    fprintf(fileID, '], dtype=np.float64)');
    fclose(fileID);
    fprintf('Data saved to ''%s''\n', path);
end