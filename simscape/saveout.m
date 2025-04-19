function saveout(out, path)
    data = [out.state_vars.Time, out.state_vars.Data];
    fprintf('Data shape = %d x %d\n', size(data, 1), size(data, 2));
    mat2py(path, data);
end