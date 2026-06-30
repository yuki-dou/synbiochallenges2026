with open('2B3P_clear.pdb', 'r') as f:
    new_lines = []
    for line in f:
        if line.startswith('HETATM'):
            new_line = line.replace('HETATM', 'ATOM  ')
            new_lines.append(new_line)
        else:
            new_lines.append(line)

with open('2B3P_new.pdb', 'w') as f:
    f.writelines(new_lines)