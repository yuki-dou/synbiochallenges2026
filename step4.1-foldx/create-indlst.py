import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
csv_path = BASE_DIR / 'step3-interim-scoring' / 'top-50-candidates.csv'

with open(csv_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    new_mutations_list = []
    for row in reader:
        new_mutations = []
        mutations = row['Mutations'].split(';')
        for mut in mutations:
            number = int(''.join([n for n in mut if n.isdigit()]))
            if number < 64:
                number +=1
            if number > 64:
                number+=3
            new_mut = mut[0] + 'A' + str(number) + mut[-1]
            new_mutations.append(new_mut)
        new_mutations_str = ','.join(new_mutations) + ';' + '\n'
        new_mutations_list.append(new_mutations_str)

with open('individual_list.txt', 'w') as file:
    file.writelines(new_mutations_list)

