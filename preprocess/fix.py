import csv

input_file = "../dataset/wmt14_translate_de-en_train.csv"
output_file = "../dataset/wmt14_translate_de-en_train_fixed.csv"

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        # print(row)
        # 确保每行只有两列（en和de）
        if len(row) == 2:
            if row[0] != '' and row[1] != '':
                writer.writerow(row)
        else:
            print(len(row))
            # print(row)