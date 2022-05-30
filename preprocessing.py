import csv
import pandas as pd

def make_dataset(file_name):
    df = pd.read_csv(f'./{file_name}.txt', sep='\t', encoding='utf-8-sig')
    csv_path = f'./{file_name}.csv'

    df['prefix'] = 'eng to kor'
    header = ['prefix', 'input_text', 'target_text']

    df_to_lst = df[header].values.tolist()
    
    with open(csv_path, "a", newline="", encoding='utf-8-sig') as f:
        write = csv.writer(f, delimiter='\t')
        write.writerow(header)

    for data in df_to_lst:
        with open(csv_path, "a", newline="", encoding='utf-8-sig') as f:
            write = csv.writer(f, delimiter='\t')
            write.writerow([data[0], data[1], data[2]])
            write.writerow(['kor to eng', data[2], data[1]])

if __name__ == "__main__":
    file_name = str(input("file Name : "))

    make_dataset(file_name)
