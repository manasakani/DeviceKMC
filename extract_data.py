import re
import sys
import scipy.io as sio

def extract_data(file_path):

    # with open(file_path, 'rt') as file_path:
    #     lines = file.readlines()

    kmc_times = []
    vacancies_list = []
    vacanciesCharged = []
    a = 0
    
    with open(file_path, "rt") as myfile:
        for line in myfile:
            if 'KMC' in line.split() and 'step' in line.split():
                a = float(line.split()[-1])
                
            if a % 10 == 0 and a <= 10000060:
                #print(a)
                if 'Global' in line.split() and 'oxygen' not in line.split():
                    en = float(line.split()[-1])
                    vacancies_list.append(en)

                if 'P_diss:' in line.split() and 'oxygen' not in line.split():
                    vacanciesCharged.append(float(line.split()[-1]))

                if 'KMC' in line.split() and 'time' in line.split() and 'reached' not in line.split():
                    kmc_times.append(float(line.split()[-1]))
   # for i in range(len(lines)):
    #    if lines[i].startswith('KMC step count:'):
         #step_count = int(re.findall(r'\d+', lines[i])[0])
      #      if step_count <= 4900 and step_count % 100 == 0:
       #         kmc_time_match = re.search(r'KMC time is: (\d+\.\d+e[-+]?\d+)', lines[i])
        #        if kmc_time_match:
          #          kmc_time = float(kmc_time_match.group(1))
           #         vacancies = int(re.search(r'# Uncharged vacancies: (\d+)', lines[i]).group(1))
            #        kmc_times.append(kmc_time)
             #       vacancies_list.append(vacancies)
            elif a > 7160000000:
                break

    return kmc_times, vacancies_list, vacanciesCharged

def save_data(kmc_times, vacancies, vacanciesCharged):
    sio.savemat('temperature.mat', {'temperature' : vacanciesCharged})
    sio.savemat('times.mat', {'kmc_times': kmc_times})
    sio.savemat('power.mat', {'power': vacancies})

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_data.py output.log")
        return

    file_path = sys.argv[1]
    kmc_times, vacancies, vacanciesCharged = extract_data(file_path)
    save_data(kmc_times, vacancies, vacanciesCharged)

if __name__ == "__main__":
    main()

