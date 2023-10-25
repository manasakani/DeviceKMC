import re
import sys
import scipy.io as sio

def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    kmc_times = []
    temperature = []
    a = 0
    
    with open(file_path, "rt") as myfile:
        for line in myfile:
            if 'KMC' in line.split() and 'step' in line.split():
                a = float(line.split()[-1])
                
            if a % 10 == 0 and a <= 100000:
                #print(a)
                if 'Total' in line.split() and 'temperature' in line.split() and 'reached' not in line.split():
                    en = float(line.split()[-1]);
                    temperature.append(en)

                if 'time' in line.split() and 'is:' in line.split():
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
            elif a > 100000:
                break

    return kmc_times, temperature

def save_data(kmc_times, temperature):
    sio.savemat('times.mat', {'kmc_times': kmc_times})
    sio.savemat('temperature.mat', {'temperature': temperature})

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_data.py output.log")
        return

    file_path = sys.argv[1]
    kmc_times, temperature = extract_data(file_path)
    save_data(kmc_times, temperature)

if __name__ == "__main__":
    main()

