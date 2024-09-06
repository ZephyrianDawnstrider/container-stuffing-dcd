import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import streamlit as st
from datetime import datetime

# Function to load data from a CSV or Excel file with different encodings
def load_data(file, file_type):
    if file_type == 'csv':
        for encoding in ['utf-8', 'latin-1', 'utf-16']:
            try:
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to load CSV file with supported encodings.")
    elif file_type == 'excel':
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

# Function to convert weight from kilograms to metric tons
def convert_weights(data):
    weight_column = 'WEIGHT' if 'WEIGHT' in data.columns else 'Weight'
    data['WEIGHT_TONS'] = data[weight_column] / 1000
    return data

# Function to find the column name that contains 'Job No' or similar
def find_job_no_column(data):
    for col in data.columns:
        if 'job' in col.lower():
            return col
    raise ValueError("No column related to 'Job No' found in the DataFrame.")

# Function to calculate the cost based on weight and volume
def calculate_cost(weight, volume):
    #dcd costing
    if 10 <= weight <= 15 and 45 <= volume <= dcd_upper_limit:
        return 103000  # 15 tons
    elif 4.5 <= weight <= 6 and 45 <= volume <= dcd_upper_limit:
        return 78000  # 6 tons
    elif 7 <= weight <= 9 and 45 <= volume <= dcd_upper_limit:
        return 90000  # 9 tons
    elif 16 <= weight <= 18 and 45 <= volume <= dcd_upper_limit:
        return 113000  # 18 tons
    elif 20 <= weight <= 24 and 45 <= volume <= dcd_upper_limit:
        return 145000  # 24 tons
    else:
        return 0  # Return 0 for cases that don't match any condition

# Function to get volume based on selected volume class
def get_volume_class(volume_class, dcd_upper_limit):
    if volume_class == 'DCD':
        return dcd_upper_limit  # Use the user-defined upper limit for DCD
    else:
        return 0  # Return 0 for unknown volume classes

# Function to calculate days remaining
def calculate_days_remaining(data):
    port_days = {
        # Hong Kong
        'HKG': 27, 'HONGKONG': 27, 'HONG_KONG': 27, 'hkg': 27, 'hongkong': 27, 'hong_kong': 27, 
        'Hkg': 27, 'Hongkong': 27, 'Hong_Kong': 27, 'hong kong': 27, 'Hong Kong': 27,

        # Shanghai
        'SHA': 29, 'SHANGHAI': 29, 'SHANG_HAI': 29, 'sha': 29, 'shanghai': 29, 'shang_hai': 29, 
        'Sha': 29, 'Shanghai': 29, 'Shang_Hai': 29,

        # Shenzhen
        'SZX': 23, 'SHZ': 23, 'SHENZHEN': 23, 'SHEN_ZHEN': 23, 'szx': 23, 'shz': 23, 
        'shenzhen': 23, 'shen_zhen': 23, 'Szx': 23, 'Shz': 23, 'Shenzhen': 23, 'Shen_Zhen': 23,

        # Ningbo
        'NIN': 26, 'NINGBO': 26, 'NING_BO': 26, 'nin': 26, 'ningbo': 26, 'ning_bo': 26, 
        'Nin': 26, 'Ningbo': 26, 'Ning_Bo': 26,

        # Qingdao
        'QIN': 34, 'QINGDAO': 34, 'QING_DAO': 34, 'qin': 34, 'qingdao': 34, 'qing_dao': 34, 
        'Qin': 34, 'Qingdao': 34, 'Qing_Dao': 34,

        # Guangzhou
        'CAN': 22, 'GUANGZHOU': 22, 'GUANG_ZHOU': 22, 'can': 22, 'guangzhou': 22, 'guang_zhou': 22, 
        'Can': 22, 'Guangzhou': 22, 'Guang_Zhou': 22,

        # Tianjin
        'TSN': 28, 'TIANJIN': 28, 'TIAN_JIN': 28, 'tsn': 28, 'tianjin': 28, 'tian_jin': 28, 
        'Tsn': 28, 'Tianjin': 28, 'Tian_Jin': 28,

        # Xiamen
        'XMN': 24, 'XIAMEN': 24, 'XIA_MEN': 24, 'xmn': 24, 'xiamen': 24, 'xia_men': 24, 
        'Xmn': 24, 'Xiamen': 24, 'Xia_Men': 24,

        # Dalian
        'DLC': 33, 'DALIAN': 33, 'DA_LIAN': 33, 'dlc': 33, 'dalian': 33, 'da_lian': 33, 
        'Dlc': 33, 'Dalian': 33, 'Da_Lian': 33,

        # Fuzhou
        'FOC': 21, 'FUZHOU': 21, 'FU_ZHOU': 21, 'foc': 21, 'fuzhou': 21, 'fu_zhou': 21, 
        'Foc': 21, 'Fuzhou': 21, 'Fu_Zhou': 21,

        # Zhuhai
        'ZUH': 20, 'ZHUHAI': 20, 'ZHU_HAI': 20, 'zuh': 20, 'zhuhai': 20, 'zhu_hai': 20, 
        'Zuh': 20, 'Zhuhai': 20, 'Zhu_Hai': 20,

        # Shekou
        'SHEKOU': 25, 'SHE_KOU': 25, 'Shekou': 25, 'She_Kou': 25, 'shekou': 25, 'she_kou': 25,

        # Yantian
        'YTN': 24, 'YANTIAN': 24, 'YAN_TIAN': 24, 'ytn': 24, 'yantian': 24, 'yan_tian': 24, 
        'Ytn': 24, 'Yantian': 24, 'Yan_Tian': 24,

        # Other Chinese Minor Ports
        'NINGDE': 26, 'NING_DE': 26, 'ningde': 26, 'ning_de': 26, 'Ningde': 26, 'Ning_De': 26,
        'JIANGYIN': 30, 'JIANG_YIN': 30, 'jiangyin': 30, 'jiang_yin': 30, 'Jiangyin': 30, 'Jiang_Yin': 30,
        'CHIWAN': 22, 'CHI_WAN': 22, 'chiwan': 22, 'chi_wan': 22, 'Chiwam': 22, 'Chi_Wan': 22,
        'ZHANJIANG': 35, 'ZHAN_JIANG': 35, 'zhanjiang': 35, 'zhan_jiang': 35, 'Zhanjiang': 35, 'Zhan_Jiang': 35,
        'WEIHAI': 32, 'WEI_HAI': 32, 'weihai': 32, 'wei_hai': 32, 'Weihai': 32, 'Wei_Hai': 32,
        'LIANYUNGANG': 31, 'LIAN_YUN_GANG': 31, 'lianyungang': 31, 'lian_yun_gang': 31, 
        'Lianyungang': 31, 'Lian_Yun_Gang': 31,

        # Major Japanese Ports
        'KIX': 21, 'OSAKA': 21, 'KIX_OSAKA': 21, 'kix': 21, 'osaka': 21, 'kix_osaka': 21, 
        'Kix': 21, 'Osaka': 21, 'KIX_OSAKA': 21, 
        'HND': 18, 'TOKYO': 18, 'HND_TOKYO': 18, 'hnd': 18, 'tokyo': 18, 'hnd_tokyo': 18, 
        'Hnd': 18, 'Tokyo': 18, 'HND_TOKYO': 18,
        'NRT': 19, 'NARITA': 19, 'NRT_NARITA': 19, 'nrt': 19, 'narita': 19, 'nrt_narita': 19, 
        'Nrt': 19, 'Narita': 19, 'NRT_NARITA': 19,
        'HIA': 25, 'HIROSHIMA': 25, 'HIA_HIROSHIMA': 25, 'hia': 25, 'hiroshima': 25, 'hia_hiroshima': 25,
        'Hia': 25, 'Hiroshima': 25, 'HIA_HIROSHIMA': 25,

        # Major South Korean Ports  
        'ICN': 20, 'SEOUL': 20, 'ICN_SEOUL': 20, 'icn': 20, 'seoul': 20, 'icn_seoul': 20, 
        'Icn': 20, 'Seoul': 20, 'ICN_SEOUL': 20, 
        'BUS': 23, 'BUSAN': 23, 'BUS_BUSAN': 23, 'bus': 23, 'busan': 23, 'bus_bus': 23, 
        'Bus': 23, 'Busan': 23, 'BUS_BUSAN': 23,
        'PKG': 21, 'PUSAN': 21, 'PKG_PUSAN': 21, 'pkg': 21, 'pusan': 21, 'pkg_pusan': 21, 
        'Pkg': 21, 'Pusan': 21, 'PKG_PUSAN': 21,
        'GMP': 22, 'GIMPO': 22, 'GMP_GIMPO': 22, 'gmp': 22, 'gimpo': 22, 'gmp_gimpo': 22, 
        'Gmp': 22, 'Gimpo': 22, 'GMP_GIMPO': 22,

        # Major Southeast Asian Ports
        'SIN': 15, 'SINGAPORE': 15, 'SIN_SINGAPORE': 15, 'sin': 15, 'singapore': 15, 'sin_singapore': 15, 
        'Sin': 15, 'Singapore': 15, 'SIN_SINGAPORE': 15,
        'BKK': 30, 'BANGKOK': 30, 'BKK_BANGKOK': 30, 'bkk': 30, 'bangkok': 30, 'bkk_bangkok': 30, 
        'Bkk': 30, 'Bangkok': 30, 'BKK_BANGKOK': 30,
        'KUL': 28, 'KUALA LUMPUR': 28, 'KUL_KUALA_LUMPUR': 28, 'kul': 28, 'kuala_lumpur': 28, 'kul_kuala_lumpur': 28, 
        'Kul': 28, 'Kuala_Lumpur': 28, 'KUL_KUALA_LUMPUR': 28,
        'JKT': 26, 'JAKARTA': 26, 'JKT_JAKARTA': 26, 'jkt': 26, 'jakarta': 26, 'jkt_jakarta': 26, 
        'Jkt': 26, 'Jakarta': 26, 'JKT_JAKARTA': 26,
        'MAN': 32, 'MANILA': 32, 'MAN_MANILA': 32, 'man': 32, 'manila': 32, 'man_manila': 32, 
        'Man': 32, 'Manila': 32, 'MAN_MANILA': 32
    }

    current_date = datetime.now().date()
    data['ETD'] = pd.to_datetime(data['ETD'], errors='coerce')
    data['DAYS_REMAINING'] = data.apply(lambda row: port_days.get(row['POL'], 0) - (current_date - row['ETD'].date()).days if pd.notnull(row['ETD']) else 0, axis=1)
    return data
# Function to optimize package selection based on weight and volume constraints
def optimize_packages(data, carry_capacity, carry_volume):
    required_columns = ['WEIGHT_TONS', 'CBM', 'DAYS_REMAINING']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Data must contain columns: {required_columns}")
        return [], 0, 0

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        st.error("Solver creation failed. Ensure that OR-Tools is properly installed.")
        return [], 0, 0

    num_packages = len(data)
    x = [solver.IntVar(0, 1, f'x[{i}]') for i in range(num_packages)]

    solver.Add(solver.Sum(data.loc[i, 'WEIGHT_TONS'] * x[i] for i in range(num_packages)) <= carry_capacity)
    solver.Add(solver.Sum(data.loc[i, 'CBM'] * x[i] for i in range(num_packages)) <= carry_volume)

    objective = solver.Sum(
        (data.loc[i, 'WEIGHT_TONS'] + data.loc[i, 'CBM']) / max(1, data.loc[i, 'DAYS_REMAINING']) * x[i]
        for i in range(num_packages)
    )
    solver.Maximize(objective)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        selected_packages = [i for i in range(num_packages) if x[i].solution_value()]
        total_weight_used = sum(data.loc[i, 'WEIGHT_TONS'] for i in selected_packages)
        total_volume_used = sum(data.loc[i, 'CBM'] for i in selected_packages)
        return selected_packages, total_weight_used, total_volume_used
    else:
        st.error("The solver did not find an optimal solution.")
        return [], 0, 0

# Function to create packages based on optimized selection
def create_packages(data, carry_capacity, carry_volume, include_cost=True):
    packages = []
    unfulfilled_due_to_volume = pd.DataFrame()

    while not data.empty:
        selected_packages, total_weight_used, total_volume_used = optimize_packages(data, carry_capacity, carry_volume)
        if not selected_packages:
            break

        selected_data = data.iloc[selected_packages]
        job_nos = selected_data[find_job_no_column(data)].tolist()
        days_remaining = selected_data['DAYS_REMAINING'].tolist()
        total_cost = calculate_cost(total_weight_used, total_volume_used) if include_cost else None

        if (not (
                (4.5 <= total_weight_used <= 6) or
                (7 <= total_weight_used <= 9) or
                (10 <= total_weight_used <= 15) or
                (16 <= total_weight_used <= 18) or
                (20 <= total_weight_used <= 24)
            )) or total_cost == 0:
            unfulfilled_due_to_volume = pd.concat([unfulfilled_due_to_volume, selected_data])
        else:
            # Include all columns from the selected data
            console_df = pd.DataFrame(selected_data)
            console_df.insert(0, 'Console', f"Console {len(packages) + 1}")

            packages.append({
                'Console Data': console_df,
                'Total Weight': round(total_weight_used, 2),
                'Total Volume': round(total_volume_used, 2),
                'Total Cost': total_cost
            })

        data = data.drop(selected_packages).reset_index(drop=True)

    return packages, unfulfilled_due_to_volume


# Function to generate a report based on different weight classes and carry volumes
def generate_report(data, weight_classes, selected_volume_class, dcd_upper_limit, include_cost=True):
    fulfilled_files = []
    unfulfilled_files = []

    weight_class_ranges = {
        '6 Tones': [4.5, 6],
        '9 Tones': [7, 9],
        '15 Tones': [10, 15],
        '18 Tones': [16, 18],
        '24 Tones': [20, 24]
    }

    volume_class = get_volume_class(selected_volume_class, dcd_upper_limit)

    if 'ALL' in weight_classes:
        selected_weight_classes = list(weight_class_ranges.keys())
    else:
        selected_weight_classes = weight_classes

    report = []

    for weight_class in selected_weight_classes:
        weight_range = weight_class_ranges[weight_class]

        packages, unfulfilled_packages = create_packages(data, weight_range[-1], volume_class, include_cost)

        fulfilled_files.extend(packages)
        unfulfilled_files.append((weight_class, unfulfilled_packages))

        summary_df = pd.DataFrame({
            'Weight Class': [weight_class],
            'Fulfilled Consoles': [len(packages)],
            'Unfulfilled Jobs': [len(unfulfilled_packages)],
            'Unfulfilled Total Weight': [round(unfulfilled_packages['WEIGHT_TONS'].sum(), 2) if not unfulfilled_packages.empty else 0],
            'Unfulfilled Total Volume': [round(unfulfilled_packages['CBM'].sum(), 2) if not unfulfilled_packages.empty else 0]
        })

        report.append({
            'Summary': summary_df,
            'Fulfilled Consoles': packages,
        })

    unfulfilled_report = []
    for weight_class, unfulfilled_df in unfulfilled_files:
        if not unfulfilled_df.empty:
            unfulfilled_report.append({
                'Weight Class': weight_class,
                'Unfulfilled Consoles': unfulfilled_df
            })

    return report, unfulfilled_report

# Streamlit UI code
st.title('Container Stuffing Optimization')
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file:
    file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
    data = load_data(uploaded_file, file_type)
    data = convert_weights(data)  # Convert weights to metric tons
    data = calculate_days_remaining(data)  # Calculate days remaining
    
    st.write("Data preview:")
    st.dataframe(data.head())

    volume_class = st.selectbox(
        "Select Volume Class",
        ['DCD']
    )

    dcd_upper_limit = 58
    if volume_class == 'DCD':
        dcd_upper_limit = st.number_input("Enter the upper limit for DCD volume (CBM)", min_value=1, value=58)

    weight_classes = st.multiselect(
        "Select Weight Classes",
        ['6 Tones', '9 Tones', '15 Tones', '18 Tones', '24 Tones', 'ALL']
    )

    include_cost = st.checkbox("Include Cost in Report", value=True)

    if st.button("Generate Report"):
        report, unfulfilled_report = generate_report(data, weight_classes, volume_class, dcd_upper_limit, include_cost)

        for section in report:
            st.subheader(f"Summary for {section['Summary']['Weight Class'][0]}")
            st.dataframe(section['Summary'])
            for i, package in enumerate(section['Fulfilled Consoles']):
                st.write(f"**Console {i + 1}:**")
                st.dataframe(package['Console Data'])
                st.write(f"Total Weight: {package['Total Weight']} Tons")
                st.write(f"Total Volume: {package['Total Volume']} CBM")
                if include_cost:
                    st.write(f"Total Cost: {package['Total Cost']}")

        st.subheader("Unfulfilled Consoles")
        for report_item in unfulfilled_report:
            st.write(f"**Weight Class: {report_item['Weight Class']}**")
            st.dataframe(report_item['Unfulfilled Consoles'])

# Functions for managing consoles and unfulfilled jobs
def manage_consoles(consoles, unfulfilled_jobs):
    # Create a dictionary to store console DataFrames
    console_dfs = {f'Console {i + 1}': console['Console Data'] for i, console in enumerate(consoles)}
    
    # Display the consoles and their totals
    for console_name, df in console_dfs.items():
        total_weight = df['Weight (Tons)'].sum()
        total_volume = df['Volume (CBM)'].sum()
        st.write(f"{console_name}: Total Weight = {total_weight:.2f} Tons, Total Volume = {total_volume:.2f} CBM")
    
    # User input to select console for modifications
    selected_console = st.selectbox("Select Console to Modify", list(console_dfs.keys()))
    
    # Options for adding or removing jobs
    st.subheader(f"Manage {selected_console}")
    selected_action = st.radio("Choose action", ["Add Jobs", "Remove Jobs"])

    if selected_action == "Add Jobs":
        if not unfulfilled_jobs.empty:
            st.write("Unfulfilled Jobs:")
            st.dataframe(unfulfilled_jobs)
            job_to_add = st.selectbox("Select Job to Add", unfulfilled_jobs['Job No'].tolist())
            if st.button("Add Job"):
                job_data = unfulfilled_jobs[unfulfilled_jobs['Job No'] == job_to_add]
                console_dfs[selected_console] = pd.concat([console_dfs[selected_console], job_data])
                unfulfilled_jobs = unfulfilled_jobs[unfulfilled_jobs['Job No'] != job_to_add]
                st.write("Job added successfully.")
                st.write(f"Updated {selected_console}:")
                total_weight = console_dfs[selected_console]['Weight (Tons)'].sum()
                total_volume = console_dfs[selected_console]['Volume (CBM)'].sum()
                st.write(f"Total Weight = {total_weight:.2f} Tons, Total Volume = {total_volume:.2f} CBM")

    elif selected_action == "Remove Jobs":
        if not console_dfs[selected_console].empty:
            st.write("Jobs in Console:")
            st.dataframe(console_dfs[selected_console])
            job_to_remove = st.selectbox("Select Job to Remove", console_dfs[selected_console]['Job No'].tolist())
            if st.button("Remove Job"):
                job_data = console_dfs[selected_console][console_dfs[selected_console]['Job No'] == job_to_remove]
                console_dfs[selected_console] = console_dfs[selected_console][console_dfs[selected_console]['Job No'] != job_to_remove]
                unfulfilled_jobs = pd.concat([unfulfilled_jobs, job_data])
                st.write("Job removed successfully.")
                st.write(f"Updated {selected_console}:")
                total_weight = console_dfs[selected_console]['Weight (Tons)'].sum()
                total_volume = console_dfs[selected_console]['Volume (CBM)'].sum()
                st.write(f"Total Weight = {total_weight:.2f} Tons, Total Volume = {total_volume:.2f} CBM")

    return console_dfs, unfulfilled_jobs

