import pandas as pd
from prettytable import PrettyTable
from loguru import logger

def operator_rank(csv_path:str):
    """
    打开operator_details.csv文件，并按耗时将算子排序
    算子下发为aten::A, aten::B
    A下发B，那么A-B的耗时是Self Host Duration, A 的耗时是 Total Host Duration
    """
    df=pd.read_csv(csv_path)
    grouped_sum = df.groupby(['Name'])[['Device Total Duration(us)', 'Host Total Duration(us)']].sum().reset_index()
    grouped_sum["Total Duration"]=grouped_sum["Device Total Duration(us)"]+grouped_sum["Host Total Duration(us)"]
    grouped_sum['Rank'] = grouped_sum['Total Duration'].rank(ascending=False)
    grouped_sum = grouped_sum.sort_values(by='Rank')

    total_device_duration=df['Device Total Duration(us)'].sum()
    total_host_duration=df['Host Total Duration(us)'].sum()
    total_duration=total_device_duration+total_host_duration

    table = PrettyTable()
    table.field_names = grouped_sum.columns
    for row in grouped_sum.itertuples(index=False, name=None):
        table.add_row(row)
    logger.info(f"Total Duration: {total_duration}  Device Total Duration(us): {total_device_duration}  Host Total Duration(us): {total_host_duration}\n{table}")


if __name__=="__main__":
    operator_rank(
        "/home/guozr/CODE/AMP/supported_models/codegeex4-all-9b/prof/bms-openmind_3122330_20241206084328185_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv"
    )