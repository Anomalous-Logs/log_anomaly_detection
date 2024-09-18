import requests
import json
import pandas as pd
import time  # Thư viện time để lấy thời gian hiện tại

BASE_URL = 'http://10.20.7.251:9090/api/v1/query_range?'


class Query:
    def json_to_dataframe(self, query: str, start = None, end = None, step = None) -> pd.DataFrame:
        full_url = Query.get_url(query, start, end, step)
        print(full_url)
        response = requests.get(full_url)
        
        if response.status_code == 200:
            json_data = response.json()
            df = pd.DataFrame(json_data['data']['result'])
            df['metric'] = df['metric'].apply(lambda x: x['instance'])
            return df
        else:
            raise Exception(f"Request failed with status code {response.status_code}")

    def get_url(query: str, start = None, end = None, step = None) -> str:
        if end == None:
            end = round(time.time())
        if start == None:
            start = end - 3600*24
        if start < 0:
            start = end + start
        if step == None:
            step = 300
        return f'{BASE_URL}query={query}&start={start}&end={end}&step={step}'


class BlackboxProbe(Query):
    def probe_failed(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'probe_success'
        return self.json_to_dataframe(query, start, end, step)

    def probe_http_failure(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'probe_http_status_code'
        return self.json_to_dataframe(query, start, end, step)

    def ssl_certificate_will_expire_soon_warning(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'last_over_time(probe_ssl_earliest_cert_expiry[10m]) - time()'
        return self.json_to_dataframe(query, start, end, step)

    def ssl_certificate_will_expire_soon_critical(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'last_over_time(probe_ssl_earliest_cert_expiry[10m]) - time()'
        return self.json_to_dataframe(query, start, end, step)

    def ssl_certificate_expired(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'last_over_time(probe_ssl_earliest_cert_expiry[10m]) - time()'
        return self.json_to_dataframe(query, start, end, step)

    def slow_http_probe(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'avg_over_time(probe_http_duration_seconds[1m])'
        return self.json_to_dataframe(query, start, end, step)

class CpuUsageQuery(Query):
    def high_cpu_usage(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'sum by (instance) (avg by (mode, instance) (rate(node_cpu_seconds_total{mode!="idle"}[2m])))'
        return self.json_to_dataframe(query, start, end, step)

    def windows_server_cpu_usage(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'avg by (instance) (rate(windows_cpu_time_total{mode="idle"}[2m]))'
        return self.json_to_dataframe(query, start, end, step)

class DiskUsageQuery(Query):
    def host_out_of_disk_space(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'node_filesystem_avail_bytes / node_filesystem_size_bytes'
        return self.json_to_dataframe(query, start, end, step)

class MemoryUsageQuery(Query):
    def host_out_of_memory(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes'
        return self.json_to_dataframe(query, start, end, step)

    def windows_server_memory_usage(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'windows_os_physical_memory_free_bytes / windows_cs_physical_memory_bytes'
        return self.json_to_dataframe(query, start, end, step)

class NetworkThroughputQuery(Query):
    def unusual_network_throughput_in(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'sum by (instance) (rate(node_network_receive_bytes_total[2m]))'
        return self.json_to_dataframe(query, start, end, step)

    def unusual_network_throughput_out(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'sum by (instance) (rate(node_network_transmit_bytes_total[2m]))'
        return self.json_to_dataframe(query, start, end, step)
    
class PrometheusMonitoringQuery(Query):
    def prometheus_target_missing(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'up'
        return self.json_to_dataframe(query, start, end, step)

class CollectorErrorQuery(Query):
    def windows_server_collector_error(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'windows_exporter_collector_success'
        return self.json_to_dataframe(query, start, end, step)

class WindowsServerServiceQuery(Query):
    def windows_server_collector_error(self, start = None, end = None, step = None) -> pd.DataFrame:
        query = 'windows_exporter_collector_success'
        return self.json_to_dataframe(query, start, end, step)


# Test phương thức 
"""
try:
    
    end_time = int(time.time())
    start_time = end_time - 3600  
    step = '60'
    cpu_query = WindowsServerServiceQuery()
    df_high_cpu_usage = cpu_query.windows_server_collector_error(start_time, end_time, step)
    print("High CPU Usage:")
    print(df_high_cpu_usage)

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
"""

if __name__ == "__main__":
    bb = BlackboxProbe()
    df = bb.probe_failed(start=-3600)
    breakpoint()
