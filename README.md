# log_anomaly_detection

Dữ liệu tải xuống từ [https://www.unb.ca/cic/datasets/cic-unsw-nb15.html](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html)

Giải nén vào thư mục data

Trích 300000 dòng đầu tiên của file bằng lệnh:
```
head -n300000 data/CICFlowMeter_out.csv > data/CICFlowMeter_out.300000head.csv
```

Tạo và kích hoạt môi trường, thực hiện với python 3.10 hoặc mới hơn
```
python -m venv .venv
. .venv/bin/activate
```

Cài các gói cần thiết để thực thi
```
pip install -r requirements.txt
```

Thực thi script thống kê đánh giá các mô hình đã xét, các model sẽ được lưu vào thư mục model, và file log sẽ được sinh ra tại compare.log
```
python eval/compare.py
```