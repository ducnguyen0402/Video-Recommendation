# Hướng dẫn cài đặt PySpark và triển khai code hệ thống gợi ý (ALS)

---

## 1. Tổng quan

Tài liệu này hướng dẫn chi tiết:

* Cài đặt môi trường chạy PySpark
* Cấu hình Apache Spark ở chế độ local
* Chuẩn bị dataset MovieLens (ml-20m)
* Upload và chạy code huấn luyện mô hình ALS

Phù hợp với môi trường Linux (Ubuntu) và server Docker như bạn đang sử dụng.

---

## 2. Yêu cầu hệ thống

### 2.1 Phần mềm cần thiết

* Python >= 3.8
* Java (OpenJDK 8 hoặc 11)
* Apache Spark
* pip (Python package manager)

---

## 3. Cài đặt môi trường

### 3.1 Cài đặt Java

```bash
sudo apt update
sudo apt install -y openjdk-11-jdk
```

Kiểm tra:

```bash
java -version
```

---

### 3.2 Cài đặt Python và pip

```bash
sudo apt install -y python3 python3-pip
```

Kiểm tra:

```bash
python3 --version
pip3 --version
```

---

### 3.3 Cài đặt PySpark

```bash
pip3 install pyspark
```

Kiểm tra:

```bash
python3 -c "import pyspark; print(pyspark.__version__)"
```

---

### 3.4 (Tùy chọn) Cài đặt Apache Spark standalone

Nếu muốn chạy spark-submit độc lập:

```bash
wget https://downloads.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xvzf spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 ~/spark
```

Cấu hình biến môi trường:

```bash
nano ~/.bashrc
```

Thêm:

```bash
export SPARK_HOME=~/spark
export PATH=$SPARK_HOME/bin:$PATH
```

Apply:

```bash
source ~/.bashrc
```

Kiểm tra:

```bash
spark-submit --version
```

---

## 4. Chuẩn bị dữ liệu

### 4.1 Tải dataset MovieLens ml-20m

```bash
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
```

### 4.2 Cấu trúc thư mục

```bash
project/
│
├── data/
│   └── ml-20m/
│       ├── ratings.csv
│       ├── movies.csv
│
├── src/
│   └── train_als.py
│
└── output/
```

---

## 5. Upload code lên server

### 5.1 Dùng SCP (từ máy local)

```bash
scp -r project/ user@server-ip:/home/user/
```

### 5.2 Dùng Git (khuyến nghị)

```bash
git clone <repo-url>
cd project
```

---

## 6. Code huấn luyện ALS

Tạo file:

```bash
nano src/train_als.py
```

Dán code:

```python
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode

spark = SparkSession.builder \
    .appName("Video Recommendation") \
    .master("local[*]") \
    .config("spark.driver.host", "0.0.0.0") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .getOrCreate()

ratings = spark.read.csv("data/ml-20m/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("data/ml-20m/movies.csv", header=True, inferSchema=True)

train, test = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop"
)

model = als.fit(train)

predictions = model.transform(test)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE =", rmse)

output_path = "./output/rmse.txt"
with open(output_path, "w") as f:
    f.write(f"RMSE: {rmse:.4f}\n")

user_recs = model.recommendForAllUsers(10)

recs = user_recs \
    .withColumn("rec", explode("recommendations")) \
    .select("userId", "rec.movieId", "rec.rating")

final = recs.join(movies, "movieId")

final.show(20, truncate=False)

input("Press Enter to exit...")
```

---

## 7. Chạy chương trình

### 7.1 Chạy bằng Python trực tiếp

```bash
cd project
python3 src/train_als.py
```

---

### 7.2 Chạy bằng spark-submit (khuyến nghị)

```bash
spark-submit src/train_als.py
```

---

## 8. Kiểm tra kết quả

### 8.1 Output console

* RMSE sẽ được in ra màn hình
* Danh sách gợi ý phim hiển thị

### 8.2 File kết quả

```bash
cat output/rmse.txt
```

---

## 9. Một số lỗi thường gặp

### 9.1 Lỗi không tìm thấy Java

Fix:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

---

### 9.2 Lỗi thiếu RAM khi chạy ml-20m


Giảm cấu hình:

```python
rank=5
maxIter=5
```

Hoặc giới hạn core:

```python
.master("local[4]")
```

---

### 9.3 Lỗi đường dẫn dữ liệu


```bash
data/ml-20m/ratings.csv
```

---

## 10. Gợi ý tối ưu (nâng cao)

* Cache dữ liệu:

```python
ratings.cache()
```

* Tune tham số ALS:

  * rank
  * regParam
  * maxIter

* Chạy trên cluster thay vì local

---

## 11. Kết luận

Hệ thống đã:

* Cài đặt thành công PySpark
* Xử lý dataset lớn (ml-20m)
* Huấn luyện mô hình ALS
* Sinh gợi ý phim theo người dùng

Đây là nền tảng để phát triển các hệ thống recommendation thực tế trong production.

---
